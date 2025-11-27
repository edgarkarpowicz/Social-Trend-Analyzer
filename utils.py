#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD BLAS PASCAL
PROGRAMACIÓN DECLARATIVA - 2025
TOMAS MOLINA Y EDGAR KARPOWICZ
============================================
CONSIGNA GENERAL:

Análisis de Redes Sociales y Detección de Tendencias

Dado un dataset de publicaciones (usuario, texto, hashtags, likes):
Procesar datos con pandas (hashtags más usados, usuarios más activos).
Usar NumPy para calcular métricas de popularidad y engagement.
Graficar con Matplotlib las tendencias de hashtags en el tiempo.
Entrenar un modelo de IA (NLP básico con TF-IDF + clustering o regresión logística) para clasificar
publicaciones entre “alta” o “baja” relevancia.

INCLUIR INTERFACE GRÁFICA
============================================
utils.py - Social Trend Analyzer
Utilidades y Validaciones
--------------------------------------------
Descripción de utils.py:
Funciones auxiliares para validación/limpieza de datasets, parseo de hashtags,
conversión de fechas, generación de etiquetas y validaciones de entradas.
============================================
Librerías utilizadas:
- pandas/numpy: Limpieza y transformación de datos.
- tkinter: Diálogos de error.
"""

import ast
import logging

import numpy as np
import pandas as pd
from tkinter import messagebox
from typing import List, Tuple
from config import REQUIRED_COLUMNS, MIN_ROWS_REQUIRED, MIN_CLASSES_REQUIRED


# ---------------------------
# Manejo de Errores
# ---------------------------

def show_error(title: str, message: str) -> None:
    """Muestra un mensaje de error usando messagebox o print como fallback."""
    try:
        messagebox.showerror(title, message)
    except Exception as e:
        logging.exception("Error showing message box")
        print(f"ERROR [{title}]: {message}")


# ---------------------------
# Validación de Dataset
# ---------------------------

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Valida el dataset y devuelve (es_valido, lista_de_mensajes, dataframe_modificado).
    
    Reglas de validación:
    - Columnas mínimas requeridas
    - Tipos y parseos (likes numérico, timestamp válido)
    - Filas suficientes para entrenamiento
    - Al menos dos clases en 'label' tras normalización/autoetiquetado
    
    Args:
        df: DataFrame a validar
        
    Returns:
        Tuple[bool, List[str], pd.DataFrame]: (es_válido, lista_de_mensajes, dataframe_limpio)
    """
    msgs: List[str] = []
    ok = True

    # Verificar columnas obligatorias
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        ok = False
        msgs.append(f"Faltan columnas obligatorias: {', '.join(sorted(missing))}")
        return ok, msgs, df

    # Validar y convertir tipos de datos
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    n_likes_na = df["likes"].isna().sum()
    if n_likes_na > 0:
        msgs.append(f"Se detectaron {n_likes_na} 'likes' no numéricos; se marcaron como NaN y se descartarán.")

    df["timestamp"] = ensure_datetime(df["timestamp"])
    n_ts_na = df["timestamp"].isna().sum()
    if n_ts_na > 0:
        msgs.append(f"Se detectaron {n_ts_na} 'timestamp' inválidos; se descartarán.")

    # Normalizar hashtags a lista
    df["hashtags_list"] = df["hashtags"].apply(parse_hashtags_cell)

    # Auto-etiquetar si falta la columna label
    df = build_label_if_missing(df)

    # Eliminar filas con datos inválidos
    df = df.dropna(subset=["timestamp", "likes"]).reset_index(drop=True)

    # Verificar cantidad mínima de filas
    if len(df) < MIN_ROWS_REQUIRED:
        ok = False
        msgs.append(
            f"El dataset tiene {len(df)} filas válidas; se requieren al menos {MIN_ROWS_REQUIRED} para entrenar.")

    # Verificar cantidad mínima de clases
    if "label" in df.columns:
        cls_counts = df["label"].value_counts()
        if len(cls_counts) < MIN_CLASSES_REQUIRED:
            ok = False
            msgs.append("Se requiere al menos 2 clases en 'label' (alta/baja) tras limpieza.")
        else:
            min_cls = int(cls_counts.min())
            if min_cls < 5:
                msgs.append(
                    "Advertencia: una de las clases tiene menos de 5 ejemplos; el modelo puede generalizar mal.")

    return ok, msgs, df


# ---------------------------
# Preprocesado de Hashtags
# ---------------------------

HASHTAG_SEPARATORS = [",", ";", "|", "/", " "]


def parse_hashtags_cell(cell) -> List[str]:
    """
    Normaliza un campo de hashtags y devuelve una lista de strings sin '#'.
    
    Acepta múltiples formatos:
    - "#python #ai" 
    - "['python','ai']"
    - "python, ai"
    - etc.
    
    Args:
        cell: Celda con hashtags en cualquier formato
        
    Returns:
        List[str]: Lista de hashtags normalizados (sin #, en minúsculas)
    """
    if pd.isna(cell):
        return []

    if isinstance(cell, list):
        return [h.lstrip("#").strip().lower() for h in cell if isinstance(h, str)]

    s = str(cell).strip()
    if not s:
        return []

    # Si parece lista Python, intentar literal_eval
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(h).lstrip("#").strip().lower() for h in parsed]
        except Exception as e:
            logging.debug(f"Could not parse hashtags as literal: {e}")
            pass

    # Reemplazar '#' por espacios para facilitar split si viene pegado
    s = s.replace("#", " ")

    # Normalizar separadores múltiples por coma
    for sep in HASHTAG_SEPARATORS:
        if sep != ",":
            s = s.replace(sep, ",")

    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return parts


# ---------------------------
# Preprocesado de Fechas
# ---------------------------

def ensure_datetime(series: pd.Series) -> pd.Series:
    """
    Convierte una serie a datetime, marcando valores inválidos como NaT.
    
    Args:
        series: Serie de pandas con fechas en formato string
        
    Returns:
        pd.Series: Serie convertida a datetime
    """
    return pd.to_datetime(series, errors="coerce")


# ---------------------------
# Generación de Labels
# ---------------------------

def build_label_if_missing(df: pd.DataFrame, likes_col: str = "likes") -> pd.DataFrame:
    """
    Construye la columna 'label' si no existe, usando el percentil 70 de likes.
    Si existe, normaliza los valores a {'alta', 'baja'}.
    
    Args:
        df: DataFrame a procesar
        likes_col: Nombre de la columna de likes
        
    Returns:
        pd.DataFrame: DataFrame con columna 'label' normalizada
    """
    df = df.copy()

    if "label" in df.columns:
        # Normalizar a {'alta','baja'} si está en otros formatos
        def normalize_label(x):
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in {"alta", "high", "relevante", "relevancia_alta"}:
                    return "alta"
                if xs in {"baja", "low", "no_relevante", "relevancia_baja"}:
                    return "baja"
            return None

        df["label"] = df["label"].apply(normalize_label)
        # Quitar filas con label inválida
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        return df

    # Si no existe 'label', crearla por percentil 70 de likes
    likes = pd.to_numeric(df[likes_col], errors="coerce")
    # Si todos los likes son NaN, retornar df sin cambios (validación posterior manejará)
    try:
        threshold = np.nanpercentile(likes, 70)
    except Exception:
        threshold = np.nan

    # Asignar 'alta' si likes >= percentil 70, caso contrario 'baja'
    df["label"] = np.where(likes >= threshold, "alta", "baja")
    return df


# ---------------------------
# Composición de Features de Texto
# ---------------------------

def compose_text_features(df: pd.DataFrame) -> pd.Series:
    """
    Une texto + hashtags normalizados para alimentar TF-IDF.
    
    Args:
        df: DataFrame con columnas 'text' y 'hashtags_list'
        
    Returns:
        pd.Series: Serie con texto combinado para análisis NLP
    """
    hashtags_joined = df["hashtags_list"].apply(lambda xs: " ".join([f"#{x}" for x in xs]))
    text = df["text"].fillna("")
    return (text + " " + hashtags_joined).str.strip()


# ---------------------------
# Validación de Entrada de Usuario
# ---------------------------

def validate_likes_input(likes_str: str) -> Tuple[bool, float]:
    """
    Valida la entrada de likes del usuario.
    
    Args:
        likes_str: String con el número de likes
        
    Returns:
        Tuple[bool, float]: (es_válido, valor_numérico)
    """
    try:
        likes_val = float(likes_str.strip())
        if likes_val < 0:
            return False, 0.0
        return True, likes_val
    except (ValueError, AttributeError):
        return False, 0.0


def validate_text_input(text: str) -> bool:
    """
    Valida que el texto de entrada no esté vacío.
    
    Args:
        text: Texto a validar
        
    Returns:
        bool: True si el texto es válido
    """
    return bool(text and text.strip())


# ---------------------------
# Utilidades de Formateo
# ---------------------------

def format_number(num: float, decimals: int = 2) -> str:
    """
    Formatea un número con la cantidad especificada de decimales.
    
    Args:
        num: Número a formatear
        decimals: Cantidad de decimales
        
    Returns:
        str: Número formateado
    """
    return f"{num:.{decimals}f}"


def format_percentage(num: float, decimals: int = 1) -> str:
    """
    Formatea un número como porcentaje.
    
    Args:
        num: Número a formatear (0.85 -> 85.0%)
        decimals: Cantidad de decimales
        
    Returns:
        str: Porcentaje formateado
    """
    return f"{num * 100:.{decimals}f}%"


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Trunca un texto si excede la longitud máxima.
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        
    Returns:
        str: Texto truncado con "..." si es necesario
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

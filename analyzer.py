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
analyzer.py - Social Trend Analyzer
Análisis y Modelado
--------------------------------------------
Descripción de analyzer.py:
Funciones para estadísticas básicas, tendencias de hashtags y entrenamiento/predicción
de un modelo TF-IDF + Regresión Logística.
============================================
Librerías utilizadas:
- pandas/numpy: Procesamiento numérico y de datos.
- scikit-learn: Vectorización TF-IDF, Regresión Logística y métricas.
- utils: Composición de features de texto.
============================================
FLUJO BÁSICO DEL MÓDULO (analyzer.py):
--------------------------------------------
Este archivo es el CEREBRO ANALÍTICO del proyecto, encargado de procesar datos
estadísticos y entrenar/usar el modelo de Machine Learning.

¿Qué hace este módulo?
1. ANÁLISIS ESTADÍSTICO BÁSICO (compute_basic_stats):
   - Identifica usuarios más activos (por cantidad de posts)
   - Calcula hashtags más populares
   - Computa métricas de engagement usando NumPy (media, mediana, desviación)
   - Crea un índice de engagement: likes / longitud_texto
   - Retorna todo en DataFrames listos para mostrar en la UI

2. ANÁLISIS DE TENDENCIAS TEMPORALES (compute_hashtag_trend):
   - Agrupa posts por fecha y hashtag
   - Crea una tabla pivot (fecha × hashtag) con conteos diarios
   - Enfoca en los top K hashtags más usados
   - Permite visualizar evolución temporal de tendencias

3. PATRONES DE ENGAGEMENT (analyze_engagement_patterns):
   - Analiza engagement por hora del día y día de la semana
   - Calcula correlación entre longitud de texto y likes
   - Identifica hashtags más efectivos (mayor promedio de likes)

4. MACHINE LEARNING - ENTRENAMIENTO (train_text_model):
   FLUJO COMPLETO:
   a) Prepara features: combina texto + hashtags usando compose_text_features()
   b) Split estratificado 75/25 (train/test) manteniendo balance de clases
   c) Crea Pipeline: TF-IDF (1-2 grams, max 20k features) → Regresión Logística
   d) Entrena modelo con class_weight='balanced' para manejar desbalance
   e) Evalúa en test set: accuracy, classification report, confusion matrix
   f) Formatea métricas para mostrar en UI de forma profesional

5. MACHINE LEARNING - PREDICCIÓN (predict_with_model):
   - Toma un nuevo texto + hashtags + likes
   - Prepara el texto igual que en entrenamiento
   - Usa el modelo entrenado para predecir: "alta" o "baja" relevancia
   - Retorna predicción + nivel de confianza (probabilidad)

6. FORMATEO DE MÉTRICAS (format_model_metrics):
   - Genera reporte visual profesional con:
     * Accuracy general
     * Tabla de precision/recall/f1-score por clase
     * Matriz de confusión
     * Interpretación inteligente según performance
     * Explicación didáctica de qué significa cada métrica

7. ANÁLISIS DE FEATURES (analyze_model_features):
   - Identifica las palabras/tokens más importantes para cada clase
   - Extrae coeficientes del modelo para explicabilidad

CONTRIBUCIÓN AL PROYECTO:
Este módulo es el CORAZÓN del análisis de datos e inteligencia artificial.
Transforma datos crudos en insights accionables (estadísticas, tendencias)
y entrena un modelo NLP capaz de predecir relevancia de posts.
Es donde se cumple la consigna principal: "entrenar un modelo de IA (NLP básico
con TF-IDF + regresión logística) para clasificar publicaciones".
============================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from config import MODEL_CONFIG, ANALYSIS_CONFIG
from utils import compose_text_features


# ---------------------------
# Análisis Estadístico Básico
# ---------------------------

def compute_basic_stats(df: pd.DataFrame, top_n: int = None) -> Dict[str, pd.DataFrame]:
    """
    Calcula estadísticas básicas del dataset incluyendo usuarios más activos,
    hashtags más usados y métricas de engagement.
    
    Args:
        df: DataFrame con los datos procesados
        top_n: Número de elementos top a mostrar (default: ANALYSIS_CONFIG)
        
    Returns:
        Dict con DataFrames de estadísticas: users_active, hashtags_top, summary
    """
    if top_n is None:
        top_n = ANALYSIS_CONFIG["top_n_default"]

    # Usuarios más activos (por cantidad de posts)
    users_active = (df.groupby("user")["text"].count()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .rename("posts")
                    .reset_index())

    # Hashtags más usados
    all_tags = df["hashtags_list"].explode()
    hashtags_top = (all_tags.value_counts()
                    .head(top_n)
                    .rename_axis("hashtag")
                    .reset_index(name="count"))

    # Métricas NumPy de popularidad/engagement
    likes = df["likes"].to_numpy(dtype=float)

    # Engagement simple: likes por longitud de texto (evita división por cero)
    text_len = df["text"].fillna("").str.split().apply(len).to_numpy(dtype=float)
    engagement = likes / np.maximum(text_len, 1.0)

    summary = pd.DataFrame({
        "likes_mean": [np.nanmean(likes)],
        "likes_median": [np.nanmedian(likes)],
        "likes_std": [np.nanstd(likes)],
        "engagement_mean": [np.nanmean(engagement)],
    })

    # Guardar engagement en el DataFrame principal
    df["engagement"] = engagement

    return {
        "users_active": users_active,
        "hashtags_top": hashtags_top,
        "summary": summary,
    }


# ---------------------------
# Análisis de Tendencias
# ---------------------------

def compute_hashtag_trend(df: pd.DataFrame, top_k: int = None) -> pd.DataFrame:
    """
    Calcula las tendencias diarias de hashtags para los más populares.
    
    Args:
        df: DataFrame con datos procesados
        top_k: Número de hashtags top a analizar (default: ANALYSIS_CONFIG)
        
    Returns:
        DataFrame pivot (fecha x hashtag) con conteos diarios
    """
    if top_k is None:
        top_k = ANALYSIS_CONFIG["top_k_hashtags"]

    # Tomar fecha (día) de timestamp
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    # Top k hashtags más usados
    top_tags = df["hashtags_list"].explode().value_counts().head(top_k).index.tolist()

    # Expandir filas (una por hashtag)
    exploded = df.explode("hashtags_list")
    exploded = exploded[exploded["hashtags_list"].isin(top_tags)]

    # Agrupar por fecha y hashtag
    daily = (exploded.groupby(["date", "hashtags_list"]).size()
             .rename("count").reset_index())

    # Crear tabla pivot
    pivot = daily.pivot(index="date", columns="hashtags_list", values="count").fillna(0).astype(int)
    pivot = pivot.sort_index()

    return pivot


def analyze_engagement_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza patrones de engagement en el dataset.
    
    Args:
        df: DataFrame con datos procesados
        
    Returns:
        Dict con análisis de patrones de engagement
    """
    # Análisis por hora del día
    df_copy = df.copy()
    df_copy["hour"] = df_copy["timestamp"].dt.hour
    hourly_engagement = df_copy.groupby("hour")["likes"].mean().sort_index()

    # Análisis por día de la semana
    df_copy["weekday"] = df_copy["timestamp"].dt.day_name()
    weekly_engagement = df_copy.groupby("weekday")["likes"].mean()

    # Correlación entre longitud de texto y likes
    df_copy["text_length"] = df_copy["text"].fillna("").str.len()
    text_likes_corr = df_copy["text_length"].corr(df_copy["likes"])

    # Hashtags más efectivos (promedio de likes)
    hashtag_effectiveness = []
    for _, row in df_copy.iterrows():
        for hashtag in row["hashtags_list"]:
            hashtag_effectiveness.append({"hashtag": hashtag, "likes": row["likes"]})

    if hashtag_effectiveness:
        hashtag_df = pd.DataFrame(hashtag_effectiveness)
        top_effective_hashtags = (hashtag_df.groupby("hashtag")["likes"]
                                  .agg(["mean", "count"])
                                  .query("count >= 3")  # Al menos 3 usos
                                  .sort_values("mean", ascending=False)
                                  .head(10))
    else:
        top_effective_hashtags = pd.DataFrame()

    return {
        "hourly_engagement": hourly_engagement,
        "weekly_engagement": weekly_engagement,
        "text_likes_correlation": text_likes_corr,
        "effective_hashtags": top_effective_hashtags
    }


# ---------------------------
# Machine Learning
# ---------------------------

def train_text_model(df: pd.DataFrame, random_state: int = None) -> Tuple[Any, str, float, np.ndarray]:
    """
    Entrena un modelo TF-IDF + Regresión Logística para clasificar relevancia alta/baja.
    
    Args:
        df: DataFrame con datos procesados y columna 'label'
        random_state: Semilla para reproducibilidad (default: MODEL_CONFIG)
        
    Returns:
        Tuple: (pipeline_entrenado, métricas_str, accuracy_float, confusion_matrix_array)
    """
    if random_state is None:
        random_state = MODEL_CONFIG["random_state"]

    # Preparar features y target
    x_train = compose_text_features(df)
    y = df["label"].astype(str)

    # Split de datos
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=random_state,
        stratify=y
    )

    # Crear pipeline
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MODEL_CONFIG["max_features"],
            ngram_range=MODEL_CONFIG["ngram_range"]
        )),
        ("lr", LogisticRegression(
            max_iter=MODEL_CONFIG["max_iter"],
            class_weight=MODEL_CONFIG["class_weight"]
        ))
    ])

    # Entrenar modelo
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["alta", "baja"])  # orden fijo

    # Formatear métricas
    metrics_text = format_model_metrics(acc, report, cm)

    return clf, metrics_text, acc, cm


def predict_with_model(model: Any, text: str, hashtags: str = "", likes: float = 0) -> Tuple[str, float]:
    """
    Realiza una predicción con el modelo entrenado.
    
    Args:
        model: Modelo entrenado
        text: Texto del post
        hashtags: Hashtags del post
        likes: Número de likes (no usado en predicción, solo para contexto)
        
    Returns:
        Tuple: (predicción, confianza)
    """
    # Preparar texto combinado
    if hashtags:
        # Normalizar hashtags
        hashtag_list = [h.strip().lstrip("#") for h in hashtags.split() if h.strip()]
        hashtag_text = " ".join([f"#{h}" for h in hashtag_list])
        combined_text = f"{text} {hashtag_text}".strip()
    else:
        combined_text = text.strip()

    # Realizar predicción
    prediction = model.predict([combined_text])[0]
    probabilities = model.predict_proba([combined_text])[0]

    # Obtener confianza (probabilidad de la clase predicha)
    classes = model.classes_
    pred_idx = list(classes).index(prediction)
    confidence = probabilities[pred_idx]

    return prediction, confidence


# ---------------------------
# Formateo de Métricas
# ---------------------------

def format_model_metrics(acc: float, report: str, cm: np.ndarray) -> str:
    """
    Formatea las métricas del modelo de manera estructurada y legible.
    
    Args:
        acc: Accuracy del modelo
        report: Classification report de sklearn
        cm: Confusion matrix
        
    Returns:
        str: Métricas formateadas para mostrar en la UI
    """
    # Extraer información del classification report
    lines = report.strip().split('\n')

    # Buscar las líneas de alta y baja
    alta_line = None
    baja_line = None

    for line in lines:
        if line.strip().startswith('alta'):
            alta_line = line.strip().split()
        elif line.strip().startswith('baja'):
            baja_line = line.strip().split()

    formatted_text = "═══════════════════════════════════════════════════════════════\n"
    formatted_text += "                    MÉTRICAS DEL MODELO                        \n"
    formatted_text += "═══════════════════════════════════════════════════════════════\n\n"

    # Accuracy principal
    formatted_text += f"🎯 ACCURACY GENERAL: {acc:.3f} ({acc * 100:.1f}%)\n\n"

    # Tabla de métricas por clase
    formatted_text += "📊 MÉTRICAS POR CLASE:\n"
    formatted_text += "┌─────────────┬───────────┬─────────┬──────────┬─────────┐\n"
    formatted_text += "│    Clase    │ Precision │ Recall  │ F1-Score │ Support │\n"
    formatted_text += "├─────────────┼───────────┼─────────┼──────────┼─────────┤\n"

    if alta_line and len(alta_line) >= 5:
        formatted_text += f"│    ALTA     │   {float(alta_line[1]):.2f}    │  {float(alta_line[2]):.2f}   │   {float(alta_line[3]):.2f}    │   {alta_line[4]:>3}   │\n"

    if baja_line and len(baja_line) >= 5:
        formatted_text += f"│    BAJA     │   {float(baja_line[1]):.2f}    │  {float(baja_line[2]):.2f}   │   {float(baja_line[3]):.2f}    │   {baja_line[4]:>3}   │\n"

    formatted_text += "└─────────────┴───────────┴─────────┴──────────┴─────────┘\n\n"

    # Matriz de confusión
    formatted_text += "🔍 MATRIZ DE CONFUSIÓN:\n"
    formatted_text += "                    Predicción\n"
    formatted_text += "                 ALTA    BAJA\n"
    formatted_text += f"Real    ALTA  │  {cm[0, 0]:3d}  │  {cm[0, 1]:3d}  │\n"
    formatted_text += f"        BAJA  │  {cm[1, 0]:3d}  │  {cm[1, 1]:3d}  │\n\n"

    # Interpretación
    formatted_text += "💡 INTERPRETACIÓN:\n"
    formatted_text += "───────────────────────────────────────────────────────────────\n"

    if acc >= 0.8:
        formatted_text += "✅ Excelente rendimiento (>80%)\n"
    elif acc >= 0.7:
        formatted_text += "✅ Buen rendimiento (70-80%)\n"
    elif acc >= 0.6:
        formatted_text += "⚠️  Rendimiento moderado (60-70%)\n"
    else:
        formatted_text += "❌ Rendimiento bajo (<60%)\n"

    # Calcular métricas adicionales
    total_samples = cm.sum()
    true_positives_alta = cm[0, 0]

    formatted_text += f"\n📈 ESTADÍSTICAS ADICIONALES:\n"
    formatted_text += f"• Total de muestras evaluadas: {total_samples}\n"
    formatted_text += f"• Predicciones correctas de ALTA: {true_positives_alta}\n"
    formatted_text += f"• Predicciones correctas de BAJA: {cm[1, 1]}\n"
    formatted_text += f"• Errores de clasificación: {cm[0, 1] + cm[1, 0]}\n\n"

    formatted_text += "📝 ¿QUÉ SIGNIFICAN ESTAS MÉTRICAS?\n"
    formatted_text += "───────────────────────────────────────────────────────────────\n"
    formatted_text += "🎯 PRECISION (Exactitud):\n"
    formatted_text += "   ¿Qué tan confiable es el modelo cuando dice 'ALTA'?\n"
    formatted_text += "   💡 Valor alto = Pocas falsas alarmas\n\n"

    formatted_text += "🔍 RECALL (Sensibilidad):\n"
    formatted_text += "   ¿Qué tan bueno es el modelo para encontrar todos los casos 'ALTA'?\n"
    formatted_text += "   💡 Valor alto = No se pierde casos importantes\n\n"

    formatted_text += "⚖️ F1-SCORE (Balance):\n"
    formatted_text += "   Combina Precision y Recall en un solo número\n"
    formatted_text += "   💡 Valor alto = Buen equilibrio entre exactitud y detección\n\n"

    # Agregar ejemplos prácticos
    if alta_line and baja_line and len(alta_line) >= 5 and len(baja_line) >= 5:
        precision_alta = float(alta_line[1])
        recall_alta = float(alta_line[2])
        support_alta = int(alta_line[4])
        support_baja = int(baja_line[4])

        formatted_text += "🎓 EJEMPLO PRÁCTICO CON TUS DATOS:\n"
        formatted_text += "───────────────────────────────────────────────────────────────\n"
        formatted_text += f"📈 Para posts de ALTA popularidad:\n"
        formatted_text += f"   • Tienes {support_alta} ejemplos reales en tu dataset\n"
        formatted_text += f"   • El modelo acierta {precision_alta * 100:.0f}% cuando predice 'ALTA'\n"
        formatted_text += f"   • Detecta {recall_alta * 100:.0f}% de todos los posts populares\n\n"

        if len(baja_line) >= 3:
            precision_baja = float(baja_line[1])
            recall_baja = float(baja_line[2])
            formatted_text += f"📉 Para posts de BAJA popularidad:\n"
            formatted_text += f"   • Tienes {support_baja} ejemplos reales en tu dataset\n"
            formatted_text += f"   • El modelo acierta {precision_baja * 100:.0f}% cuando predice 'BAJA'\n"
            formatted_text += f"   • Detecta {recall_baja * 100:.0f}% de todos los posts poco populares\n\n"

    # Recomendaciones
    formatted_text += "🚀 RECOMENDACIONES:\n"
    formatted_text += "───────────────────────────────────────────────────────────────\n"
    if acc < 0.7:
        formatted_text += "• Considera aumentar el tamaño del dataset\n"
        formatted_text += "• Revisa la calidad de las etiquetas\n"
        formatted_text += "• Prueba con diferentes parámetros del modelo\n"
    else:
        formatted_text += "• El modelo muestra buen rendimiento\n"
        formatted_text += "• Puedes usar las predicciones con confianza\n"
        formatted_text += "• Considera validación cruzada para mayor robustez\n"

    return formatted_text


# ---------------------------
# Análisis de Features
# ---------------------------

def analyze_model_features(model: Any, top_n: int = 20) -> Dict[str, Any]:
    """
    Analiza las features más importantes del modelo entrenado.
    
    Args:
        model: Pipeline entrenado con TF-IDF + LogisticRegression
        top_n: Número de features top a mostrar
        
    Returns:
        Dict con análisis de features importantes
    """
    try:
        # Obtener el vectorizador y el clasificador
        tfidf = model.named_steps['tfidf']
        lr = model.named_steps['lr']

        # Obtener nombres de features
        feature_names = tfidf.get_feature_names_out()

        # Obtener coeficientes para cada clase
        coef_alta = lr.coef_[0] if lr.classes_[0] == 'alta' else -lr.coef_[0]

        # Features más importantes para clase "alta"
        top_indices = np.argsort(coef_alta)[-top_n:][::-1]
        top_features_alta = [(feature_names[i], coef_alta[i]) for i in top_indices]

        # Features más importantes para clase "baja" (coeficientes más negativos)
        bottom_indices = np.argsort(coef_alta)[:top_n]
        top_features_baja = [(feature_names[i], abs(coef_alta[i])) for i in bottom_indices]

        return {
            "features_alta": top_features_alta,
            "features_baja": top_features_baja,
            "total_features": len(feature_names)
        }
    except Exception as e:
        return {
            "error": f"No se pudo analizar las features: {str(e)}",
            "features_alta": [],
            "features_baja": [],
            "total_features": 0
        }

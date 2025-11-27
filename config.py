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
config.py - Social Trend Analyzer
Configuración y Estilos
--------------------------------------------
Descripción de config.py:
Constantes globales de colores, estilos TTK, parámetros de modelo/analítica,
y utilidades para cargar/guardar configuración de usuario.
============================================
Librerías utilizadas:
- os/json: Gestión de archivos de configuración.
- typing: Tipado estático.
"""
import logging
import os
import json
from typing import Dict, Any

# ---------------------------
# Esquema de Colores Moderno
# ---------------------------

# Colores principales - Inspirado en diseño contemporáneo
PRIMARY = "#6366f1"  # Indigo vibrante
SECONDARY = "#8b5cf6"  # Púrpura elegante
ACCENT = "#06b6d4"  # Cyan brillante
SUCCESS = "#10b981"  # Verde esmeralda
WARNING = "#f59e0b"  # Ámbar cálido
ERROR = "#ef4444"  # Rojo coral

# Fondos con gradientes sutiles
BG_MAIN = "#0f172a"  # Slate 900 - Fondo principal
BG_PANEL = "#1e293b"  # Slate 800 - Paneles
BG_CARD = "#334155"  # Slate 700 - Cards destacadas
BG_HOVER = "#475569"  # Slate 600 - Hover states

# Textos con mejor contraste
FG_TEXT = "#f1f5f9"  # Slate 100 - Texto principal
FG_MUTED = "#94a3b8"  # Slate 400 - Texto secundario
FG_ACCENT = "#e2e8f0"  # Slate 200 - Texto destacado

# Bordes y separadores
BORDER = "#475569"  # Slate 600
BORDER_LIGHT = "#64748b"  # Slate 500

# ---------------------------
# Configuraciones de la Aplicación
# ---------------------------

# Configuraciones de ventana
WINDOW_CONFIG = {
    "title": "Análisis de Redes Sociales y Tendencias – NLP",
    "geometry": "1400x900",
    "min_width": 1350,
    "min_height": 800,
    "bg": BG_MAIN
}

# Configuraciones del modelo ML
MODEL_CONFIG = {
    "test_size": 0.25,
    "random_state": 42,
    "max_features": 20000,
    "ngram_range": (1, 2),
    "max_iter": 200,
    "class_weight": "balanced"
}

# Configuraciones de análisis
ANALYSIS_CONFIG = {
    "top_n_default": 10,
    "top_k_hashtags": 5,
    "percentile_threshold": 70
}

# Configuraciones de gráficos
PLOT_CONFIG = {
    "figure_size": (7, 6),
    "dpi": 100,
    "tight_layout": True
}


# ---------------------------
# Paths y Archivos
# ---------------------------

def get_settings_path() -> str:
    """Retorna la ruta del archivo de configuraciones del usuario."""
    home = os.path.expanduser("~")
    return os.path.join(home, ".social_trend_analyzer.json")


def get_assets_dir() -> str:
    """Retorna la ruta del directorio de assets y lo crea si no existe."""
    home = os.path.expanduser("~")
    path = os.path.join(home, ".social_trend_analyzer_assets")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------
# Gestión de Configuraciones
# ---------------------------

def load_user_settings() -> Dict[str, Any]:
    """Carga las configuraciones del usuario desde el archivo JSON."""
    path = get_settings_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {"show_training_tutorial": data.get("show_training_tutorial", True)}
        except Exception as e:
            logging.exception("Error loading user settings")
            pass
    return {"show_training_tutorial": True}


def save_user_settings(settings: Dict[str, Any]) -> None:
    """Guarda las configuraciones del usuario en el archivo JSON."""
    path = get_settings_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error al guardar configuraciones: {e}")


# ---------------------------
# Configuraciones de Estilo TTK
# ---------------------------

def get_style_configurations() -> Dict[str, Dict]:
    """Retorna las configuraciones de estilo para los widgets TTK."""
    return {
        "TFrame": {
            "configure": {"background": BG_MAIN, "borderwidth": 0},
            "map": {}
        },
        "Header.TFrame": {
            "configure": {"background": BG_MAIN, "relief": "flat", "borderwidth": 0},
            "map": {}
        },
        "Card.TFrame": {
            "configure": {
                "background": BG_PANEL, "relief": "flat", "borderwidth": 1,
                "bordercolor": BORDER
            },
            "map": {}
        },
        "Header.TLabel": {
            "configure": {
                "background": BG_MAIN, "foreground": FG_TEXT,
                "font": ("Segoe UI", 18, "bold")
            },
            "map": {}
        },
        "Subheader.TLabel": {
            "configure": {
                "background": BG_MAIN, "foreground": FG_MUTED,
                "font": ("Segoe UI", 10)
            },
            "map": {}
        },
        "TLabel": {
            "configure": {
                "background": BG_PANEL, "foreground": FG_TEXT,
                "font": ("Segoe UI", 10)
            },
            "map": {}
        },
        "Accent.TButton": {
            "configure": {
                "background": PRIMARY, "foreground": "#ffffff",
                "borderwidth": 0, "focuscolor": "none",
                "font": ("Segoe UI", 10, "bold")
            },
            "map": {
                "background": [("active", SECONDARY), ("pressed", "#4f46e5")],
                "relief": [("pressed", "sunken"), ("!pressed", "flat")]
            }
        },
        "Success.TButton": {
            "configure": {
                "background": SUCCESS, "foreground": "#ffffff",
                "borderwidth": 0, "focuscolor": "none",
                "font": ("Segoe UI", 10, "bold")
            },
            "map": {
                "background": [("active", "#059669"), ("pressed", "#047857")],
                "relief": [("pressed", "sunken"), ("!pressed", "flat")]
            }
        },
        "Flat.TButton": {
            "configure": {
                "background": BG_CARD, "foreground": FG_TEXT,
                "borderwidth": 1, "bordercolor": BORDER,
                "focuscolor": "none", "font": ("Segoe UI", 10)
            },
            "map": {
                "background": [("active", BG_HOVER), ("pressed", BORDER)],
                "bordercolor": [("focus", PRIMARY), ("active", BORDER_LIGHT)],
                "relief": [("pressed", "sunken"), ("!pressed", "flat")]
            }
        },
        "Status.TLabel": {
            "configure": {
                "background": BG_PANEL, "foreground": FG_MUTED,
                "font": ("Segoe UI", 9)
            },
            "map": {}
        },
        "Treeview": {
            "configure": {
                "background": BG_CARD, "foreground": FG_TEXT,
                "fieldbackground": BG_CARD, "borderwidth": 1,
                "bordercolor": BORDER, "font": ("Segoe UI", 9)
            },
            "map": {
                "background": [("selected", PRIMARY)],
                "foreground": [("selected", "#ffffff")]
            }
        },
        "Treeview.Heading": {
            "configure": {
                "background": BG_HOVER, "foreground": FG_TEXT,
                "borderwidth": 1, "bordercolor": BORDER,
                "font": ("Segoe UI", 9, "bold")
            },
            "map": {
                "background": [("active", BORDER_LIGHT)],
                "relief": [("pressed", "sunken"), ("!pressed", "flat")]
            }
        },
        "TNotebook": {
            "configure": {
                "background": BG_MAIN, "borderwidth": 0,
                "tabmargins": [0, 0, 0, 0]
            },
            "map": {}
        },
        "TNotebook.Tab": {
            "configure": {
                "background": BG_CARD, "foreground": FG_MUTED,
                "padding": [20, 12], "borderwidth": 1,
                "bordercolor": BORDER, "font": ("Segoe UI", 10)
            },
            "map": {
                "background": [("selected", BG_PANEL), ("active", BG_HOVER)],
                "foreground": [("selected", FG_TEXT), ("active", FG_TEXT)],
                "bordercolor": [("selected", PRIMARY), ("active", BORDER_LIGHT)]
            }
        },
        "TEntry": {
            "configure": {
                "fieldbackground": BG_CARD, "foreground": FG_TEXT,
                "bordercolor": BORDER, "borderwidth": 1,
                "font": ("Segoe UI", 10)
            },
            "map": {
                "bordercolor": [("focus", PRIMARY), ("active", SECONDARY)],
                "borderwidth": [("focus", 2), ("!focus", 1)],
                "fieldbackground": [("focus", "#ffffff")]
            }
        },
        # Entrada específica con texto negro para mayor contraste en campos de predicción
        "Input.TEntry": {
            "configure": {
                "fieldbackground": "#ffffff", "foreground": "#000000",
                "bordercolor": BORDER, "borderwidth": 1,
                "font": ("Segoe UI", 10)
            },
            "map": {
                "bordercolor": [("focus", PRIMARY), ("active", SECONDARY)],
                "borderwidth": [("focus", 2), ("!focus", 1)],
                "fieldbackground": [("focus", "#ffffff")],
                "foreground": [("disabled", FG_MUTED), ("!disabled", "#000000")]
            }
        },
        "TCombobox": {
            "configure": {
                "fieldbackground": BG_CARD, "foreground": FG_TEXT,
                "bordercolor": BORDER, "borderwidth": 1,
                "font": ("Segoe UI", 10)
            },
            "map": {
                "bordercolor": [("focus", PRIMARY), ("active", SECONDARY)],
                "borderwidth": [("focus", 2), ("!focus", 1)],
                "fieldbackground": [("focus", "#ffffff")]
            }
        },
        "Horizontal.TScrollbar": {
            "configure": {
                "background": BG_PANEL, "troughcolor": BG_MAIN,
                "bordercolor": BORDER, "arrowcolor": FG_MUTED,
                "darkcolor": BG_CARD, "lightcolor": BG_HOVER,
                "borderwidth": 1, "relief": "flat"
            },
            "map": {
                "background": [("active", BG_HOVER), ("pressed", PRIMARY)],
                "arrowcolor": [("active", FG_TEXT), ("pressed", "#ffffff")],
                "relief": [("pressed", "sunken"), ("!pressed", "flat")]
            }
        },
        "Vertical.TScrollbar": {
            "configure": {
                "background": BG_PANEL, "troughcolor": BG_MAIN,
                "bordercolor": BORDER, "arrowcolor": FG_MUTED,
                "darkcolor": BG_CARD, "lightcolor": BG_HOVER,
                "borderwidth": 1, "relief": "flat"
            },
            "map": {
                "background": [("active", BG_HOVER), ("pressed", PRIMARY)],
                "arrowcolor": [("active", FG_TEXT), ("pressed", "#ffffff")],
                "relief": [("pressed", "sunken"), ("!pressed", "flat")]
            }
        }
    }


# ---------------------------
# Constantes de Validación
# ---------------------------

REQUIRED_COLUMNS = {"user", "text", "hashtags", "likes", "timestamp"}
VALID_LABEL_VALUES = {"alta", "baja"}
MIN_ROWS_REQUIRED = 10
MIN_CLASSES_REQUIRED = 2

# ---------------------------
# Mensajes de la Aplicación
# ---------------------------

MESSAGES = {
    "ready": "Listo. Cargue un CSV para comenzar.",
    "dataset_loaded": "Dataset cargado: {filename} | {rows} filas",
    "model_trained": "Modelo entrenado correctamente.",
    "no_model": "(entrene el modelo para ver métricas)",
    "prediction_ready": "Predicción: {prediction} (confianza: {confidence:.1%})",
    "error_no_dataset": "Primero debe cargar un dataset válido.",
    "error_no_model": "Primero debe entrenar el modelo.",
    "error_invalid_likes": "Ingrese un número válido de likes.",
    "error_empty_text": "Ingrese texto para predecir."
}

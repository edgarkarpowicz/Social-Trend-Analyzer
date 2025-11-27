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
main.py - Social Trend Analyzer
Aplicación Principal - Punto de entrada de la aplicación
--------------------------------------------
Descripción de main.py:
Contiene la clase principal SocialTrendAnalyzer que gestiona la ventana principal de la aplicación,
maneja la carga de datos, el análisis, el entrenamiento del modelo y la predicción.
También incluye la función main() que inicia la aplicación.
============================================
Librerías utilizadas:
- tkinter: Para la interfaz gráfica de usuario.
- pandas: Para el procesamiento y análisis de datos.
- matplotlib: Para la generación de gráficos.
- config: Módulo propio para configuraciones de la aplicación.
- utils: Módulo propio con funciones utilitarias.
- analyzer: Módulo propio con funciones de análisis y modelado.
- ui_components: Módulo propio con componentes de UI reutilizables.
============================================
"""

import os
import traceback
import tkinter as tk
from tkinter import ttk
from typing import Optional
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# Importar módulos propios
from config import (
    WINDOW_CONFIG, BG_MAIN, get_style_configurations,
    load_user_settings
)
from utils import (
    show_error, validate_dataset, build_label_if_missing
)
from analyzer import (
    compute_basic_stats, compute_hashtag_trend, train_text_model,
    predict_with_model
)
from ui_components import (
    TutorialDialog, UIBuilder, UIUtils, UIValidators
)

matplotlib.use("TkAgg")


# ---------------------------
# Clase Principal de la Aplicación
# ---------------------------


class SocialTrendAnalyzer(tk.Tk):
    """Aplicación principal para análisis de tendencias en redes sociales."""

    def __init__(self):
        super().__init__()
        self.focus_hashtag = None
        self.df = None
        self.pred_entry = None
        self.likes_entry = None
        self.metrics_text = None
        self.figure = None
        self.canvas = None
        self.ax = None
        self.tree_summary = None
        self.tree_hashtags = None
        self.tree_users = None
        self.focus_combo = None
        self.status = None
        self._setup_window()
        self._initialize_data()
        self._setup_ui()

    def _setup_window(self):
        """Configura la ventana principal."""
        self.title(WINDOW_CONFIG["title"])
        self.geometry(WINDOW_CONFIG["geometry"])
        self.minsize(WINDOW_CONFIG["min_width"], WINDOW_CONFIG["min_height"])
        self.configure(bg=BG_MAIN)

    def _initialize_data(self):
        """Inicializa las variables de datos."""
        self.df: Optional[pd.DataFrame] = None
        self.model = None
        self.settings = load_user_settings()
        self.focus_hashtag: Optional[str] = None
        self.assets = self._ensure_assets()

    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        self._init_style()
        self.ui_builder = UIBuilder(self)
        self.ui_builder.build_main_ui()

    @staticmethod
    def _init_style():
        """Inicializa los estilos TTK."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Obtener configuraciones de estilo
        style_configs = get_style_configurations()

        # Aplicar configuraciones de estilo
        for style_name, config in style_configs.items():
            # Aplicar configuración normal si existe
            if "configure" in config:
                style.configure(style_name, **config["configure"])

            # Aplicar mapeo si existe
            if "map" in config and config["map"]:
                style.map(style_name, **config["map"])

    def _ensure_assets(self) -> dict:
        """
        Genera assets de tutorial (imágenes PNG) si no existen.
        Utiliza Matplotlib para crear imágenes mock.
        """
        assets = {}
        assets_dir = self._get_assets_dir()

        # CSV preview
        csv_png = os.path.join(assets_dir, "csv_preview.png")
        if not os.path.exists(csv_png):
            self._create_csv_preview(csv_png)
        assets["csv_preview"] = csv_png

        # Métricas preview
        metrics_png = os.path.join(assets_dir, "metrics_preview.png")
        if not os.path.exists(metrics_png):
            self._create_metrics_preview(metrics_png)
        assets["metrics_preview"] = metrics_png

        return assets

    @staticmethod
    def _get_assets_dir() -> str:
        """Obtiene el directorio de assets, creándolo si no existe."""
        home = os.path.expanduser("~")
        path = os.path.join(home, ".social_trend_analyzer_assets")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _create_csv_preview(filepath: str):
        """Crea una imagen de preview del CSV."""
        fig = plt.figure(figsize=(6, 1.8), dpi=150)
        ax = fig.add_subplot(111)
        ax.axis('off')

        cols = ["user", "text", "hashtags", "likes", "timestamp"]
        data = [
            ["@maria", "Analizando datos con pandas", "#data #python", 120, "2025-10-03 10:15:00"],
            ["@juan", "TF-IDF + LR funciona bien", "#nlp #ml", 75, "2025-10-05 14:20:00"],
            ["@ana", "Gráficos en Matplotlib", "#tech", 34, "2025-10-06 09:05:00"],
        ]

        table = ax.table(cellText=data, colLabels=cols, cellLoc='center', loc='center')
        table.scale(1, 1.3)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#30363d')
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor('#161b22')
                cell.set_text_props(color='#c9d1d9', fontweight='bold')
            else:
                cell.set_facecolor('#0d1117')
                cell.set_text_props(color='#c9d1d9')

        fig.tight_layout()
        fig.savefig(filepath, transparent=True)
        plt.close(fig)

    @staticmethod
    def _create_metrics_preview(filepath: str):
        """Crea una imagen de preview de métricas."""
        fig = plt.figure(figsize=(6, 2.2), dpi=150)
        ax = fig.add_subplot(111)
        ax.axis('off')

        sample = (
            "Accuracy: 0.84\n"
            "Classification report:\n"
            "            precision    recall  f1-score   support\n"
            "       alta       0.86      0.82      0.84        56\n"
            "       baja       0.83      0.86      0.84        54\n"
            "    accuracy                           0.84       110"
        )

        ax.text(0, 1, sample, va='top', ha='left', family='monospace', color='#c9d1d9')
        fig.tight_layout()
        fig.savefig(filepath, transparent=True)
        plt.close(fig)

    # ---------------------------
    # Métodos de Manejo de Eventos
    # ---------------------------

    def on_load_csv(self):
        """Maneja la carga de archivos CSV."""
        filepath = UIUtils.select_file("Seleccionar archivo CSV", [("CSV files", "*.csv"), ("All files", "*.*")])
        if not filepath:
            return

        try:
            # Cargar CSV
            df = pd.read_csv(filepath)

            # Validar estructura
            is_valid, errors, df = validate_dataset(df)
            if not is_valid:
                error_msg = "Errores en el dataset:\n" + "\n".join(errors)
                show_error("Dataset inválido", error_msg)
                return

            # Procesar datos
            self.df = self._process_dataset(df)

            # Actualizar UI
            self._update_ui_after_load()

            # Actualizar status
            self.status.set(f"Dataset cargado: {len(self.df)} filas, {len(self.df.columns)} columnas")

        except Exception as e:
            show_error("Error al cargar CSV", f"No se pudo cargar el archivo:\n{str(e)}")

    @staticmethod
    def _process_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Procesa el dataset después de cargarlo."""
        df = build_label_if_missing(df)

        return df

    def _update_ui_after_load(self):
        """Actualiza la UI después de cargar un dataset."""
        if self.df is None:
            return

        # Limpiar modelo anterior
        self.model = None

        # Actualizar combo de hashtags
        all_hashtags = self.df["hashtags_list"].explode().value_counts().head(20).index.tolist()
        UIUtils.update_combobox_values(self.focus_combo, all_hashtags)

        # Limpiar foco
        self.focus_hashtag = None

    def on_show_stats(self):
        """Muestra estadísticas básicas del dataset."""
        if not UIValidators.require_dataframe(self.df, "mostrar estadísticas"):
            return

        try:
            # Calcular estadísticas
            stats = compute_basic_stats(self.df)

            # Actualizar trees
            UIUtils.populate_tree(self.tree_users, stats["users_active"])
            UIUtils.populate_tree(self.tree_hashtags, stats["hashtags_top"])
            UIUtils.populate_tree(self.tree_summary, stats["summary"])

            # Actualizar status
            self.status.set("Estadísticas calculadas y mostradas en las pestañas")

        except Exception as e:
            show_error("Error en estadísticas", f"No se pudieron calcular las estadísticas:\n{str(e)}")

    def on_plot_trend(self):
        """Genera gráfico de tendencias de hashtags."""
        if not UIValidators.require_dataframe(self.df, "generar gráfico de tendencias"):
            return

        try:
            # Calcular tendencias
            trend_data = compute_hashtag_trend(self.df)

            if trend_data.empty:
                UIUtils.show_warning_dialog("Sin datos", "No hay suficientes datos para generar tendencias.")
                return

            # Limpiar gráfico anterior
            self.ax.clear()

            # Plotear tendencias
            self._plot_hashtag_trends(trend_data)

            # Actualizar canvas
            self.canvas.draw()

            # Actualizar status
            self.status.set("Gráfico de tendencias actualizado")

        except Exception as e:
            show_error("Error en gráfico", f"No se pudo generar el gráfico:\n{str(e)}")

    def _plot_hashtag_trends(self, trend_data: pd.DataFrame):
        """Plotea las tendencias de hashtags."""
        # Configurar gráfico
        self.ax.set_title("Tendencia diaria de hashtags (Top)")
        self.ax.set_xlabel("Fecha")
        self.ax.set_ylabel("Conteo")

        # Plotear cada hashtag
        for hashtag in trend_data.columns:
            alpha = 1.0 if hashtag == self.focus_hashtag else 0.6
            linewidth = 3 if hashtag == self.focus_hashtag else 1.5

            self.ax.plot(trend_data.index, trend_data[hashtag],
                         label=f"#{hashtag}", alpha=alpha, linewidth=linewidth)

        # Configurar leyenda y formato
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.grid(True, alpha=0.3)

        # Rotar etiquetas de fecha si es necesario
        if len(trend_data.index) > 10:
            self.ax.tick_params(axis='x', rotation=45)

        self.figure.tight_layout()

    def on_train_flow(self):
        """Inicia el flujo de entrenamiento del modelo."""
        if not UIValidators.require_dataframe(self.df, "entrenar el modelo"):
            return

        if self.settings.get("show_training_tutorial", True):
            TutorialDialog(self, on_start=self._do_train, settings_ref=self.settings, assets=self.assets)
        else:
            self._do_train()

    def _do_train(self):
        """Ejecuta el entrenamiento del modelo."""
        if not UIValidators.require_dataframe(self.df, "entrenar el modelo"):
            return

        try:
            # Validar dataset para entrenamiento
            is_valid, errors, validated_df = validate_dataset(self.df)
            if not is_valid:
                error_msg = "No se puede entrenar:\n" + "\n".join(errors)
                show_error("Entrenamiento", error_msg)
                return

            # Actualizar self.df con el dataset validado
            self.df = validated_df

            # Entrenar modelo
            self.status.set("Entrenando modelo... Por favor espere.")
            self.update()  # Forzar actualización de UI

            model, metrics_text, accuracy, confusion_matrix = train_text_model(self.df)

            # Guardar modelo
            self.model = model

            # Mostrar métricas
            UIUtils.update_metrics_text(self.metrics_text, metrics_text)

            # Actualizar status
            self.status.set(f"Modelo entrenado exitosamente. Accuracy: {accuracy:.3f}")

            UIUtils.show_info_dialog("Entrenamiento completo",
                                     f"El modelo se entrenó exitosamente.\n"
                                     f"Accuracy: {accuracy:.3f}\n"
                                     f"Revise la pestaña 'Métricas Modelo' para más detalles.")

        except Exception as e:
            show_error("Error en entrenamiento", f"No se pudo entrenar el modelo:\n{str(e)}")
            self.status.set("Error en entrenamiento")

    def on_predict(self):
        """Realiza una predicción con el modelo entrenado."""
        if not UIValidators.require_trained_model(self.model):
            return

        # Validar entradas
        is_valid, text, hashtags, likes = UIValidators.validate_prediction_inputs(
            self.pred_entry, self.likes_entry
        )

        if not is_valid:
            return

        try:
            # Realizar predicción
            prediction, confidence = predict_with_model(self.model, text, hashtags, likes)

            # Mostrar resultado
            result_msg = (
                f"Texto: {text}\n"
                f"Hashtags: {hashtags}\n"
                f"Likes estimados: {likes}\n\n"
                f"Predicción: {prediction.upper()}\n"
                f"Confianza: {confidence:.2%}"
            )

            UIUtils.show_info_dialog("Resultado de Predicción", result_msg)

            # Actualizar status
            self.status.set(f"Predicción: {prediction} (confianza: {confidence:.2%})")

        except Exception as e:
            show_error("Error en predicción", f"No se pudo realizar la predicción:\n{str(e)}")

    def on_focus_combo_change(self, event=None):
        """Maneja el cambio en el combo de foco de hashtag."""
        selected = self.focus_combo.get()
        if selected:
            self.focus_hashtag = selected
            self.on_plot_trend()  # Regenerar gráfico con foco

    def clear_focus(self):
        """Limpia el foco de hashtag."""
        self.focus_hashtag = None
        self.focus_combo.set("")
        if self.df is not None:
            self.on_plot_trend()  # Regenerar gráfico sin foco

    def on_focus_hashtag(self, event=None):
        """Maneja la selección de hashtag en la tabla para enfocar en gráfico."""
        selection = self.tree_hashtags.selection()
        if selection:
            item = self.tree_hashtags.item(selection[0])
            hashtag = item['values'][0]  # Primera columna es el hashtag
            self.focus_hashtag = hashtag
            self.focus_combo.set(hashtag)
            if self.df is not None:
                self.on_plot_trend()  # Regenerar gráfico con foco


# ---------------------------
# Función Principal
# ---------------------------

def main():
    """Función principal de la aplicación."""
    try:
        app = SocialTrendAnalyzer()
        app.mainloop()
    except Exception as e:
        print(f"Error fatal en la aplicación: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

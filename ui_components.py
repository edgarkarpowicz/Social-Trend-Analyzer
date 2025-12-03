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
ui_components.py - Social Trend Analyzer
Componentes y Utilidades de Interfaz
--------------------------------------------
Descripción de ui_components.py:
Diálogos, construcción de la UI principal, widgets reutilizables, tooltips
y validadores de entradas para la interfaz basada en Tkinter/TTK.
============================================
Librerías utilizadas:
- tkinter/ttk: Construcción de interfaz.
- matplotlib: Embedding de figuras en Tk.
- pandas: Interacción con DataFrames en la UI.
============================================
FLUJO BÁSICO DEL MÓDULO (ui_components.py):
--------------------------------------------
Este archivo es el CONSTRUCTOR DE INTERFAZ del proyecto, conteniendo todos
los componentes visuales reutilizables y la lógica de construcción de la UI.

¿Qué hace este módulo?
1. DIÁLOGO TUTORIAL (TutorialDialog):
   - Ventana modal educativa que se muestra antes del entrenamiento
   - Navegación paso a paso (5 pasos) con botones Anterior/Siguiente
   - Muestra:
     * Requisitos del dataset
     * Qué aprende el modelo
     * Cómo se valida
     * Buenas prácticas
     * Interpretación de resultados
   - Incluye imágenes generadas (preview de CSV y métricas)
   - Checkbox "No volver a mostrar" que persiste preferencia
   - Se centra automáticamente sobre ventana padre

2. CONSTRUCTOR DE UI PRINCIPAL (UIBuilder):
   Clase responsable de construir TODA la interfaz gráfica.

   a) build_main_ui():
      - Orquesta construcción completa de la ventana
      - Llama a métodos específicos para cada sección

   b) _build_header():
      - Crea barra superior con título y subtítulo
      - Aplica estilos Header.TLabel y Subheader.TLabel

   c) _build_toolbar():
      - Crea barra de herramientas con scroll horizontal
      - Contiene:
        * Botones de acción (Cargar CSV, Estadísticas, Gráfico, Entrenar)
        * Sección de predicción (Entry para texto/hashtags/likes, botón Predecir)
        * Sección de foco en hashtags (Combobox, botón Limpiar)
      - Usa Canvas + Scrollbar para manejar overflow horizontal
      - Añade tooltips explicativos a cada botón

   d) _build_main_panel():
      - Panel central dividido en dos columnas:
        * Izquierda: Notebook con pestañas (Usuarios, Hashtags, Métricas, Modelo)
        * Derecha: Área de gráfico (canvas Matplotlib embebido)

   e) _build_tabs():
      - Crea 4 pestañas:
        * TAB 1 - Usuarios: Treeview mostrando usuarios más activos
        * TAB 2 - Hashtags: Treeview con hashtags populares (clickeable para foco)
        * TAB 3 - Métricas: Treeview con estadísticas de engagement
        * TAB 4 - Modelo: TextBox con métricas detalladas del ML

   f) _build_plot_area():
      - Crea Figure de Matplotlib
      - Embebe en FigureCanvasTkAgg para integrarlo en Tkinter
      - Configura fondo oscuro acorde al tema

   g) _build_status_bar():
      - Barra inferior con mensajes de estado
      - Muestra: "Listo", "Dataset cargado: X filas", "Modelo entrenado", etc.

3. TOOLTIP (Clase):
   - Implementa tooltips (mensajes emergentes) al posar cursor sobre widgets
   - Aparece tras 400ms de hover
   - Se oculta automáticamente al quitar cursor
   - Usa after() de Tkinter para scheduling

4. UTILIDADES DE UI (UIUtils):
   Métodos estáticos para operaciones comunes:
   
   - select_file(): Diálogo para seleccionar archivo CSV
   - populate_tree(): Llena Treeview con datos de DataFrame
   - clear_tree(): Limpia contenido de Treeview
   - update_combobox_values(): Actualiza opciones de Combobox
   - create_scrollable_tree(): Factory para crear Treeview con scrollbars
   - add_tooltip(): Helper para añadir tooltip a cualquier widget
   - format_tree_data(): Formatea valores numéricos para mostrar en tabla
   - resize_treeview_columns(): Ajusta ancho de columnas automáticamente

5. VALIDADORES DE UI (UIValidators):
   Validaciones específicas para interacciones:
   
   - require_dataframe(): Verifica que haya dataset cargado antes de operar
   - require_trained_model(): Verifica que el modelo esté entrenado antes de predecir
   - validate_prediction_inputs(): Valida campos de entrada para predicción
     * Texto no vacío
     * Likes numérico válido
     * Retorna tuple (is_valid, text, hashtags, likes)

CONTRIBUCIÓN AL PROYECTO:
Este módulo es el RESPONSABLE DE LA EXPERIENCIA DE USUARIO. Separa
completamente la lógica de presentación (UI) de la lógica de negocio
(análisis, ML). Permite que main.py se enfoque en coordinación mientras
ui_components maneja todos los detalles visuales.

Implementa un diseño MODULAR y REUTILIZABLE:
- Cualquier componente (Tooltip, Tree, Dialog) puede usarse independientemente
- Sigue patrón Builder para construcción incremental de UI compleja
- Validadores centralizados evitan código duplicado

Cumple con la consigna de "INCLUIR INTERFACE GRÁFICA" de forma profesional
con tema oscuro moderno, tooltips, scroll, navegación por pestañas y
experiencia pulida.
============================================
"""
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any, Callable, Optional

import matplotlib
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import (
    BG_MAIN, BG_PANEL, FG_TEXT, BORDER,
    save_user_settings
)
from utils import validate_likes_input, validate_text_input

matplotlib.use("TkAgg")


# Las variables de color ya están importadas directamente desde config

# ---------------------------
# Diálogo Tutorial
# ---------------------------

class TutorialDialog(tk.Toplevel):
    """Pequeño tutorial paso a paso que se muestra antes de entrenar el modelo."""

    def __init__(self, master, on_start: Callable, settings_ref: Dict, assets: Optional[Dict] = None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Guía rápida: Entrenamiento de IA")
        self.configure(bg=BG_PANEL)
        self.resizable(False, False)
        self.on_start = on_start
        self.settings_ref = settings_ref
        self.assets = assets or {}
        self.step = 0

        self._build_tutorial_ui()
        self._center_on_parent(master)

    def _build_tutorial_ui(self):
        """Construye la interfaz del diálogo tutorial."""
        self.container = ttk.Frame(self, style="Card.TFrame")
        self.container.grid(row=0, column=0, padx=16, pady=16)

        self.title_lbl = ttk.Label(self.container, text="Cómo entrena la app (resumen)", style="Header.TLabel")
        self.title_lbl.grid(row=0, column=0, sticky="w")

        # Body area (text + image)
        body = ttk.Frame(self.container, style="Card.TFrame")
        body.grid(row=1, column=0, pady=(8, 8), sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)

        self.text = tk.Text(body, width=70, height=14, bg=BG_PANEL, fg=FG_TEXT,
                            insertbackground=FG_TEXT, relief="flat", borderwidth=0, wrap="word",
                            font=("Segoe UI", 10))
        self.text.grid(row=0, column=0, sticky="nsew")

        img_frame = ttk.Frame(body, style="Card.TFrame")
        img_frame.grid(row=0, column=1, sticky="nsw", padx=(12, 0))
        self.img_label = ttk.Label(img_frame)
        self.img_label.grid(row=0, column=0, sticky="n")
        self.img_caption = ttk.Label(img_frame, text="", style="Subheader.TLabel")
        self.img_caption.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.text.configure(state="disabled")

        # Checkbox para no mostrar más
        self.no_show_var = tk.BooleanVar(value=False)
        self.no_show_chk = ttk.Checkbutton(self.container, text="No volver a mostrar esta guía",
                                           variable=self.no_show_var)
        self.no_show_chk.grid(row=2, column=0, sticky="w", pady=(0, 8))

        # Botones de navegación
        btns = ttk.Frame(self.container, style="Card.TFrame")
        btns.grid(row=3, column=0, sticky="e")
        self.prev_btn = ttk.Button(btns, text="← Anterior", style="Flat.TButton", command=self.prev_step)
        self.next_btn = ttk.Button(btns, text="Siguiente →", style="Accent.TButton", command=self.next_step)
        self.start_btn = ttk.Button(btns, text="Comenzar entrenamiento", style="Accent.TButton",
                                    command=self._start)
        self.prev_btn.grid(row=0, column=0, padx=4)
        self.next_btn.grid(row=0, column=1, padx=4)
        self.start_btn.grid(row=0, column=2, padx=4)

        self._render_step()

    def _center_on_parent(self, master):
        """Centra el diálogo sobre la ventana principal."""
        self.update_idletasks()
        x = master.winfo_rootx() + (master.winfo_width() // 2 - self.winfo_width() // 2)
        y = master.winfo_rooty() + (master.winfo_height() // 2 - self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _set_text(self, content: str):
        """Establece el contenido del área de texto."""
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, content)
        self.text.configure(state="disabled")

    def _set_image(self, key: Optional[str], caption: str = ""):
        """Establece la imagen y caption del tutorial."""
        if key and key in self.assets:
            self.img = tk.PhotoImage(file=self.assets[key])
            self.img_label.configure(image=self.img)
            self.img_caption.configure(text=caption)
        else:
            self.img_label.configure(image="")
            self.img_caption.configure(text="")

    def _render_step(self):
        """Renderiza el paso actual del tutorial."""
        steps = [
            (
                "1) Requisitos del dataset",
                "• Columnas: user, text, hashtags, likes, timestamp (YYYY-MM-DD HH:MM:SS).\n"
                "• Si no hay columna label (alta/baja), la app la genera por percentil 70 de likes.\n"
                "• Los hashtags pueden venir como '#ai #ml', 'ai, ml' o lista; se normalizan automáticamente.",
                "csv_preview",
                "Ejemplo visual de CSV válido"
            ),
            (
                "2) Qué aprende el modelo",
                "• Modelo: TF-IDF (1-2 grams) + Regresión Logística con class_weight=balanced.\n"
                "• Features: texto + hashtags (como tokens '#hashtag').\n"
                "• Objetivo: predecir 'alta' o 'baja' relevancia.",
                None,
                ""
            ),
            (
                "3) Cómo se valida",
                "• Split automático 75%/25% estratificado.\n"
                "• Se muestran Accuracy, reporte de clasificación y matriz de confusión.\n"
                "• Consejo: cuanto más y mejor texto/hashtags, mejor generaliza.",
                "metrics_preview",
                "Métricas esperadas (mock)"
            ),
            (
                "4) Buenas prácticas rápidas",
                "• Evitá duplicados exactos.\n"
                "• Curá los hashtags (sin ruido).\n"
                "• No te obsesiones con likes absolutos: engagement y contenido importan.",
                None,
                ""
            ),
            (
                "5) Interpretación y siguientes pasos",
                "• 'Alta' ≠ éxito garantizado, es probabilidad según el patrón aprendido.\n"
                "• Probá nuevos textos con 'Predicción'.\n"
                "• Próximas mejoras: guardar modelo, ROC-AUC, k-fold, tuning C.",
                None,
                ""
            )
        ]

        if self.step < len(steps):
            title, body, img_key, caption = steps[self.step]
            self.title_lbl.configure(text=title)
            self._set_text(body)
            self._set_image(img_key, caption)

        self.prev_btn.configure(state=("normal" if self.step > 0 else "disabled"))
        self.next_btn.configure(state=("normal" if self.step < len(steps) - 1 else "disabled"))

    def prev_step(self):
        """Navega al paso anterior."""
        if self.step > 0:
            self.step -= 1
            self._render_step()

    def next_step(self):
        """Navega al siguiente paso."""
        if self.step < 4:  # 5 pasos en total (0-4)
            self.step += 1
            self._render_step()

    def _start(self):
        """Inicia el entrenamiento y cierra el diálogo."""
        # Persistir preferencia
        self.settings_ref["show_training_tutorial"] = not self.no_show_var.get()
        try:
            save_user_settings(self.settings_ref)
        except Exception as e:
            logging.exception("Error saving tutorial preference")
            pass
        self.destroy()
        if callable(self.on_start):
            self.on_start()


# ---------------------------
# Constructor de UI Principal
# ---------------------------

class UIBuilder:
    """Clase responsable de construir los componentes de la interfaz principal."""

    def __init__(self, parent_app):
        self.app = parent_app

    def build_main_ui(self):
        """Construye la interfaz principal de la aplicación."""
        self.app.columnconfigure(0, weight=1)
        self.app.rowconfigure(2, weight=1)

        # Header
        self._build_header()

        # Toolbar con scroll horizontal
        self._build_toolbar()

        # Panel principal con tabs y gráfico
        self._build_main_panel()

        # Status bar
        self._build_status_bar()

    def _build_header(self):
        """Construye el header de la aplicación."""
        header = ttk.Frame(self.app, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.columnconfigure(0, weight=1)

        top_bar = ttk.Frame(header)
        top_bar.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 12))
        top_bar.columnconfigure(0, weight=1)

        ttk.Label(top_bar, text="🌐 Social Trend Analyzer", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top_bar, text="🔬 Análisis Inteligente • 📊 Machine Learning • 🎯 Predicciones",
                  style="Subheader.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _build_toolbar(self):
        """Construye el toolbar con scroll horizontal."""
        # Contenedor principal del toolbar con scrollbar horizontal
        toolbar_container = ttk.Frame(self.app)
        toolbar_container.grid(row=1, column=0, sticky="ew", padx=24, pady=(8, 16))
        toolbar_container.columnconfigure(0, weight=1)
        toolbar_container.rowconfigure(0, weight=1)

        # Canvas para el scrollbar horizontal con mejor integración visual
        toolbar_canvas = tk.Canvas(toolbar_container, height=80, bg=BG_MAIN,
                                   highlightthickness=1, highlightcolor=BORDER,
                                   highlightbackground=BORDER, relief="flat")
        toolbar_canvas.grid(row=0, column=0, sticky="ew", padx=1, pady=(1, 0))

        # Scrollbar horizontal con estilo personalizado
        toolbar_scrollbar = ttk.Scrollbar(toolbar_container, orient="horizontal",
                                          command=toolbar_canvas.xview, style="Horizontal.TScrollbar")
        toolbar_scrollbar.grid(row=1, column=0, sticky="ew", padx=1, pady=(0, 1))
        toolbar_canvas.configure(xscrollcommand=toolbar_scrollbar.set)

        # Frame scrollable dentro del canvas
        toolbar = ttk.Frame(toolbar_canvas)
        toolbar_window = toolbar_canvas.create_window((0, 0), window=toolbar, anchor="nw")

        card = ttk.Frame(toolbar, style="Card.TFrame")
        card.pack(fill="x", padx=0, pady=0)

        # Configurar scroll
        self._setup_toolbar_scroll(toolbar_canvas, toolbar_scrollbar, toolbar, toolbar_window)

        # Botones principales
        self._build_toolbar_buttons(card)

    @staticmethod
    def _setup_toolbar_scroll(canvas, scrollbar, toolbar, toolbar_window):
        """Configura el comportamiento del scroll del toolbar."""

        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Ajustar el ancho del frame interno al canvas si es necesario
            canvas_width = canvas.winfo_width()
            toolbar_width = toolbar.winfo_reqwidth()
            if toolbar_width < canvas_width:
                canvas.itemconfig(toolbar_window, width=canvas_width)

            # Mostrar/ocultar scrollbar según sea necesario
            if toolbar_width > canvas_width:
                scrollbar.grid(row=1, column=0, sticky="ew")
            else:
                scrollbar.grid_remove()

        # Función para scroll con rueda del mouse
        def on_mousewheel(event):
            # Solo hacer scroll horizontal si hay contenido que se desborda
            canvas_width = canvas.winfo_width()
            toolbar_width = toolbar.winfo_reqwidth()
            if toolbar_width > canvas_width:
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        toolbar.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_scroll_region)
        canvas.bind("<MouseWheel>", on_mousewheel)
        toolbar.bind("<MouseWheel>", on_mousewheel)

    def _build_toolbar_buttons(self, card):
        """Construye los botones del toolbar."""
        # Botones principales
        ttk.Button(card, text="📁 Cargar Dataset", style="Accent.TButton",
                   command=self.app.on_load_csv).grid(row=0, column=0, padx=(12, 8), pady=16)
        ttk.Button(card, text="📊 Estadísticas", style="Flat.TButton",
                   command=self.app.on_show_stats).grid(row=0, column=1, padx=6, pady=16)
        ttk.Button(card, text="📈 Tendencias", style="Flat.TButton",
                   command=self.app.on_plot_trend).grid(row=0, column=2, padx=6, pady=16)
        ttk.Button(card, text="🚀 Entrenar IA", style="Success.TButton",
                   command=self.app.on_train_flow).grid(row=0, column=3, padx=6, pady=16)

        # Sección de predicción
        self._build_prediction_section(card)

        # Sección de foco hashtag
        self._build_focus_section(card)

    def _build_prediction_section(self, card):
        """Construye la sección de predicción."""
        pred = ttk.Frame(card, style="Card.TFrame")
        pred.grid(row=0, column=4, padx=12, pady=12, sticky="ns")

        ttk.Label(pred, text="📝 Texto + #hashtags:").grid(
            row=0, column=0, padx=(8, 6), pady=12, sticky="w"
        )
        self.app.pred_entry = ttk.Entry(pred, width=30, style="Input.TEntry")
        self.app.pred_entry.grid(row=0, column=1, padx=4)

        ttk.Label(pred, text="❤️ Likes est.:").grid(
            row=0, column=2, padx=(6, 4), sticky="w"
        )
        self.app.likes_entry = ttk.Entry(pred, width=6, style="Input.TEntry")
        self.app.likes_entry.insert(0, "0")
        self.app.likes_entry.grid(row=0, column=3, padx=4)

        ttk.Button(
            pred, text="🔮 Clasificar", style="Accent.TButton",
            command=self.app.on_predict
        ).grid(row=0, column=4, padx=(6, 4))

        # Icono de información con tooltip (explica qué hace "Clasificar")
        info_lbl = ttk.Label(pred, text="ℹ", cursor="hand2")
        info_lbl.grid(row=0, column=5, padx=(2, 8))

        tooltip_text = (
            "Al pulsar 'Clasificar', la app usa el modelo entrenado para predecir la relevancia\n"
            "de tu publicación a partir del texto, los #hashtags y los likes estimados.\n\n"
            "Qué hace:\n"
            "• Procesa el texto (sin los #) y los #hashtags ingresados.\n"
            "• Usa los likes estimados como una característica numérica adicional.\n"
            "• Genera la clase: 'ALTA' o 'BAJA'.\n"
            "• Calcula la confianza (probabilidad) asociada a la predicción.\n\n"
            "Qué verás:\n"
            "• Un diálogo con: Texto, Hashtags, Likes estimados, Predicción y Confianza (%).\n"
            "• La barra de estado se actualiza con el resultado."
        )
        UIUtils.add_tooltip(info_lbl, tooltip_text)
        # También permitir clic para mostrar la explicación como diálogo
        info_lbl.bind(
            "<Button-1>",
            lambda e: UIUtils.show_info_dialog("¿Qué hace 'Clasificar'?", tooltip_text),
            add=True,
        )

    def _build_focus_section(self, card):
        """Construye la sección de foco hashtag."""
        focus = ttk.Frame(card, style="Card.TFrame")
        focus.grid(row=0, column=5, padx=12, pady=12, sticky="ns")

        ttk.Label(focus, text="🎯 Foco hashtag:").grid(row=0, column=0, padx=(8, 6), sticky="w")
        self.app.focus_combo = ttk.Combobox(focus, state="readonly", width=16, values=[])
        self.app.focus_combo.grid(row=0, column=1, padx=4)
        self.app.focus_combo.bind("<<ComboboxSelected>>", self.app.on_focus_combo_change)

        ttk.Button(focus, text="🔄 Quitar foco", style="Flat.TButton",
                   command=self.app.clear_focus).grid(row=0, column=2, padx=(8, 12))

    def _build_main_panel(self):
        """Construye el panel principal con tabs y gráfico."""
        main = ttk.Panedwindow(self.app, orient=tk.HORIZONTAL)
        main.grid(row=2, column=0, sticky="nsew", padx=24, pady=(12, 20))

        # Panel izquierdo con tabs
        left_card = ttk.Frame(main, style="Card.TFrame")
        left_card.columnconfigure(0, weight=1)
        left_card.rowconfigure(0, weight=1)
        left = ttk.Notebook(left_card)
        left.grid(row=0, column=0, sticky="nsew")

        # Crear tabs
        self._build_tabs(left)

        # Panel derecho con gráfico
        right = ttk.Frame(main, style="Card.TFrame")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_plot_area(right)

        main.add(left_card, weight=2)
        main.add(right, weight=3)

    def _build_tabs(self, notebook):
        """Construye las pestañas del panel izquierdo."""
        self.app.tab_users = ttk.Frame(notebook, style="Card.TFrame")
        self.app.tab_hashtags = ttk.Frame(notebook, style="Card.TFrame")
        self.app.tab_summary = ttk.Frame(notebook, style="Card.TFrame")
        self.app.tab_metrics = ttk.Frame(notebook, style="Card.TFrame")

        notebook.add(self.app.tab_users, text="👤 Usuarios activos")
        notebook.add(self.app.tab_hashtags, text="🔖 Hashtags top")
        notebook.add(self.app.tab_summary, text="📊 Resumen NumPy")
        notebook.add(self.app.tab_metrics, text="🎯 Métricas Modelo")

        # Construir contenido de tabs
        self.app.tree_users = self._build_tree(self.app.tab_users, ["user", "posts"])
        self.app.tree_hashtags = self._build_tree(self.app.tab_hashtags, ["hashtag", "count"])
        self.app.tree_hashtags.bind("<<TreeviewSelect>>", self.app.on_focus_hashtag)
        self.app.tree_summary = self._build_tree(self.app.tab_summary,
                                                 ["likes_mean", "likes_median", "likes_std", "engagement_mean"])

        # Tab de métricas con texto scrollable
        self._build_metrics_tab()

    @staticmethod
    def _build_tree(parent, columns: List[str]):
        """Construye un Treeview con scrollbar."""
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview, style="Vertical.TScrollbar")
        tree.configure(yscrollcommand=vsb.set)

        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, anchor="center")

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        return tree

    def _build_metrics_tab(self):
        """Construye el tab de métricas con área de texto scrollable."""
        metrics_frame = ttk.Frame(self.app.tab_metrics, style="Card.TFrame")
        metrics_frame.pack(fill="both", expand=True, padx=12, pady=12)

        self.app.metrics_text = tk.Text(metrics_frame, height=20, bg=BG_PANEL, fg=FG_TEXT,
                                        insertbackground=FG_TEXT, relief="flat", borderwidth=0,
                                        font=("Consolas", 9), wrap="none")

        # Scrollbars para el texto de métricas con estilos personalizados
        metrics_vsb = ttk.Scrollbar(metrics_frame, orient="vertical",
                                    command=self.app.metrics_text.yview, style="Vertical.TScrollbar")
        metrics_hsb = ttk.Scrollbar(metrics_frame, orient="horizontal",
                                    command=self.app.metrics_text.xview, style="Horizontal.TScrollbar")
        self.app.metrics_text.configure(yscrollcommand=metrics_vsb.set, xscrollcommand=metrics_hsb.set)

        # Grid layout para métricas
        self.app.metrics_text.grid(row=0, column=0, sticky="nsew")
        metrics_vsb.grid(row=0, column=1, sticky="ns")
        metrics_hsb.grid(row=1, column=0, sticky="ew")

        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)

        self.app.metrics_text.configure(state="disabled")

    def _build_plot_area(self, parent):
        """Construye el área de gráficos."""
        self.app.figure = Figure(figsize=(7, 6), dpi=100)
        self.app.ax = self.app.figure.add_subplot(111)
        self.app.ax.set_title("Tendencia diaria de hashtags (Top)")
        self.app.ax.set_xlabel("Fecha")
        self.app.ax.set_ylabel("Conteo")

        self.app.canvas = FigureCanvasTkAgg(self.app.figure, master=parent)
        self.app.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    def _build_status_bar(self):
        """Construye la barra de estado."""
        statusbar = ttk.Frame(self.app, style="Card.TFrame")
        statusbar.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))

        self.app.status = tk.StringVar(value="Listo. Cargue un CSV para comenzar.")
        ttk.Label(statusbar, textvariable=self.app.status, style="Status.TLabel").pack(anchor="w", padx=12, pady=8)


# ---------------------------
# Utilidades de UI
# ---------------------------

class Tooltip:
    """Tooltip simple que aparece al posar el cursor sobre un widget."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 400, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._tipwindow: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None

        # Enlaces para mostrar/ocultar
        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<ButtonPress>", self._on_leave, add=True)
        widget.bind("<FocusIn>", self._on_enter, add=True)
        widget.bind("<FocusOut>", self._on_leave, add=True)

    def _on_enter(self, _=None):
        self._schedule()

    def _on_leave(self, _=None):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self._after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tipwindow:
            return
        try:
            x = self.widget.winfo_rootx() + 16
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except Exception:
            return

        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        # Marco y etiqueta con estilos básicos (usamos tk.Label para controlar bg/fg)
        frame = tk.Frame(tw, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1, bd=0)
        frame.pack(fill="both", expand=True)
        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            bg=BG_PANEL,
            fg=FG_TEXT,
            padx=8,
            pady=6,
            wraplength=self.wraplength,
            anchor="w"
        )
        label.pack(fill="both", expand=True)

        self._tipwindow = tw

    def _hide(self):
        if self._tipwindow is not None:
            try:
                self._tipwindow.destroy()
            finally:
                self._tipwindow = None

class UIUtils:
    """Utilidades para manejo de componentes UI."""

    @staticmethod
    def add_tooltip(widget: tk.Widget, text: str, delay: int = 400, wraplength: int = 360) -> Tooltip:
        """Adjunta un tooltip a un widget y lo devuelve para usos avanzados."""
        return Tooltip(widget, text, delay=delay, wraplength=wraplength)

    @staticmethod
    def populate_tree(tree: ttk.Treeview, df: pd.DataFrame, max_rows: int = 200):
        """Popula un Treeview con datos de un DataFrame."""
        # Limpiar contenido existente
        for item in tree.get_children():
            tree.delete(item)

        if df is None or df.empty:
            return

        # Limitar filas para rendimiento
        df_display = df.head(max_rows)
        for _, row in df_display.iterrows():
            tree.insert("", tk.END, values=[row[c] for c in df_display.columns])

    @staticmethod
    def update_metrics_text(text_widget: tk.Text, content: str):
        """Actualiza el contenido de un widget Text de métricas."""
        text_widget.configure(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)
        text_widget.configure(state="disabled")

    @staticmethod
    def update_combobox_values(combobox: ttk.Combobox, values: List[str]):
        """Actualiza los valores de un Combobox."""
        combobox['values'] = values
        if values:
            combobox.current(0)
        else:
            combobox.set("")

    @staticmethod
    def show_info_dialog(title: str, message: str):
        """Muestra un diálogo de información."""
        messagebox.showinfo(title, message)

    @staticmethod
    def show_warning_dialog(title: str, message: str):
        """Muestra un diálogo de advertencia."""
        messagebox.showwarning(title, message)

    @staticmethod
    def show_error_dialog(title: str, message: str):
        """Muestra un diálogo de error."""
        messagebox.showerror(title, message)

    @staticmethod
    def ask_yes_no(title: str, message: str) -> bool:
        """Muestra un diálogo de confirmación."""
        return messagebox.askyesno(title, message)

    @staticmethod
    def select_file(title: str = "Seleccionar archivo", filetypes: List[tuple] = None) -> str:
        """Abre un diálogo para seleccionar archivo."""
        if filetypes is None:
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        return filedialog.askopenfilename(title=title, filetypes=filetypes)

    @staticmethod
    def select_save_file(title: str = "Guardar archivo", filetypes: List[tuple] = None,
                         defaultextension: str = ".csv") -> str:
        """Abre un diálogo para guardar archivo."""
        if filetypes is None:
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        return filedialog.asksaveasfilename(title=title, filetypes=filetypes, defaultextension=defaultextension)


# ---------------------------
# Validadores de UI
# ---------------------------

class UIValidators:
    """Validadores específicos para componentes de UI."""

    @staticmethod
    def validate_prediction_inputs(text_entry: ttk.Entry, likes_entry: ttk.Entry) -> tuple[bool, str, str, float]:
        """
        Valida las entradas de predicción.
        
        Returns:
            tuple: (es_valido, texto, hashtags, likes)
        """
        text_input = text_entry.get().strip()
        likes_input = likes_entry.get().strip()

        # Validar texto
        if not validate_text_input(text_input):
            UIUtils.show_warning_dialog("Entrada inválida", "Por favor ingrese un texto válido.")
            return False, "", "", 0.0

        # Validar likes
        is_valid_likes, likes = validate_likes_input(likes_input)
        if not is_valid_likes:
            UIUtils.show_warning_dialog("Likes inválidos", "Por favor ingrese un número válido de likes (mayor o igual a 0).")
            return False, "", "", 0.0

        # Separar texto y hashtags
        parts = text_input.split()
        text_parts = []
        hashtag_parts = []

        for part in parts:
            if part.startswith('#'):
                hashtag_parts.append(part)
            else:
                text_parts.append(part)

        text = ' '.join(text_parts)
        hashtags = ' '.join(hashtag_parts)

        return True, text, hashtags, likes

    @staticmethod
    def require_dataframe(df: pd.DataFrame, operation_name: str = "operación") -> bool:
        """
        Verifica que existe un DataFrame válido.
        
        Args:
            df: DataFrame a validar
            operation_name: Nombre de la operación para el mensaje de error
            
        Returns:
            bool: True si el DataFrame es válido
        """
        if df is None or df.empty:
            UIUtils.show_warning_dialog("Dataset requerido",
                                        f"Primero cargue un CSV válido para realizar esta {operation_name}.")
            return False
        return True

    @staticmethod
    def require_trained_model(model: Any, operation_name: str = "predicción") -> bool:
        """
        Verifica que existe un modelo entrenado.
        
        Args:
            model: Modelo a validar
            operation_name: Nombre de la operación para el mensaje de error
            
        Returns:
            bool: True si el modelo es válido
        """
        if model is None:
            UIUtils.show_warning_dialog("Modelo requerido",
                                        f"Primero entrene el modelo de IA para realizar {operation_name}.")
            return False
        return True

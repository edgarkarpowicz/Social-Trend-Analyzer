# 🌐 Social Trend Analyzer

**Análisis de Redes Sociales y Detección de Tendencias con Machine Learning**

## 📋 Descripción

Aplicación de escritorio para el análisis inteligente de tendencias en redes sociales. Procesa datasets de publicaciones utilizando técnicas de análisis de datos y machine learning para predecir la relevancia de contenidos.

**Autores:** Tomas Molina y Edgar Karpowicz  
**Materia:** Programación Declarativa - 2025  
**Universidad:** Blas Pascal

## ✨ Características Principales

- 📊 **Análisis Estadístico**: Usuarios más activos, hashtags populares, métricas de engagement
- 📈 **Visualización de Tendencias**: Gráficos temporales de hashtags con enfoque interactivo
- 🤖 **Machine Learning**: Clasificación automática de relevancia (Alta/Baja) usando TF-IDF + Regresión Logística
- 🎯 **Predicción en Tiempo Real**: Clasifica nuevos posts y estima su potencial de engagement
- 🎨 **Interfaz Moderna**: GUI con Tkinter y tema oscuro profesional
- 📝 **Tutorial Integrado**: Guía paso a paso para entrenamiento del modelo

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **pandas>=2.0.0**: Procesamiento y análisis de datos
- **numpy>=1.24.0**: Cálculos numéricos y métricas
- **matplotlib>=3.7.0**: Visualizaciones y gráficos
- **scikit-learn>=1.3.0**: Machine Learning (TF-IDF, Regresión Logística)
- **Tkinter**: Interfaz gráfica de usuario

## 📦 Instalación

### 1. Clonar o Descargar el Proyecto

```bash
cd "d:\2025\PD\Final PD\PD FINAL"
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

```bash
python main.py
```

## 📁 Estructura del Proyecto

```
PD FINAL/
├── main.py                 # Punto de entrada, clase principal
├── analyzer.py             # Funciones de análisis y ML
├── config.py               # Configuraciones y constantes
├── ui_components.py        # Componentes de interfaz gráfica
├── utils.py                # Utilidades y validaciones
└── README.md               # Este archivo
```

## 📊 Formato del Dataset

El archivo CSV debe contener las siguientes columnas:

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `user` | string | Nombre de usuario | @maria |
| `text` | string | Contenido del post | "Analizando datos con pandas" |
| `hashtags` | string | Hashtags del post | "#data #python" |
| `likes` | int/float | Cantidad de likes | 120 |
| `timestamp` | string | Fecha y hora | "2025-10-03 10:15:00" |

### Formatos Aceptados para Hashtags:
- `"#python #ai"`
- `"['python', 'ai']"`
- `"python, ai"`

### Columna Label (Opcional):
Si el dataset no incluye una columna `label` con valores `alta`/`baja`, la aplicación la generará automáticamente usando el percentil 70 de likes.

## 🚀 Uso de la Aplicación

### 1. Cargar Dataset
- Clic en **"📁 Cargar Dataset"**
- Seleccionar archivo CSV con el formato especificado
- La aplicación valida automáticamente la estructura

### 2. Ver Estadísticas
- Clic en **"📊 Estadísticas"**
- Explora las pestañas:
  - **👤 Usuarios activos**: Top usuarios por cantidad de posts
  - **🔖 Hashtags top**: Hashtags más utilizados
  - **📊 Resumen NumPy**: Métricas calculadas con NumPy

### 3. Visualizar Tendencias
- Clic en **"📈 Tendencias"**
- Observa la evolución temporal de hashtags populares
- Usa **🎯 Foco hashtag** para destacar un hashtag específico

### 4. Entrenar Modelo de IA
- Clic en **"🚀 Entrenar IA"**
- Sigue el tutorial interactivo (opcional)
- El modelo se entrena automáticamente
- Revisa métricas detalladas en la pestaña **🎯 Métricas Modelo**

### 5. Realizar Predicciones
- Ingresa texto y hashtags en **"📝 Texto + #hashtags"**
- Ingresa likes estimados (opcional)
- Clic en **"🔮 Clasificar"**
- Obtén predicción: Alta/Baja con nivel de confianza

## 🎯 Modelo de Machine Learning

### Arquitectura:
- **Vectorización**: TF-IDF con bi-gramas (1-2)
- **Clasificador**: Regresión Logística con pesos balanceados
- **Features**: Texto + Hashtags normalizados
- **Validación**: Split 75/25 estratificado

### Métricas Reportadas:
- ✅ Accuracy General
- 📊 Precision, Recall, F1-Score por clase
- 🔍 Matriz de Confusión
- 💡 Interpretación contextualizada
- 📈 Estadísticas adicionales
- 🚀 Recomendaciones automáticas

## ⚙️ Configuración

Editar `config.py` para personalizar:

```python
# Configuraciones del modelo ML
MODEL_CONFIG = {
    "test_size": 0.25,           # Tamaño del conjunto de prueba
    "random_state": 42,          # Semilla para reproducibilidad
    "max_features": 20000,       # Máximo de features TF-IDF
    "ngram_range": (1, 2),       # Rango de n-gramas
    "max_iter": 200,             # Iteraciones máximas
    "class_weight": "balanced"   # Balanceo de clases
}

# Configuraciones de análisis
ANALYSIS_CONFIG = {
    "top_n_default": 10,         # Top N elementos a mostrar
    "top_k_hashtags": 5,         # Top K hashtags para tendencias
    "percentile_threshold": 70   # Percentil para etiquetado automático
}
```

## 🐛 Solución de Problemas

### Error: "Faltan columnas obligatorias"
- Verifica que el CSV tenga todas las columnas requeridas
- Revisa que los nombres sean exactos (minúsculas)

### Error: "Se requieren al menos X filas"
- El dataset necesita mínimo 10 filas válidas
- Al menos 5 ejemplos por clase (alta/baja)

### Error: "timestamp inválidos"
- Formato esperado: `YYYY-MM-DD HH:MM:SS`
- Ejemplo: `2025-10-03 10:15:00`

### Rendimiento Lento
- Datasets grandes (>10,000 filas) pueden tardar
- El entrenamiento es más rápido en la segunda ejecución

## 📚 Documentación Adicional

- Comentarios inline en cada módulo
- Docstrings en todas las funciones

## 🔧 Desarrollo

### Estructura Modular:
- `main.py`: Aplicación principal y ventana
- `analyzer.py`: Lógica de análisis y ML
- `config.py`: Configuraciones centralizadas
- `ui_components.py`: Componentes de UI reutilizables
- `utils.py`: Funciones auxiliares y validaciones

### Principios Aplicados:
- ✅ Separación de responsabilidades
- ✅ Código DRY (Don't Repeat Yourself)
- ✅ Type hints para claridad
- ✅ Documentación exhaustiva
- ✅ Manejo robusto de errores

## 📄 Licencia

Proyecto educativo para la materia de Programación Declarativa.

---

**¿Preguntas o sugerencias?** Contacta a los autores.


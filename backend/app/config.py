# Importamos os para gestionar rutas relativas y absolutas del proyecto
import os

# Determinamos el directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta a la carpeta que almacena los modelos serializados
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ruta completa al modelo de clasificación de gestos
GESTURE_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.joblib")

# Ruta al archivo con la lista de clases en JSON
CLASSES_PATH = os.path.join(MODELS_DIR, "label_classes.json")

# Ruta al modelo TFLite de MediaPipe para detección de manos
HAND_LANDMARKER_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# Ruta al directorio que contiene el frontend estático
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

# Umbral mínimo de confianza para aceptar una predicción como válida (0.0 a 1.0)
MIN_CONFIDENCE_THRESHOLD = 0.50

# Número de detecciones consecutivas necesarias para registrar una letra en el búfer
STABILITY_THRESHOLD = 5

# Título del microservicio en la documentación de Swagger
API_TITLE = "Traductor de Lengua de Señas a Texto para Atención al Cliente"

# Descripción del microservicio para la documentación de Swagger
API_DESCRIPTION = """
API REST para la traducción en tiempo real de Lengua de Señas a Texto.
Diseñado para resolver barreras de comunicación en ventanillas de atención para personas con discapacidad auditiva.

**Características:**
- Detección de puntos clave con Google MediaPipe Hands (21 keypoints 3D).
- Clasificación de señas mediante modelo RandomForest de Scikit-Learn.
- Búfer de texto acumulativo con estabilizador temporal y limpieza instantánea.
"""

# Versión actual de la API
API_VERSION = "1.0.0"

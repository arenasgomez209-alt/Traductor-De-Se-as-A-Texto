# Importamos os para gestión de rutas de archivos en el sistema operativo
import os
# Importamos urllib.request para descargar automáticamente el modelo de MediaPipe si es necesario
import urllib.request
# Importamos typing para tipar listas, tuplas y opciones
from typing import List, Dict, Any, Optional, Tuple
# Importamos OpenCV para procesamiento de imágenes y dibujo de esqueleto
import cv2
# Importamos NumPy para manipulación matricial de imágenes
import numpy as np
# Importamos mediapipe para la detección de puntos clave de la mano
import mediapipe as mp
# Importamos las tareas de visión de MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Conexiones estándar de los 21 puntos clave de la mano para dibujar el esqueleto
HAND_CONNECTIONS = [
    # Conexiones del pulgar (muñeca -> base pulgar -> articulaciones -> punta)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Conexiones del dedo índice
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Conexiones del dedo medio
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Conexiones del dedo anular
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Conexiones del dedo meñique
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# URL oficial de Google para descargar el modelo TFLite de puntos clave de manos
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Clase principal para encapsular la detección de manos con MediaPipe
class HandDetector:
    """
    Detector de manos que utiliza Google MediaPipe HandLandmarker.
    Descarga el modelo si no existe y extrae los 21 puntos clave de cada mano.
    """
    # Constructor de la clase
    def __init__(self, model_path: Optional[str] = None, max_num_hands: int = 1, min_detection_confidence: float = 0.5):
        # Si no se especifica ruta, definimos una por defecto dentro de backend/models
        if model_path is None:
            # Obtenemos el directorio base del proyecto
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Ruta hacia la carpeta models
            models_dir = os.path.join(base_dir, "models")
            # Creamos la carpeta models si aún no existe
            os.makedirs(models_dir, exist_ok=True)
            # Ruta completa al archivo del modelo .task
            self.model_path = os.path.join(models_dir, "hand_landmarker.task")
        else:
            # Usamos la ruta provista por el usuario
            self.model_path = model_path

        # Aseguramos que el archivo del modelo esté presente en disco
        self._ensure_model_exists()

        # Leemos el modelo directamente en memoria como bytes binarios
        # Esto previene errores de codificación de rutas en C++ cuando la ruta contiene caracteres como 'ñ'
        with open(self.model_path, 'rb') as f:
            model_bytes = f.read()

        # Opciones base de configuración pasando el buffer de bytes en memoria
        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        # Opciones específicas del detector de manos
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        # Creamos la instancia del detector de manos de MediaPipe
        self.detector = vision.HandLandmarker.create_from_options(options)

    # Método privado para verificar y descargar el modelo si no existe
    def _ensure_model_exists(self) -> None:
        # Verificamos si el archivo ya existe en disco
        if not os.path.exists(self.model_path):
            # Imprimimos mensaje informativo de descarga
            print(f"[HandDetector] Descargando modelo MediaPipe desde {MODEL_URL}...")
            # Descargamos el archivo desde Google Storage
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            # Notificamos éxito en la descarga
            print(f"[HandDetector] Modelo descargado con éxito en: {self.model_path}")

    # Método principal para procesar un frame OpenCV y extraer los puntos clave
    def detect(self, bgr_image: np.ndarray) -> List[List[Dict[str, float]]]:
        """
        Detecta manos en una imagen en formato BGR (OpenCV).
        Retorna una lista de manos, donde cada mano contiene 21 diccionarios con {'x', 'y', 'z'}.
        """
        # Validamos que la imagen no sea nula
        if bgr_image is None or bgr_image.size == 0:
            # Si la imagen es inválida, retornamos lista vacía
            return []

        # Convertimos la imagen de BGR a RGB porque MediaPipe trabaja en RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Creamos un objeto mp.Image requerido por el nuevo MediaPipe Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Ejecutamos la inferencia sobre la imagen
        detection_result = self.detector.detect(mp_image)

        # Lista para almacenar todas las manos detectadas
        all_hands = []

        # Verificamos si se detectaron puntos clave en al menos una mano
        if detection_result.hand_landmarks:
            # Iteramos sobre cada mano detectada
            for hand_landmarks in detection_result.hand_landmarks:
                # Lista para los 21 puntos de esta mano
                hand_points = []
                # Iteramos sobre cada uno de los 21 puntos clave
                for lm in hand_landmarks:
                    # Agregamos las coordenadas normalizadas x, y, z
                    hand_points.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z)
                    })
                # Añadimos la mano procesada a la lista general
                all_hands.append(hand_points)

        # Retornamos la lista de manos encontradas
        return all_hands

    # Método para dibujar el esqueleto y articulaciones sobre una imagen OpenCV
    def draw_landmarks(self, image: np.ndarray, landmarks_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Dibuja los puntos y líneas de conexión de la mano sobre la imagen provista.
        Retorna la imagen con el esqueleto dibujado.
        """
        # Obtenemos dimensiones de la imagen (alto, ancho)
        h, w = image.shape[:2]

        # Convertimos coordenadas normalizadas (0 a 1) a píxeles enteros
        pixel_points = []
        # Iteramos cada punto clave
        for pt in landmarks_list:
            # Multiplicamos coordenada normalizada por el ancho y alto
            px = int(pt['x'] * w)
            py = int(pt['y'] * h)
            # Guardamos la tupla de coordenadas en píxeles
            pixel_points.append((px, py))

        # Dibujamos las líneas de conexión ósea entre articulaciones
        for start_idx, end_idx in HAND_CONNECTIONS:
            # Verificamos que ambos índices estén dentro del rango válido
            if start_idx < len(pixel_points) and end_idx < len(pixel_points):
                # Punto de inicio
                p1 = pixel_points[start_idx]
                # Punto final
                p2 = pixel_points[end_idx]
                # Dibujamos la línea de conexión con color cian (255, 200, 0)
                cv2.line(image, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)

        # Dibujamos los círculos en cada articulación (keypoint)
        for i, (px, py) in enumerate(pixel_points):
            # Si es la punta de los dedos (índices 4, 8, 12, 16, 20) usamos color naranja
            if i in [4, 8, 12, 16, 20]:
                # Círculo naranja en la punta
                cv2.circle(image, (px, py), 6, (0, 140, 255), -1, cv2.LINE_AA)
            else:
                # Círculo verde en las articulaciones intermedias
                cv2.circle(image, (px, py), 4, (0, 255, 128), -1, cv2.LINE_AA)

        # Retornamos la imagen anotada
        return image

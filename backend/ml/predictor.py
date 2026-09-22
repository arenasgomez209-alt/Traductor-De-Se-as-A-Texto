# Importamos os para gestionar rutas del sistema
import os
# Importamos json para cargar metadatos de clases
import json
# Importamos numpy para manejo de arreglos y probabilidades
import numpy as np
# Importamos joblib para cargar el modelo Random Forest entrenado
import joblib
# Importamos Union, Dict y Any para tipado de funciones
from typing import Union, Dict, Any, List, Optional
# Importamos el extractor y normalizador de características
from backend.ml.feature_extractor import normalize_landmarks

# Clase que encapsula la inferencia y predicción del modelo
class GesturePredictor:
    """
    Clasificador de señas basado en RandomForestClassifier.
    Recibe puntos clave de la mano, extrae características invariantes
    y devuelve la letra predicha junto con su nivel de confianza.
    """
    # Constructor de la clase
    def __init__(self, model_path: Optional[str] = None, classes_path: Optional[str] = None):
        # Directorio base del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Directorio de modelos
        models_dir = os.path.join(base_dir, "models")

        # Asignamos ruta del modelo si no se especificó
        if model_path is None:
            self.model_path = os.path.join(models_dir, "gesture_model.joblib")
        else:
            self.model_path = model_path

        # Asignamos ruta de clases si no se especificó
        if classes_path is None:
            self.classes_path = os.path.join(models_dir, "label_classes.json")
        else:
            self.classes_path = classes_path

        # Verificamos si el modelo ya está entrenado
        if not os.path.exists(self.model_path):
            print(f"[GesturePredictor] Modelo no encontrado en {self.model_path}. Iniciando entrenamiento automático...")
            # Importamos el script de entrenamiento
            from backend.ml.train_model import train_and_save_model
            # Ejecutamos el entrenamiento para generar el modelo
            train_and_save_model(models_dir)

        # Registramos la fecha de última modificación para recarga dinámica
        self.last_load_time = os.path.getmtime(self.model_path) if os.path.exists(self.model_path) else 0

        # Cargamos el modelo clasificador con joblib
        self.model = joblib.load(self.model_path)

        # Cargamos las clases del archivo JSON
        if os.path.exists(self.classes_path):
            with open(self.classes_path, 'r', encoding='utf-8') as f:
                self.classes = json.load(f)
        else:
            # Si no existe el archivo JSON, extraemos las clases directamente del modelo
            self.classes = list(self.model.classes_)

    # Método para verificar si el modelo fue reentrenado y recargarlo en memoria
    def reload_if_updated(self) -> None:
        if os.path.exists(self.model_path):
            current_mtime = os.path.getmtime(self.model_path)
            if current_mtime > self.last_load_time:
                print("[GesturePredictor] Detectada actualización del modelo en disco. Recargando...")
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.classes_path):
                    with open(self.classes_path, 'r', encoding='utf-8') as f:
                        self.classes = json.load(f)
                else:
                    self.classes = list(self.model.classes_)
                self.last_load_time = current_mtime
                print("[GesturePredictor] ¡Nuevo modelo recargado exitosamente en memoria!")

    # Método principal para clasificar puntos clave de la mano
    def predict(self, raw_landmarks: Union[List[Dict[str, float]], List[List[float]], np.ndarray, List[float]]) -> Dict[str, Any]:
        """
        Clasifica una pose de mano a partir de sus 21 puntos clave.
        Retorna un diccionario con formato:
        {
            "letter": "A",
            "confidence": 0.92
        }
        """
        # Verificamos si hubo reentrenamiento reciente y recargamos automáticamente
        self.reload_if_updated()

        # Extraemos el vector de características normalizado (dimensión 74)
        features = normalize_landmarks(raw_landmarks)

        # Reorganizamos el vector como una matriz de 1 fila (1, 63)
        sample = features.reshape(1, -1)

        # Obtenemos las probabilidades para cada clase mediante predict_proba
        probabilities = self.model.predict_proba(sample)[0]

        # Obtenemos el índice de la clase con mayor probabilidad
        top_index = int(np.argmax(probabilities))

        # Obtenemos la etiqueta de la clase ganadora
        predicted_letter = str(self.model.classes_[top_index])

        # Obtenemos el valor de confianza y lo redondeamos a 2 decimales
        confidence = float(np.round(probabilities[top_index], 2))

        # Retornamos el resultado con el contrato exacto requerido
        return {
            "letter": predicted_letter,
            "confidence": confidence
        }

# Importamos os para crear directorios y rutas
import os
# Importamos json para guardar la lista de clases ordenadas y cargar dataset real
import json
# Importamos numpy para cálculos de coordenadas, rotaciones y ruido
import numpy as np
# Importamos joblib para serializar el modelo entrenado de Scikit-Learn
import joblib
# Importamos RandomForestClassifier desde sklearn
from sklearn.ensemble import RandomForestClassifier
# Importamos train_test_split para evaluar el modelo
from sklearn.model_selection import train_test_split
# Importamos métricas de evaluación
from sklearn.metrics import accuracy_score
# Importamos el normalizador de puntos clave
from backend.ml.feature_extractor import normalize_landmarks

# Función para generar la estructura anatómica base de la mano
def build_base_hand():
    """
    Construye una mano neutra con 21 puntos clave tridimensionales.
    Punto 0 es la muñeca y los demás representan falanges y nudillos.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    # Muñeca en el origen
    lm[0] = [0.0, 0.0, 0.0]
    # Pulgar (1, 2, 3, 4)
    lm[1] = [-0.15, 0.12, -0.02]
    lm[2] = [-0.25, 0.22, -0.03]
    lm[3] = [-0.32, 0.30, -0.04]
    lm[4] = [-0.36, 0.38, -0.05]
    # Dedo Índice (5, 6, 7, 8)
    lm[5] = [-0.10, 0.35, 0.0]
    lm[6] = [-0.11, 0.48, 0.0]
    lm[7] = [-0.12, 0.58, 0.0]
    lm[8] = [-0.12, 0.68, 0.0]
    # Dedo Medio (9, 10, 11, 12)
    lm[9] = [0.0, 0.37, 0.0]
    lm[10] = [0.0, 0.52, 0.0]
    lm[11] = [0.0, 0.63, 0.0]
    lm[12] = [0.0, 0.74, 0.0]
    # Dedo Anular (13, 14, 15, 16)
    lm[13] = [0.09, 0.34, 0.0]
    lm[14] = [0.10, 0.47, 0.0]
    lm[15] = [0.11, 0.57, 0.0]
    lm[16] = [0.11, 0.67, 0.0]
    # Dedo Meñique (17, 18, 19, 20)
    lm[17] = [0.17, 0.28, 0.0]
    lm[18] = [0.20, 0.38, 0.0]
    lm[19] = [0.22, 0.46, 0.0]
    lm[20] = [0.24, 0.54, 0.0]
    return lm

# Función para flexionar o doblar un dedo hacia la palma
def curl_finger(lm, base_idx, fold_factor=1.0):
    knuckle_y = lm[base_idx][1]
    lm[base_idx + 1] = [lm[base_idx][0] * 0.9, knuckle_y + 0.06 * (1 - fold_factor), -0.10 * fold_factor]
    lm[base_idx + 2] = [lm[base_idx][0] * 0.8, knuckle_y + 0.02 * (1 - fold_factor), -0.15 * fold_factor]
    lm[base_idx + 3] = [lm[base_idx][0] * 0.7, knuckle_y - 0.05 * fold_factor, -0.12 * fold_factor]

# Generador anatómico preciso de cada letra del abecedario en lengua de señas
def get_canonical_sign(letter: str) -> np.ndarray:
    lm = build_base_hand()

    # A: Puño cerrado, pulgar vertical apoyado al lado del índice
    if letter == 'A':
        curl_finger(lm, 5)
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[1] = [-0.14, 0.15, -0.05]
        lm[2] = [-0.18, 0.25, -0.06]
        lm[3] = [-0.16, 0.35, -0.07]
        lm[4] = [-0.13, 0.42, -0.05]

    # B: Mano abierta, cuatro dedos juntos verticales, pulgar doblado sobre la palma
    elif letter == 'B':
        lm[1] = [-0.10, 0.12, -0.05]
        lm[2] = [-0.05, 0.18, -0.12]
        lm[3] = [0.00, 0.22, -0.14]
        lm[4] = [0.04, 0.23, -0.15]

    # C: Dedos y pulgar curvados formando una 'C'
    elif letter == 'C':
        for idx in [5, 9, 13, 17]:
            lm[idx + 1][2] = -0.12
            lm[idx + 2][2] = -0.18
            lm[idx + 3][2] = -0.15
            lm[idx + 3][1] = lm[idx][1] + 0.15
        lm[3] = [-0.20, 0.20, -0.12]
        lm[4] = [-0.12, 0.15, -0.15]

    # D: Índice vertical, los demás dedos doblados formando un círculo con el pulgar
    elif letter == 'D':
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[3] = [-0.08, 0.25, -0.10]
        lm[4] = [0.00, 0.28, -0.12]

    # E: Todos los dedos doblados con las puntas tocando el pulgar
    elif letter == 'E':
        curl_finger(lm, 5, fold_factor=1.2)
        curl_finger(lm, 9, fold_factor=1.2)
        curl_finger(lm, 13, fold_factor=1.2)
        curl_finger(lm, 17, fold_factor=1.2)
        lm[3] = [-0.08, 0.16, -0.15]
        lm[4] = [0.02, 0.18, -0.16]

    # F: Índice y pulgar tocándose en círculo (OK), otros 3 dedos extendidos arriba
    elif letter == 'F':
        lm[6] = [-0.15, 0.38, -0.08]
        lm[7] = [-0.20, 0.30, -0.12]
        lm[8] = [-0.18, 0.24, -0.12]
        lm[3] = [-0.22, 0.20, -0.08]
        lm[4] = [-0.18, 0.24, -0.12]

    # G: Índice señalando hacia un lado, pulgar paralelo, otros dedos cerrados
    elif letter == 'G':
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[6] = [-0.18, 0.38, -0.05]
        lm[7] = [-0.28, 0.38, -0.05]
        lm[8] = [-0.38, 0.38, -0.05]
        lm[3] = [-0.22, 0.30, -0.05]
        lm[4] = [-0.32, 0.30, -0.05]

    # H: Índice y medio juntos horizontales, otros dedos cerrados
    elif letter == 'H':
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[6] = [-0.18, 0.38, -0.05]
        lm[7] = [-0.28, 0.38, -0.05]
        lm[8] = [-0.38, 0.38, -0.05]
        lm[10] = [-0.16, 0.33, -0.05]
        lm[11] = [-0.26, 0.33, -0.05]
        lm[12] = [-0.36, 0.33, -0.05]
        lm[4] = [-0.12, 0.25, -0.08]

    # I: Meñique levantado, los demás dedos cerrados en puño
    elif letter == 'I':
        curl_finger(lm, 5)
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        lm[4] = [-0.05, 0.25, -0.10]

    # K: Índice arriba, medio inclinado hacia adelante, pulgar entre ambos
    elif letter == 'K':
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[10] = [0.0, 0.45, -0.10]
        lm[11] = [0.0, 0.52, -0.20]
        lm[12] = [0.0, 0.58, -0.30]
        lm[4] = [-0.05, 0.40, -0.05]

    # L: Pulgar e índice extendidos formando una 'L'
    elif letter == 'L':
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[1] = [-0.18, 0.15, -0.02]
        lm[2] = [-0.28, 0.18, -0.03]
        lm[3] = [-0.38, 0.20, -0.04]
        lm[4] = [-0.48, 0.22, -0.05]

    # M: Tres dedos (índice, medio, anular) doblados sobre el pulgar oculto
    elif letter == 'M':
        curl_finger(lm, 5, fold_factor=1.1)
        curl_finger(lm, 9, fold_factor=1.1)
        curl_finger(lm, 13, fold_factor=1.1)
        curl_finger(lm, 17)
        lm[4] = [0.08, 0.22, -0.05]

    # N: Dos dedos (índice y medio) doblados sobre el pulgar oculto
    elif letter == 'N':
        curl_finger(lm, 5, fold_factor=1.1)
        curl_finger(lm, 9, fold_factor=1.1)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[4] = [0.00, 0.22, -0.05]

    # O: Todos los dedos curvados tocando la punta del pulgar formando un círculo
    elif letter == 'O':
        for idx in [5, 9, 13, 17]:
            lm[idx + 1] = [lm[idx][0] * 0.9, lm[idx][1] + 0.10, -0.14]
            lm[idx + 2] = [lm[idx][0] * 0.7, lm[idx][1] + 0.08, -0.22]
            lm[idx + 3] = [-0.08, 0.26, -0.20]
        lm[3] = [-0.14, 0.22, -0.14]
        lm[4] = [-0.08, 0.26, -0.20]

    # R: Índice y medio cruzados extendidos hacia arriba
    elif letter == 'R':
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[4] = [-0.05, 0.25, -0.10]
        # Cruzamos índice sobre medio
        lm[8] = [0.02, 0.68, 0.05]
        lm[12] = [-0.04, 0.72, -0.05]

    # S: Puño cerrado con pulgar cruzado por encima de los 4 dedos
    elif letter == 'S':
        curl_finger(lm, 5)
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[3] = [0.00, 0.26, 0.05]
        lm[4] = [0.08, 0.26, 0.05]

    # U: Índice y medio juntos extendidos hacia arriba
    elif letter == 'U':
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[4] = [-0.05, 0.25, -0.10]
        lm[7][0] = -0.04
        lm[8][0] = -0.03
        lm[11][0] = 0.02
        lm[12][0] = 0.03

    # V: Índice y medio separados en 'V' (signo de la paz)
    elif letter == 'V':
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[4] = [-0.05, 0.25, -0.10]
        lm[7][0] = -0.18
        lm[8][0] = -0.25
        lm[11][0] = 0.12
        lm[12][0] = 0.20

    # W: Índice, medio y anular en abanico
    elif letter == 'W':
        curl_finger(lm, 17)
        lm[4] = [0.10, 0.22, -0.10]
        lm[8][0] = -0.22
        lm[12][0] = 0.0
        lm[16][0] = 0.22

    # Y: Pulgar y meñique abiertos hacia los extremos (shaka)
    elif letter == 'Y':
        curl_finger(lm, 5)
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        lm[2] = [-0.28, 0.18, -0.03]
        lm[3] = [-0.38, 0.22, -0.04]
        lm[4] = [-0.48, 0.25, -0.05]
        lm[18] = [0.24, 0.36, 0.0]
        lm[19] = [0.32, 0.44, 0.0]
        lm[20] = [0.40, 0.52, 0.0]

    # ESPACIO: Mano extendida horizontalmente
    elif letter == 'ESPACIO':
        for i in range(1, 21):
            lm[i][1] = lm[i][1] * 0.4
            lm[i][2] = -0.05

    return lm

# Función para aumentar datos mediante rotaciones 3D, escala y ruido
def augment_landmarks(base_lm: np.ndarray, num_samples: int = 300) -> np.ndarray:
    samples = []
    for _ in range(num_samples):
        lm = base_lm.copy()

        # Rotaciones ligeras en 3D
        theta_z = np.radians(np.random.uniform(-20.0, 20.0))
        theta_x = np.radians(np.random.uniform(-15.0, 15.0))
        theta_y = np.radians(np.random.uniform(-15.0, 15.0))

        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z),  np.cos(theta_z), 0],
            [0,               0,                1]
        ], dtype=np.float32)

        Rx = np.array([
            [1, 0,                0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x),  np.cos(theta_x)]
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(theta_y),  0, np.sin(theta_y)],
            [0,                1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ], dtype=np.float32)

        R = np.dot(Rz, np.dot(Rx, Ry))
        lm = np.dot(lm, R.T)

        # Escala aleatoria
        scale = np.random.uniform(0.80, 1.20)
        lm = lm * scale

        # Ruido gaussiano
        noise = np.random.normal(0.0, 0.015, size=lm.shape).astype(np.float32)
        lm = lm + noise

        # Extraemos vector de 74 características anatómicas e invariantes
        features = normalize_landmarks(lm)
        samples.append(features)

    return np.array(samples, dtype=np.float32)

# Función principal para entrenar y guardar el modelo RandomForest
def train_and_save_model(output_dir: str = None) -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "models")
    os.makedirs(output_dir, exist_ok=True)

    # Letras del abecedario soportadas
    classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        'K', 'L', 'M', 'N', 'O', 'R', 'S', 'U', 'V', 'W', 'Y', 'ESPACIO'
    ]
    print(f"[Entrenamiento] Generando dataset anatómico para {len(classes)} letras de señas...")

    X_list = []
    y_list = []

    # 1. Muestras aumentadas sintéticas
    for label in classes:
        canonical_pose = get_canonical_sign(label)
        samples = augment_landmarks(canonical_pose, num_samples=300)
        X_list.append(samples)
        y_list.extend([label] * len(samples))

    # 2. Muestras reales capturadas por el usuario si existen en backend/data/dataset.json
    real_data_path = os.path.join(base_dir, "data", "dataset.json")
    if os.path.exists(real_data_path):
        try:
            with open(real_data_path, 'r', encoding='utf-8') as f:
                real_dataset = json.load(f)
            real_count = 0
            for item in real_dataset:
                lbl = item.get("label")
                lm_raw = item.get("landmarks")
                if lbl and lm_raw:
                    feat = normalize_landmarks(lm_raw)
                    X_list.append(feat.reshape(1, -1))
                    y_list.append(lbl)
                    real_count += 1
            print(f"[Entrenamiento] Se incorporaron {real_count} muestras reales capturadas por el usuario.")
        except Exception as e:
            print(f"[Entrenamiento] Aviso al leer dataset real: {e}")

    X = np.vstack(X_list)
    y = np.array(y_list)

    print(f"[Entrenamiento] Total de muestras: {X.shape[0]} con {X.shape[1]} características por muestra.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[Entrenamiento] Entrenando RandomForestClassifier (120 estimadores)...")
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Entrenamiento] ¡Exactitud global del modelo: {acc * 100:.2f}%!")

    # Guardar modelo
    model_file = os.path.join(output_dir, "gesture_model.joblib")
    joblib.dump(clf, model_file)
    print(f"[Entrenamiento] Guardado en: {model_file}")

    # Guardar clases
    classes_file = os.path.join(output_dir, "label_classes.json")
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(list(clf.classes_), f, ensure_ascii=False, indent=2)
    print(f"[Entrenamiento] Clases guardadas en: {classes_file}")

if __name__ == "__main__":
    train_and_save_model()

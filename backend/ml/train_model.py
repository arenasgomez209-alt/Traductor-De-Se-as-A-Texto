# Importamos os para crear directorios y rutas
import os
# Importamos json para guardar la lista de clases ordenadas
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
from sklearn.metrics import classification_report, accuracy_score
# Importamos el normalizador de puntos clave
from backend.ml.feature_extractor import normalize_landmarks

# Función para generar la estructura anatómica base de la mano
def build_base_hand():
    """
    Construye una mano neutra con 21 puntos clave tridimensionales.
    Punto 0 es la muñeca y los demás representan falanges y nudillos.
    """
    # Arreglo de 21 puntos con 3 coordenadas (x, y, z)
    lm = np.zeros((21, 3), dtype=np.float32)
    # Muñeca en el origen
    lm[0] = [0.0, 0.0, 0.0]
    # Pulgar (articulación carpometacarpiana, metacarpofalángica, interfalángica y punta)
    lm[1] = [-0.15, 0.12, -0.02]
    lm[2] = [-0.25, 0.22, -0.03]
    lm[3] = [-0.32, 0.30, -0.04]
    lm[4] = [-0.36, 0.38, -0.05]
    # Dedo Índice (nudillo base, falange proximal, falange media, punta)
    lm[5] = [-0.10, 0.35, 0.0]
    lm[6] = [-0.11, 0.48, 0.0]
    lm[7] = [-0.12, 0.58, 0.0]
    lm[8] = [-0.12, 0.68, 0.0]
    # Dedo Medio
    lm[9] = [0.0, 0.37, 0.0]
    lm[10] = [0.0, 0.52, 0.0]
    lm[11] = [0.0, 0.63, 0.0]
    lm[12] = [0.0, 0.74, 0.0]
    # Dedo Anular
    lm[13] = [0.09, 0.34, 0.0]
    lm[14] = [0.10, 0.47, 0.0]
    lm[15] = [0.11, 0.57, 0.0]
    lm[16] = [0.11, 0.67, 0.0]
    # Dedo Meñique
    lm[17] = [0.17, 0.28, 0.0]
    lm[18] = [0.20, 0.38, 0.0]
    lm[19] = [0.22, 0.46, 0.0]
    lm[20] = [0.24, 0.54, 0.0]
    # Retornamos los puntos base
    return lm

# Función para flexionar o cerrar un dedo hacia la palma
def curl_finger(lm, base_idx, fold_factor=1.0):
    """
    Dobla las 3 articulaciones de un dedo hacia la palma de la mano.
    """
    # Nudillo base del dedo
    knuckle_y = lm[base_idx][1]
    # Falange proximal flexionada
    lm[base_idx + 1] = [lm[base_idx][0] * 0.9, knuckle_y + 0.06 * (1 - fold_factor), -0.10 * fold_factor]
    # Falange intermedia flexionada
    lm[base_idx + 2] = [lm[base_idx][0] * 0.8, knuckle_y + 0.02 * (1 - fold_factor), -0.15 * fold_factor]
    # Punta del dedo tocando la palma
    lm[base_idx + 3] = [lm[base_idx][0] * 0.7, knuckle_y - 0.05 * fold_factor, -0.12 * fold_factor]

# Generador de posturas canónicas para cada letra de señas
def get_canonical_sign(letter: str) -> np.ndarray:
    """
    Retorna la posición espacial de los 21 puntos clave según la seña de la letra.
    """
    # Obtenemos la estructura anatómica de la mano
    lm = build_base_hand()

    # Letra A: Puño cerrado, cuatro dedos doblados, pulgar extendido al costado del índice
    if letter == 'A':
        curl_finger(lm, 5)   # Índice doblado
        curl_finger(lm, 9)   # Medio doblado
        curl_finger(lm, 13)  # Anular doblado
        curl_finger(lm, 17)  # Meñique doblado
        lm[1] = [-0.14, 0.15, -0.05]
        lm[2] = [-0.18, 0.25, -0.06]
        lm[3] = [-0.16, 0.35, -0.07]
        lm[4] = [-0.13, 0.42, -0.05]  # Pulgar vertical al lado del puño

    # Letra B: Mano abierta, cuatro dedos juntos verticales, pulgar doblado sobre la palma
    elif letter == 'B':
        # Los 4 dedos quedan extendidos verticalmente
        # Pulgar doblado cruzando la palma
        lm[1] = [-0.10, 0.12, -0.05]
        lm[2] = [-0.05, 0.18, -0.12]
        lm[3] = [0.00, 0.22, -0.14]
        lm[4] = [0.04, 0.23, -0.15]

    # Letra C: Mano curvada formando la letra C con todos los dedos y el pulgar
    elif letter == 'C':
        for idx in [5, 9, 13, 17]:
            lm[idx + 1][2] = -0.12
            lm[idx + 2][2] = -0.18
            lm[idx + 3][2] = -0.15
            lm[idx + 3][1] = lm[idx][1] + 0.15
        lm[3] = [-0.20, 0.20, -0.12]
        lm[4] = [-0.12, 0.15, -0.15]

    # Letra D: Dedo índice apuntando hacia arriba, los demás dedos forman un círculo con el pulgar
    elif letter == 'D':
        # Índice extendido hacia arriba (queda por defecto)
        curl_finger(lm, 9)   # Medio doblado
        curl_finger(lm, 13)  # Anular doblado
        curl_finger(lm, 17)  # Meñique doblado
        # Pulgar tocando el dedo medio
        lm[3] = [-0.08, 0.25, -0.10]
        lm[4] = [0.00, 0.28, -0.12]

    # Letra E: Todos los dedos doblados con las puntas tocando el pulgar doblado
    elif letter == 'E':
        curl_finger(lm, 5, fold_factor=1.2)
        curl_finger(lm, 9, fold_factor=1.2)
        curl_finger(lm, 13, fold_factor=1.2)
        curl_finger(lm, 17, fold_factor=1.2)
        lm[3] = [-0.08, 0.16, -0.15]
        lm[4] = [0.02, 0.18, -0.16]

    # Letra F: Pulgar e índice formando un círculo (seña de OK), otros 3 dedos extendidos
    elif letter == 'F':
        # Índice doblado hacia el pulgar
        lm[6] = [-0.15, 0.38, -0.08]
        lm[7] = [-0.20, 0.30, -0.12]
        lm[8] = [-0.18, 0.24, -0.12]
        # Pulgar tocando la punta del índice
        lm[3] = [-0.22, 0.20, -0.08]
        lm[4] = [-0.18, 0.24, -0.12]
        # Medio, Anular y Meñique permanecen extendidos

    # Letra G: Índice señalando hacia el frente horizontalmente, pulgar paralelo
    elif letter == 'G':
        curl_finger(lm, 9)
        curl_finger(lm, 13)
        curl_finger(lm, 17)
        lm[6] = [-0.18, 0.38, -0.05]
        lm[7] = [-0.26, 0.38, -0.05]
        lm[8] = [-0.35, 0.38, -0.05] # Índice horizontal hacia un lado
        lm[3] = [-0.22, 0.30, -0.05]
        lm[4] = [-0.30, 0.30, -0.05] # Pulgar paralelo

    # Letra H: Índice y medio juntos extendidos horizontalmente
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

    # Letra I: Meñique levantado hacia arriba, todos los demás dedos cerrados en puño
    elif letter == 'I':
        curl_finger(lm, 5)   # Índice doblado
        curl_finger(lm, 9)   # Medio doblado
        curl_finger(lm, 13)  # Anular doblado
        lm[4] = [-0.05, 0.25, -0.10] # Pulgar sobre el puño
        # Meñique queda levantado hacia arriba

    # Letra L: Pulgar e índice extendidos a 90 grados formando una 'L', los demás doblados
    elif letter == 'L':
        curl_finger(lm, 9)   # Medio doblado
        curl_finger(lm, 13)  # Anular doblado
        curl_finger(lm, 17)  # Meñique doblado
        # Índice apuntando verticalmente arriba
        # Pulgar apuntando hacia afuera
        lm[1] = [-0.18, 0.15, -0.02]
        lm[2] = [-0.28, 0.18, -0.03]
        lm[3] = [-0.38, 0.20, -0.04]
        lm[4] = [-0.48, 0.22, -0.05]

    # Letra O: Todos los dedos redondeados tocando las puntas formando una 'O'
    elif letter == 'O':
        for idx in [5, 9, 13, 17]:
            lm[idx + 1] = [lm[idx][0] * 0.9, lm[idx][1] + 0.10, -0.14]
            lm[idx + 2] = [lm[idx][0] * 0.7, lm[idx][1] + 0.08, -0.22]
            lm[idx + 3] = [-0.08, 0.26, -0.20]
        lm[3] = [-0.14, 0.22, -0.14]
        lm[4] = [-0.08, 0.26, -0.20]

    # Letra U: Índice y medio levantados juntos hacia arriba, anular y meñique doblados
    elif letter == 'U':
        curl_finger(lm, 13)  # Anular doblado
        curl_finger(lm, 17)  # Meñique doblado
        lm[4] = [-0.05, 0.25, -0.10] # Pulgar sobre anular
        # Juntamos índice y medio
        lm[7][0] = -0.04
        lm[8][0] = -0.03
        lm[11][0] = 0.02
        lm[12][0] = 0.03

    # Letra V: Índice y medio levantados separados en forma de 'V', los demás dedos doblados
    elif letter == 'V':
        curl_finger(lm, 13)  # Anular doblado
        curl_finger(lm, 17)  # Meñique doblado
        lm[4] = [-0.05, 0.25, -0.10] # Pulgar cerrando dedos
        # Separamos índice y medio
        lm[7][0] = -0.18
        lm[8][0] = -0.24
        lm[11][0] = 0.12
        lm[12][0] = 0.18

    # Letra W: Tres dedos extendidos (índice, medio, anular), pulgar sosteniendo el meñique
    elif letter == 'W':
        curl_finger(lm, 17)  # Meñique doblado
        lm[4] = [0.10, 0.22, -0.10] # Pulgar sostiene meñique
        # Índice, medio y anular en abanico
        lm[8][0] = -0.20
        lm[12][0] = 0.0
        lm[16][0] = 0.20

    # Letra Y: Pulgar y meñique extendidos hacia afuera, los tres dedos del centro doblados
    elif letter == 'Y':
        curl_finger(lm, 5)   # Índice doblado
        curl_finger(lm, 9)   # Medio doblado
        curl_finger(lm, 13)  # Anular doblado
        # Pulgar abierto hacia la izquierda
        lm[2] = [-0.28, 0.18, -0.03]
        lm[3] = [-0.38, 0.22, -0.04]
        lm[4] = [-0.48, 0.25, -0.05]
        # Meñique abierto hacia la derecha
        lm[18] = [0.24, 0.36, 0.0]
        lm[19] = [0.32, 0.44, 0.0]
        lm[20] = [0.40, 0.52, 0.0]

    # Gesto ESPACIO: Mano abierta plana orientada horizontalmente para separar palabras
    elif letter == 'ESPACIO':
        # Toda la mano extendida con rotación plana
        for i in range(1, 21):
            lm[i][1] = lm[i][1] * 0.4
            lm[i][2] = -0.05

    # Retornamos la posición tridimensional calculada
    return lm

# Función para aplicar aumento de datos sintético y robusto (rotaciones 3D, escala, ruido gaussiano)
def augment_landmarks(base_lm: np.ndarray, num_samples: int = 250) -> np.ndarray:
    """
    Genera múltiples variaciones realistas a partir de una postura base:
    - Ruido gaussiano en cada articulación
    - Rotaciones aleatorias en ejes X, Y y Z
    - Variaciones de escala y proporciones óseas
    """
    # Lista para almacenar todas las muestras generadas
    samples = []

    # Generamos la cantidad solicitada de muestras aumentadas
    for _ in range(num_samples):
        # Clonamos la postura base
        lm = base_lm.copy()

        # Generamos ángulos aleatorios de rotación en radianes (-15° a +15°)
        theta_z = np.radians(np.random.uniform(-18.0, 18.0))
        theta_x = np.radians(np.random.uniform(-12.0, 12.0))
        theta_y = np.radians(np.random.uniform(-12.0, 12.0))

        # Matriz de rotación en Z
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z),  np.cos(theta_z), 0],
            [0,               0,                1]
        ], dtype=np.float32)

        # Matriz de rotación en X
        Rx = np.array([
            [1, 0,                0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x),  np.cos(theta_x)]
        ], dtype=np.float32)

        # Matriz de rotación en Y
        Ry = np.array([
            [np.cos(theta_y),  0, np.sin(theta_y)],
            [0,                1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ], dtype=np.float32)

        # Matriz de rotación combinada
        R = np.dot(Rz, np.dot(Rx, Ry))

        # Aplicamos la rotación a los 21 puntos
        lm = np.dot(lm, R.T)

        # Variación aleatoria de escala (entre 85% y 115%)
        scale = np.random.uniform(0.85, 1.15)
        lm = lm * scale

        # Añadimos ruido gaussiano ligero simulando imperfecciones del sensor de cámara
        noise = np.random.normal(0.0, 0.012, size=lm.shape).astype(np.float32)
        lm = lm + noise

        # Traslación aleatoria en el espacio visual
        offset = np.random.uniform(-0.15, 0.15, size=(1, 3)).astype(np.float32)
        lm = lm + offset

        # Normalizamos la postura usando nuestro extractor de características
        features = normalize_landmarks(lm)

        # Guardamos el vector normalizado
        samples.append(features)

    # Convertimos a arreglo NumPy bidimensional (num_samples, 63)
    return np.array(samples, dtype=np.float32)

# Función principal para entrenar y guardar el modelo RandomForest
def train_and_save_model(output_dir: str = None) -> None:
    """
    Crea el conjunto de datos, entrena el RandomForest y guarda el modelo serializado.
    """
    # Si no se especifica directorio, usamos backend/models
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "models")

    # Aseguramos que la carpeta exista
    os.makedirs(output_dir, exist_ok=True)

    # Clases del abecedario de señas y comando de espacio
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'O', 'U', 'V', 'W', 'Y', 'ESPACIO']
    print(f"[Entrenamiento] Generando dataset para {len(classes)} clases de señas...")

    # Listas para almacenar características X y etiquetas y
    X_list = []
    y_list = []

    # Iteramos cada clase generando muestras sintéticas aumentadas
    for label in classes:
        # Obtenemos la postura canónica
        canonical_pose = get_canonical_sign(label)
        # Generamos 350 muestras con variaciones por cada clase
        samples = augment_landmarks(canonical_pose, num_samples=350)
        # Acumulamos características
        X_list.append(samples)
        # Acumulamos etiquetas correspondientes
        y_list.extend([label] * len(samples))

    # Concatenamos todas las muestras en una sola matriz X
    X = np.vstack(X_list)
    # Convertimos etiquetas a arreglo NumPy
    y = np.array(y_list)

    print(f"[Entrenamiento] Total de muestras generadas: {X.shape[0]} con dimensión {X.shape[1]}")

    # Dividimos en conjunto de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[Entrenamiento] Entrenando RandomForestClassifier (100 árboles)...")
    # Instanciamos el clasificador Random Forest de Scikit-Learn
    clf = RandomForestClassifier(
        n_estimators=100,       # Cantidad de árboles en el bosque
        max_depth=16,           # Profundidad máxima para prevenir sobreajuste
        random_state=42,        # Semilla para reproducibilidad
        n_jobs=-1               # Uso de todos los núcleos del procesador
    )

    # Ajustamos el modelo a los datos de entrenamiento
    clf.fit(X_train, y_train)

    # Realizamos predicciones sobre el conjunto de prueba
    y_pred = clf.predict(X_test)
    # Calculamos la precisión global
    acc = accuracy_score(y_test, y_pred)
    print(f"[Entrenamiento] ¡Modelo entrenado con éxito! Exactitud (Accuracy): {acc * 100:.2f}%")

    # Ruta del archivo del modelo serializado
    model_file = os.path.join(output_dir, "gesture_model.joblib")
    # Guardamos el clasificador entrenado con joblib
    joblib.dump(clf, model_file)
    print(f"[Entrenamiento] Modelo guardado en: {model_file}")

    # Ruta del archivo de clases en JSON
    classes_file = os.path.join(output_dir, "label_classes.json")
    # Guardamos la lista de clases en formato JSON
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"[Entrenamiento] Clases guardadas en: {classes_file}")

# Punto de entrada para ejecución directa del script
if __name__ == "__main__":
    # Ejecutamos el entrenamiento
    train_and_save_model()

# Importamos numpy para operaciones vectoriales y matriciales eficientes
import numpy as np
# Importamos List, Union y Dict para tipado estático claro
from typing import List, Union, Dict, Any

# Función principal para extraer características avanzadas e invariantes de los 21 puntos clave
def normalize_landmarks(raw_landmarks: Union[List[Dict[str, float]], List[List[float]], np.ndarray, List[float]]) -> np.ndarray:
    """
    Normaliza 21 puntos clave (x, y, z) de la mano y calcula características anatómicas
    adicionales (ratios de extensión de dedos y distancias entre puntas).
    Retorna un vector NumPy de características de longitud 74, altamente robusto e invariante.
    """
    # Lista temporal para acumular las coordenadas ordenadas
    points = []

    # Verificamos si la entrada es una lista de diccionarios [{'x':.., 'y':.., 'z':..}]
    if isinstance(raw_landmarks, list) and len(raw_landmarks) > 0 and isinstance(raw_landmarks[0], dict):
        # Iteramos cada diccionario extrayendo x, y, z
        for pt in raw_landmarks:
            # Añadimos la terna [x, y, z] a la lista de puntos
            points.append([float(pt.get('x', 0.0)), float(pt.get('y', 0.0)), float(pt.get('z', 0.0))])

    # Verificamos si es una lista plana de 63 valores consecutivos [x0, y0, z0, x1, y1, z1, ...]
    elif isinstance(raw_landmarks, (list, np.ndarray)) and len(raw_landmarks) == 63 and not isinstance(raw_landmarks[0], (list, dict, np.ndarray)):
        # Convertimos a arreglo NumPy
        flat_arr = np.array(raw_landmarks, dtype=np.float32)
        # Redimensionamos a 21 filas y 3 columnas (x, y, z)
        points = flat_arr.reshape(21, 3).tolist()

    # Verificamos si es una lista de listas o arreglo bidimensional (21, 3)
    elif isinstance(raw_landmarks, (list, np.ndarray)):
        # Iteramos cada elemento asegurando que tenga 3 dimensiones
        for pt in raw_landmarks:
            # Extraemos coordenadas x, y, z
            points.append([float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0])

    # Convertimos los puntos recopilados a una matriz NumPy de tipo float32
    coords = np.array(points, dtype=np.float32)

    # Validamos que exactamente tengamos 21 puntos clave de la mano (estándar MediaPipe)
    if coords.shape[0] != 21:
        # Si no hay 21 puntos, rellenamos o recortamos para mantener consistencia de 21 puntos
        padded = np.zeros((21, 3), dtype=np.float32)
        # Copiamos los puntos disponibles
        n = min(21, coords.shape[0])
        # Asignamos los puntos existentes
        padded[:n] = coords[:n]
        # Reemplazamos coords con la matriz completada
        coords = padded

    # Paso 1: Invariancia a la traslación. Tomamos el punto 0 (muñeca) como origen (0,0,0)
    wrist_point = coords[0].copy()
    # Restamos las coordenadas de la muñeca a todos los 21 puntos
    relative_coords = coords - wrist_point

    # Paso 2: Invariancia a la escala. Calculamos la distancia euclidiana de cada punto al origen
    distances = np.linalg.norm(relative_coords, axis=1)
    # Obtenemos la distancia máxima para usarla como factor de normalización
    max_dist = np.max(distances)

    # Evitamos división por cero si la mano está colapsada o vacía
    if max_dist > 1e-6:
        # Dividimos cada coordenada relativa por la distancia máxima
        normalized_coords = relative_coords / max_dist
    else:
        # Si la distancia máxima es casi cero, conservamos las coordenadas relativas
        normalized_coords = relative_coords

    # Paso 3: Características Anatómicas Invariantes de los Dedos
    # Índices estándar de MediaPipe:
    # Puntas de los 5 dedos: Pulgar(4), Índice(8), Medio(12), Anular(16), Meñique(20)
    # Articulaciones PIP intermedias: Pulgar(2), Índice(6), Medio(10), Anular(14), Meñique(18)
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    # Calculamos el ratio de extensión de cada dedo: distancia(punta, muñeca) / distancia(pip, muñeca)
    # Si el dedo está estirado, el ratio es > 1.2; si está doblado hacia la palma, el ratio es < 0.95
    # Este ratio es 100% inmune a rotaciones, inclinación y escala de la mano
    extension_ratios = []
    for tip_idx, pip_idx in zip(tips, pips):
        d_tip = np.linalg.norm(relative_coords[tip_idx])
        d_pip = np.linalg.norm(relative_coords[pip_idx])
        ratio = float(d_tip / (d_pip + 1e-6))
        extension_ratios.append(ratio)

    # Paso 4: Distancias euclidianas relativas entre puntas de dedos clave
    # Permite diferenciar con precisión señas como:
    # - 'U' (índice y medio juntos) vs 'V' (índice y medio separados)
    # - 'F' y 'O' (pulgar tocando índice) vs 'D' o 'B'
    tip_distances = [
        float(np.linalg.norm(normalized_coords[4] - normalized_coords[8])),   # Pulgar - Índice
        float(np.linalg.norm(normalized_coords[8] - normalized_coords[12])),  # Índice - Medio
        float(np.linalg.norm(normalized_coords[12] - normalized_coords[16])), # Medio - Anular
        float(np.linalg.norm(normalized_coords[16] - normalized_coords[20])), # Anular - Meñique
        float(np.linalg.norm(normalized_coords[4] - normalized_coords[12])),  # Pulgar - Medio
        float(np.linalg.norm(normalized_coords[4] - normalized_coords[20])),  # Pulgar - Meñique
    ]

    # Vector 1: Coordenadas espaciales 3D normalizadas aplanadas (63 valores)
    base_features = normalized_coords.flatten()

    # Concatenamos todo en un vector unificado de 74 características anatómicas
    full_feature_vector = np.concatenate([
        base_features,
        np.array(extension_ratios, dtype=np.float32),
        np.array(tip_distances, dtype=np.float32)
    ])

    # Retornamos el vector de características enriquecido
    return full_feature_vector

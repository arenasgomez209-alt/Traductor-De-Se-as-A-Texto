# ==============================================================================
# CAPTURADOR DE SEÑAS REALES (capturar_señas.py)
# ==============================================================================
# Esta herramienta te permite grabar muestras reales de tus manos con tu propia cámara
# para que el modelo aprenda con exactitud tus señas personales y de tu entorno.

import sys
import os
import time
import json
import cv2
import numpy as np

# Aseguramos que la raíz del proyecto esté en el PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend.ml.hand_detector import HandDetector
from backend.ml.train_model import train_and_save_model

def main():
    print("\n" + "=" * 70)
    print("  HERRAMIENTA DE CAPTURA DE SEÑAS REALES CON TU CÁMARA")
    print("  Instrucciones:")
    print("    1. Coloca tu mano frente a la cámara haciendo una letra (ej. A, B, C, L, O, V).")
    print("    2. Presiona en el teclado la letra que estás haciendo.")
    print("    3. Se capturarán 30 muestras en 1 segundo.")
    print("    4. Puedes capturar tantas letras como quieras.")
    print("    5. Presiona 'T' para reentrenar el modelo con tus datos reales.")
    print("    6. Presiona 'Q' o ESC para salir.")
    print("=" * 70 + "\n")

    # Carpeta donde se guardará el dataset real
    data_dir = os.path.join(CURRENT_DIR, "backend", "data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_file = os.path.join(data_dir, "dataset.json")

    # Cargamos datos existentes si los hay
    existing_samples = []
    if os.path.exists(dataset_file):
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                existing_samples = json.load(f)
            print(f"[Info] Se cargaron {len(existing_samples)} muestras guardadas previamente.")
        except Exception:
            existing_samples = []

    # Inicializamos detector de MediaPipe
    detector = HandDetector()

    # Abrimos cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] No se pudo abrir la cámara web.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Capturador de Senas Reales"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 650)

    capturing_label = None
    samples_needed = 30
    captured_count = 0
    message = "Presiona una letra (A-Z) para empezar a capturar"
    message_color = (255, 255, 255)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Efecto espejo
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detectar mano
            hands = detector.detect(frame)
            hand_present = len(hands) > 0

            if hand_present:
                detector.draw_landmarks(frame, hands[0])

            # Si estamos en modo captura
            if capturing_label and hand_present:
                # Guardamos la muestra
                existing_samples.append({
                    "label": capturing_label,
                    "landmarks": hands[0],
                    "timestamp": time.time()
                })
                captured_count += 1

                # Guardamos en disco cada 10 muestras
                if captured_count >= samples_needed:
                    with open(dataset_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_samples, f, indent=2)
                    message = f"¡Exito! 30 muestras de '{capturing_label}' guardadas. Total acumulado: {len(existing_samples)}"
                    message_color = (0, 255, 128)
                    capturing_label = None
                    captured_count = 0
                else:
                    message = f"Capturando '{capturing_label}': [{captured_count}/{samples_needed}]"
                    message_color = (0, 200, 255)

            # Dibujamos overlay de interfaz
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 30), -1)
            cv2.rectangle(overlay, (0, h - 70), (w, h), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Textos informativos
            cv2.putText(frame, "Grabador de Senas - Pulsa una tecla de letra (A-Z) para grabar", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Pulsa 'T' para REENTRENAR  |  Pulsa 'Q' para SALIR", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

            # Mensaje de estado inferior
            cv2.putText(frame, message, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, message_color, 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break

            # Tecla 'T': Reentrenar el modelo con los datos
            elif key == ord('t') or key == ord('T'):
                print("\n[Reentrenamiento] Entrenando modelo con las muestras recopiladas...")
                message = "Reentrenando modelo, por favor espera unos segundos..."
                message_color = (0, 200, 255)
                # Ejecutamos el reentrenamiento
                train_and_save_model()
                message = "¡Modelo reentrenado con éxito con tus datos reales!"
                message_color = (0, 255, 0)
                print("[Reentrenamiento] ¡Listo!")

            # Teclas alfanuméricas de letras (A-Z)
            elif 65 <= key <= 90 or 97 <= key <= 122:
                letter = chr(key).upper()
                if not hand_present:
                    message = f"Por favor pon tu mano visible frente a la camara para grabar '{letter}'"
                    message_color = (0, 100, 255)
                else:
                    capturing_label = letter
                    captured_count = 0
                    message = f"Iniciando captura para la letra '{letter}'..."
                    message_color = (0, 255, 255)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# ==============================================================================
# EJECUTOR DE CÁMARA EN ESCRITORIO (run_camera_desktop.py)
# ==============================================================================
# Permite probar la detección de señas en tiempo real con la cámara web local
# y una ventana nativa de OpenCV, sin necesidad de abrir un navegador web.

# Importamos sys y os para configuración de rutas
import sys
import os
# Importamos time para cálculo de FPS (cuadros por segundo)
import time
# Importamos OpenCV para captura de video e interfaz gráfica de escritorio
import cv2
# Importamos NumPy para creación de capas gráficas e interfaces
import numpy as np

# Aseguramos que la raíz del proyecto esté en el PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Importamos los módulos internos del proyecto
from backend.ml.hand_detector import HandDetector
from backend.ml.predictor import GesturePredictor
from backend.app.buffer_manager import TextBufferManager
from backend.app.config import MIN_CONFIDENCE_THRESHOLD, STABILITY_THRESHOLD

# Función para dibujar la interfaz de usuario tipo HUD (Heads-Up Display) sobre el frame
def draw_hud(frame: np.ndarray, letter: str, confidence: float, buffer_text: str, fps: float) -> np.ndarray:
    """
    Dibuja un panel moderno con la letra detectada, barra de confianza,
    texto acumulado y atajos de teclado directamente sobre la imagen.
    """
    h, w = frame.shape[:2]

    # Creamos un overlay semi-transparente en la parte superior e inferior
    overlay = frame.copy()

    # Barra superior oscura para título y FPS
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 28), -1)
    # Panel lateral derecho para la letra y confianza
    cv2.rectangle(overlay, (w - 240, 75), (w - 15, 270), (25, 28, 38), -1)
    # Panel inferior para el texto acumulado del cliente
    cv2.rectangle(overlay, (15, h - 90), (w - 15, h - 15), (20, 24, 32), -1)

    # Fusionamos el overlay con transparencia (alpha blending)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Bordes estéticos de los paneles
    cv2.rectangle(frame, (w - 240, 75), (w - 15, 270), (0, 200, 255), 2)
    cv2.rectangle(frame, (15, h - 90), (w - 15, h - 15), (0, 255, 150), 2)

    # Texto del encabezado superior
    cv2.putText(frame, "Traductor de Senas para Atencion al Cliente", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

    # Título de la tarjeta de letra
    cv2.putText(frame, "SENA DETECTADA", (w - 225, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Letra grande detectada
    display_letter = letter if letter != "NINGUNA" else "-"
    color_letter = (0, 255, 128) if confidence >= MIN_CONFIDENCE_THRESHOLD else (100, 100, 220)
    cv2.putText(frame, display_letter, (w - 165, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_letter, 4, cv2.LINE_AA)

    # Texto y barra de progreso de confianza
    conf_pct = int(confidence * 100)
    cv2.putText(frame, f"Certeza: {conf_pct}%", (w - 225, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Fondo de la barra de progreso
    bar_x = w - 225
    bar_y = 230
    bar_w = 195
    bar_h = 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    # Relleno de la barra según el porcentaje
    fill_w = int(bar_w * confidence)
    bar_color = (0, 220, 100) if confidence >= 0.70 else (0, 180, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Panel inferior: Texto acumulado en el búfer
    cv2.putText(frame, "TEXTO ACUMULADO (VENTANILLA):", (30, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)

    # Texto del cliente (con cursor parpadeante)
    cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
    shown_text = (buffer_text + cursor) if buffer_text else f"(Esperando senas...){cursor}"
    cv2.putText(frame, shown_text, (30, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Barra informativa de controles en pantalla
    controls_text = "[C] Limpiar Texto  |  [ESPACIO] Separar  |  [Q] Salir"
    cv2.putText(frame, controls_text, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)

    return frame

# Función principal para ejecutar la ventana de cámara
def main():
    print("\n" + "=" * 70)
    print("  INICIANDO VISOR DE CÁMARA DE ESCRITORIO (OpenCV)")
    print("  Controles interactivos en la ventana:")
    print("    - Tecla 'C': Limpiar texto acumulado")
    print("    - Tecla 'ESPACIO': Añadir un espacio entre palabras")
    print("    - Tecla 'Q' o ESC: Salir del programa")
    print("=" * 70 + "\n")

    # Inicializamos detector y clasificador
    print("[1/3] Cargando detector MediaPipe Hands...")
    detector = HandDetector()

    print("[2/3] Cargando clasificador RandomForest...")
    predictor = GesturePredictor()

    print("[3/3] Inicializando manejador de búfer de texto...")
    buffer_mgr = TextBufferManager(
        min_consecutive_frames=STABILITY_THRESHOLD,
        min_confidence=MIN_CONFIDENCE_THRESHOLD
    )

    # Abrimos la cámara web (índice 0 por defecto)
    print("\nAbriendo dispositivo de cámara...")
    cap = cv2.VideoCapture(0)

    # Verificamos si la cámara pudo abrirse
    if not cap.isOpened():
        print("[AVISO] No se pudo abrir la cámara web física (índice 0).")
        print("Puedes ejecutar 'python run_local_demo.py' para probar todos los endpoints y el modelo en consola.")
        return

    # Ajustamos resolución deseada (1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Variables de control de tiempo y FPS
    prev_time = time.time()
    current_letter = "NINGUNA"
    current_confidence = 0.0

    # Nombre de la ventana OpenCV
    window_name = "Traductor de Senas a Texto - Atencion al Cliente"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 650)

    try:
        # Bucle principal de captura y traducción
        while True:
            # Leemos un fotograma de la cámara
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar fotograma de la cámara.")
                break

            # Volteamos horizontalmente la imagen para efecto espejo natural
            frame = cv2.flip(frame, 1)

            # Calculamos FPS instantáneos
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # Detectamos manos con MediaPipe
            hands = detector.detect(frame)

            # Si detectamos al menos una mano
            if hands and len(hands) > 0:
                landmarks = hands[0]

                # Dibujamos el esqueleto sobre la imagen
                detector.draw_landmarks(frame, landmarks)

                # Clasificamos la pose con RandomForest
                pred = predictor.predict(landmarks)
                current_letter = pred["letter"]
                current_confidence = pred["confidence"]

                # Incorporamos al búfer de texto
                buffer_mgr.add_prediction(current_letter, current_confidence)
            else:
                current_letter = "NINGUNA"
                current_confidence = 0.0

            # Obtenemos el texto acumulado del cliente
            buffer_text = buffer_mgr.get_buffer()

            # Dibujamos el panel de control HUD
            display_frame = draw_hud(frame, current_letter, current_confidence, buffer_text, fps)

            # Mostramos el fotograma en la ventana de escritorio
            cv2.imshow(window_name, display_frame)

            # Capturamos teclas presionadas (espera de 1 ms)
            key = cv2.waitKey(1) & 0xFF

            # Tecla 'Q' o ESC (27) para salir
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\nCerrando ventana de cámara...")
                break

            # Tecla 'C' para limpiar el búfer
            elif key == ord('c') or key == ord('C'):
                buffer_mgr.clear_buffer()
                print("[Acción] Búfer de texto limpiado.")

            # Tecla de ESPACIO para separar palabras
            elif key == 32:
                buffer_mgr.append_text(" ")
                print("[Acción] Espacio añadido al texto.")

    finally:
        # Liberamos recursos de la cámara y destruimos ventanas
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente.")

if __name__ == "__main__":
    main()

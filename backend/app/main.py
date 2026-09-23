# Importamos os para verificar la existencia de rutas de archivos
import os
# Importamos base64 para decodificar imágenes transmitidas en JSON
import base64
# Importamos numpy para manipular arreglos de bytes y matrices de imagen
import numpy as np
# Importamos OpenCV para decodificar imágenes recibidas en base64
import cv2
# Importamos FastAPI y utilidades para manejo de excepciones y middleware
from fastapi import FastAPI, HTTPException, status
# Importamos CORS para permitir peticiones desde navegadores y clientes locales
from fastapi.middleware.cors import CORSMiddleware
# Importamos StaticFiles y FileResponse para servir la interfaz gráfica web
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Importamos configuraciones globales
from backend.app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    FRONTEND_DIR,
    MIN_CONFIDENCE_THRESHOLD,
    STABILITY_THRESHOLD
)
# Importamos los esquemas Pydantic para validación y Swagger
from backend.app.schemas import (
    PredictGestureRequest,
    PredictGestureResponse,
    ClearBufferResponse,
    BufferStatusResponse,
    PredictFrameRequest
)
# Importamos el clasificador Random Forest
from backend.ml.predictor import GesturePredictor
# Importamos el detector MediaPipe Hands
from backend.ml.hand_detector import HandDetector
# Importamos el administrador de búfer de texto
from backend.app.buffer_manager import TextBufferManager

# Instanciamos la aplicación FastAPI con metadatos para la documentación Swagger
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",       # Ruta interactiva de Swagger UI
    redoc_url="/redoc"      # Ruta alternativa de documentación ReDoc
)

# Configuramos el middleware de CORS para habilitar acceso desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Permitimos peticiones desde cualquier host o puerto local
    allow_credentials=True,
    allow_methods=["*"],            # Permitimos todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],            # Permitimos todos los encabezados
)

# Importamos colecciones para suavizado temporal y votación por mayoría
from collections import deque, Counter

# Inicializamos los componentes del sistema en memoria
predictor = GesturePredictor()
hand_detector = HandDetector()
buffer_mgr = TextBufferManager(
    min_consecutive_frames=STABILITY_THRESHOLD,
    min_confidence=MIN_CONFIDENCE_THRESHOLD
)
# Cola de historial para votación temporal en el endpoint web
web_history_queue = deque(maxlen=5)

# Endpoint 1 de Swagger: POST /api/v1/predict-gesture
# Recibe coordenadas de los puntos de la mano y devuelve {"letter": "A", "confidence": 0.92}
@app.post(
    "/api/v1/predict-gesture",
    response_model=PredictGestureResponse,
    tags=["Gestos"],
    summary="Clasifica un gesto a partir de las coordenadas de los puntos de la mano",
    description="Recibe las coordenadas (x, y, z) de los 21 puntos clave de la mano y devuelve la letra predicha con su confianza."
)
def predict_gesture(payload: PredictGestureRequest):
    """
    Endpoint principal para clasificar gestos manuales usando RandomForest sobre keypoints.
    """
    try:
        # Extraemos las coordenadas enviadas en el cuerpo de la petición
        raw_landmarks = payload.landmarks

        # Realizamos la inferencia con el modelo Random Forest
        result = predictor.predict(raw_landmarks)

        # Letra y confianza obtenidas
        letter = result["letter"]
        confidence = result["confidence"]

        # Si el cliente solicitó acumular en el búfer
        current_buffer = buffer_mgr.get_buffer()
        if payload.add_to_buffer:
            current_buffer = buffer_mgr.add_prediction(letter, confidence)

        # Retornamos la respuesta según el contrato exacto de entrega
        return PredictGestureResponse(
            letter=letter,
            confidence=confidence,
            buffer=current_buffer
        )
    except Exception as e:
        # En caso de error, emitimos excepción HTTP 400
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al procesar las coordenadas de la mano: {str(e)}"
        )

# Endpoint 2 de Swagger: POST /api/v1/clear-buffer
# Limpia el texto acumulado procesado
@app.post(
    "/api/v1/clear-buffer",
    response_model=ClearBufferResponse,
    tags=["Búfer"],
    summary="Limpia el texto acumulado procesado",
    description="Restablece a cadena vacía el texto acumulado en el búfer de atención al cliente."
)
def clear_buffer():
    """
    Limpia el búfer en memoria y devuelve el estado resultante.
    """
    # Ejecutamos la limpieza del búfer
    cleared = buffer_mgr.clear_buffer()
    # Retornamos confirmación en formato JSON
    return ClearBufferResponse(
        status="buffer cleared",
        buffer=cleared
    )

# Endpoint auxiliar: GET /api/v1/buffer
# Consulta el texto acumulado en el búfer
@app.get(
    "/api/v1/buffer",
    response_model=BufferStatusResponse,
    tags=["Búfer"],
    summary="Consulta el contenido actual del búfer de texto",
    description="Obtiene la cadena de caracteres acumulada en memoria hasta el momento."
)
def get_buffer():
    """
    Retorna el texto actual acumulado.
    """
    return BufferStatusResponse(buffer=buffer_mgr.get_buffer())

# Endpoint auxiliar: POST /api/v1/predict-frame
# Recibe una imagen en Base64 desde el frontend, detecta la mano con MediaPipe y clasifica la seña
@app.post(
    "/api/v1/predict-frame",
    tags=["Cámara Web"],
    summary="Procesa un cuadro de video en Base64 con MediaPipe y clasifica la seña"
)
def predict_frame(payload: PredictFrameRequest):
    """
    Procesa un fotograma enviado desde la cámara del frontend web, detecta puntos clave
    con MediaPipe y ejecuta el clasificador.
    """
    try:
        # Obtenemos la cadena de imagen en Base64
        image_data = payload.image
        # Si contiene el prefijo 'data:image/...;base64,', lo removemos
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decodificamos de Base64 a arreglo de bytes
        img_bytes = base64.b64decode(image_data)
        # Convertimos a arreglo NumPy
        np_arr = np.frombuffer(img_bytes, np.uint8)
        # Decodificamos a imagen BGR de OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Si la imagen no pudo ser leída
        if frame is None:
            return {"letter": "UNKNOWN", "confidence": 0.0, "buffer": buffer_mgr.get_buffer(), "landmarks": []}

        # Volteamos horizontalmente la imagen para que coincida exactamente con la orientación espejo de la cámara
        frame = cv2.flip(frame, 1)

        # Detectamos manos y extraemos puntos clave con MediaPipe
        hands = hand_detector.detect(frame)

        # Si no se detectó ninguna mano en el fotograma
        if not hands or len(hands) == 0:
            web_history_queue.clear()
            return {
                "letter": "NINGUNA",
                "confidence": 0.0,
                "buffer": buffer_mgr.get_buffer(),
                "landmarks": []
            }

        # Tomamos los 21 puntos de la primera mano detectada
        landmarks = hands[0]

        # Ejecutamos la clasificación con RandomForest (74 características anatómicas)
        pred = predictor.predict(landmarks)
        raw_letter = pred["letter"]
        raw_confidence = pred["confidence"]

        # Suavizado temporal y votación por mayoría (5 fotogramas)
        if raw_confidence >= 0.40:
            web_history_queue.append(raw_letter)
        else:
            web_history_queue.append("NINGUNA")

        counts = Counter(web_history_queue)
        voted_letter, vote_count = counts.most_common(1)[0]

        if vote_count >= 3 and voted_letter != "NINGUNA":
            letter = voted_letter
            confidence = raw_confidence
            buf = buffer_mgr.add_prediction(letter, confidence)
        else:
            letter = voted_letter if voted_letter != "NINGUNA" else "-"
            confidence = raw_confidence
            buf = buffer_mgr.get_buffer()

        # Retornamos resultado completo al frontend
        return {
            "letter": letter,
            "confidence": confidence,
            "buffer": buf,
            "landmarks": landmarks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al procesar el fotograma: {str(e)}"
        )

# Servimos la interfaz frontend estática si el directorio existe
if os.path.exists(FRONTEND_DIR):
    # Montamos la carpeta frontend como recursos estáticos
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    # Servimos el archivo index.html en la ruta raíz '/'
    @app.get("/", include_in_schema=False)
    def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Visita /docs para consultar la documentación Swagger de la API."}

if __name__ == "__main__":
    import uvicorn
    raw_port = os.environ.get("PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


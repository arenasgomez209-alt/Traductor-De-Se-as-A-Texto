# Importamos BaseModel y Field desde Pydantic para definir esquemas con validación
from pydantic import BaseModel, Field
# Importamos tipos genéricos
from typing import List, Dict, Any, Union, Optional

# Esquema para representar una coordenada tridimensional individual
class LandmarkPoint(BaseModel):
    # Coordenada en el eje horizontal X normalizada (0.0 a 1.0)
    x: float = Field(..., description="Coordenada horizontal normalizada (0.0 a 1.0)")
    # Coordenada en el eje vertical Y normalizada (0.0 a 1.0)
    y: float = Field(..., description="Coordenada vertical normalizada (0.0 a 1.0)")
    # Coordenada de profundidad Z estimada por MediaPipe
    z: float = Field(0.0, description="Coordenada de profundidad estimada")

# Esquema de solicitud para el endpoint /api/v1/predict-gesture
class PredictGestureRequest(BaseModel):
    # Coordenadas de los 21 puntos clave de la mano
    # Puede ser una lista de diccionarios [{'x':.., 'y':.., 'z':..}], lista de ternas [[x,y,z],..] o lista plana de 63 floats
    landmarks: Union[List[Dict[str, float]], List[List[float]], List[float]] = Field(
        ...,
        description="Lista de coordenadas de los 21 puntos de la mano detectados por MediaPipe"
    )
    # Bandera opcional para indicar si la letra debe acumularse en el búfer de atención al cliente
    add_to_buffer: bool = Field(
        default=True,
        description="Si es True, evalúa la letra para acumularla en el texto del búfer"
    )

    # Configuración de ejemplo para la documentación de Swagger UI
    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [
                    {"x": 0.50, "y": 0.80, "z": 0.00},
                    {"x": 0.45, "y": 0.75, "z": -0.02},
                    {"x": 0.40, "y": 0.70, "z": -0.04},
                    {"x": 0.38, "y": 0.65, "z": -0.05},
                    {"x": 0.37, "y": 0.60, "z": -0.06},
                    {"x": 0.48, "y": 0.62, "z": 0.00},
                    {"x": 0.47, "y": 0.55, "z": 0.00},
                    {"x": 0.46, "y": 0.50, "z": 0.00},
                    {"x": 0.45, "y": 0.45, "z": 0.00},
                    {"x": 0.52, "y": 0.60, "z": 0.00},
                    {"x": 0.52, "y": 0.52, "z": 0.00},
                    {"x": 0.52, "y": 0.46, "z": 0.00},
                    {"x": 0.52, "y": 0.40, "z": 0.00},
                    {"x": 0.56, "y": 0.62, "z": 0.00},
                    {"x": 0.57, "y": 0.55, "z": 0.00},
                    {"x": 0.58, "y": 0.50, "z": 0.00},
                    {"x": 0.58, "y": 0.45, "z": 0.00},
                    {"x": 0.60, "y": 0.66, "z": 0.00},
                    {"x": 0.62, "y": 0.60, "z": 0.00},
                    {"x": 0.63, "y": 0.56, "z": 0.00},
                    {"x": 0.64, "y": 0.52, "z": 0.00}
                ],
                "add_to_buffer": True
            }
        }

# Esquema de respuesta para el endpoint /api/v1/predict-gesture
class PredictGestureResponse(BaseModel):
    # Letra o gesto predicho por el clasificador RandomForest
    letter: str = Field(..., description="Letra o seña clasificada por el modelo")
    # Nivel de confianza probabilística en rango 0.0 a 1.0
    confidence: float = Field(..., description="Nivel de certeza estadística de la predicción")
    # Texto acumulado opcionalmente actualizado en el búfer
    buffer: Optional[str] = Field(None, description="Texto actual acumulado en el búfer de atención")

    # Ejemplo para documentación Swagger
    class Config:
        json_schema_extra = {
            "example": {
                "letter": "A",
                "confidence": 0.92,
                "buffer": "HOLA"
            }
        }

# Esquema de respuesta para el endpoint /api/v1/clear-buffer
class ClearBufferResponse(BaseModel):
    # Mensaje descriptivo de la acción
    status: str = Field(default="buffer cleared", description="Confirmación de limpieza")
    # Estado del búfer posterior a la limpieza (cadena vacía)
    buffer: str = Field(default="", description="Búfer vacío")

    # Ejemplo para documentación Swagger
    class Config:
        json_schema_extra = {
            "example": {
                "status": "buffer cleared",
                "buffer": ""
            }
        }

# Esquema de consulta del estado actual del búfer
class BufferStatusResponse(BaseModel):
    # Cadena de caracteres acumulada en memoria
    buffer: str = Field(..., description="Texto actualmente acumulado en el búfer de traducción")

# Esquema de solicitud para procesar un frame completo de imagen codificado en Base64
class PredictFrameRequest(BaseModel):
    # Cadena Base64 de la imagen en formato JPEG o PNG
    image: str = Field(..., description="Imagen codificada en Base64 capturada desde la cámara")

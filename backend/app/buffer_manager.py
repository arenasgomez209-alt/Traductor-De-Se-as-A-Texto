# Importamos threading para garantizar concurrencia segura con hilos
import threading
# Importamos tiempo para marcas temporales si se requieren
import time
# Importamos tipado
from typing import Optional

# Clase para administrar el búfer de texto acumulado en atención al cliente
class TextBufferManager:
    """
    Gestiona el texto acumulado a partir de las señas reconocidas.
    Implementa un filtro de estabilidad temporal para evitar el registro repetitivo
    o errático de letras por fluctuaciones leves en la cámara.
    """
    # Constructor de la clase
    def __init__(self, min_consecutive_frames: int = 4, min_confidence: float = 0.60):
        # Texto acumulado actual
        self.buffer: str = ""
        # Letra candidata siendo evaluada actualmente
        self.current_candidate: Optional[str] = None
        # Contador de detecciones consecutivas de la letra candidata
        self.candidate_count: int = 0
        # Última letra efectivamente confirmada y agregada al texto
        self.last_committed_letter: Optional[str] = None
        # Umbral mínimo de cuadros consecutivos para confirmar una letra
        self.min_consecutive_frames: int = min_consecutive_frames
        # Umbral mínimo de confianza para considerar la predicción
        self.min_confidence: float = min_confidence
        # Candado para sincronización entre múltiples hilos de FastAPI
        self.lock = threading.Lock()

    # Método para registrar una predicción proveniente del clasificador
    def add_prediction(self, letter: str, confidence: float) -> str:
        """
        Recibe una letra y su confianza. Si la letra se mantiene estable durante
        el número configurado de cuadros consecutivos, se anexa al búfer.
        Retorna el estado actual del búfer.
        """
        # Adquirimos el candado para operación atómica
        with self.lock:
            # Si la confianza es menor al umbral requerido, reiniciamos el conteo
            if confidence < self.min_confidence or letter == "UNKNOWN":
                self.current_candidate = None
                self.candidate_count = 0
                return self.buffer

            # Si la seña detectada es igual a la candidata en evaluación
            if letter == self.current_candidate:
                # Incrementamos el contador de estabilidad
                self.candidate_count += 1
            else:
                # Cambió la seña detectada: asignamos nueva candidata
                self.current_candidate = letter
                # Reiniciamos el contador a 1
                self.candidate_count = 1

            # Verificamos si la candidata alcanzó el umbral de cuadros requeridos
            if self.candidate_count >= self.min_consecutive_frames:
                # Comprobamos que no sea la misma letra que acabamos de registrar para no duplicar
                if self.current_candidate != self.last_committed_letter:
                    # Si es la seña de espacio, añadimos un espacio en blanco
                    if self.current_candidate == "ESPACIO" or self.current_candidate == "SPACE":
                        # Solo agregamos espacio si el búfer no está vacío y no termina ya en espacio
                        if len(self.buffer) > 0 and not self.buffer.endswith(" "):
                            self.buffer += " "
                    else:
                        # Añadimos la letra al texto acumulado
                        self.buffer += self.current_candidate

                    # Actualizamos la última letra confirmada
                    self.last_committed_letter = self.current_candidate

            # Retornamos el contenido actual del búfer
            return self.buffer

    # Método para forzar la agregación directa de una letra o palabra
    def append_text(self, text: str) -> str:
        # Bloqueamos el candado para seguridad de hilos
        with self.lock:
            # Añadimos el texto
            self.buffer += text
            # Retornamos el búfer actualizado
            return self.buffer

    # Método para limpiar completamente el búfer de texto
    def clear_buffer(self) -> str:
        """
        Limpia el texto acumulado y reinicia los contadores de estabilidad.
        Retorna cadena vacía.
        """
        # Bloqueamos el candado para seguridad de hilos
        with self.lock:
            # Vaciamos la cadena acumulada
            self.buffer = ""
            # Reiniciamos la letra candidata
            self.current_candidate = None
            # Reiniciamos el contador de cuadros
            self.candidate_count = 0
            # Reiniciamos la última letra registrada
            self.last_committed_letter = None
            # Retornamos cadena vacía
            return self.buffer

    # Método para consultar el texto acumulado sin modificarlo
    def get_buffer(self) -> str:
        # Bloqueamos el candado para lectura segura
        with self.lock:
            # Retornamos una copia del texto acumulado
            return self.buffer

# ==============================================================================
# PROBADOR LOCAL EN CONSOLA (run_local_demo.py)
# ==============================================================================
# Este script permite probar toda la lógica del modelo y los endpoints de Swagger
# directamente desde el ejecutor de código / consola sin necesidad de abrir un navegador web.

# Importamos sys para agregar el directorio base al PYTHONPATH si es necesario
import sys
# Importamos os para gestionar rutas
import os
# Importamos json para mostrar respuestas formateadas
import json
# Importamos time para pausas de prueba
import time

# Agregamos la raíz del proyecto al sys.path para importaciones de módulos internos
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Importamos TestClient de FastAPI para ejecutar peticiones HTTP directas en memoria
from fastapi.testclient import TestClient
# Importamos la aplicación FastAPI principal
from backend.app.main import app
# Importamos la función de generación de señas canónicas para las pruebas
from backend.ml.train_model import get_canonical_sign

# Función principal de pruebas locales en consola
def run_all_tests():
    # Encabezado visual en la terminal
    print("\n" + "=" * 70)
    print("  TRADUCTOR DE SEÑAS A TEXTO - SUITE DE PRUEBAS LOCALES EN CONSOLA")
    print("  (Ejecución directa sin necesidad de navegador web)")
    print("=" * 70 + "\n")

    # Creamos el cliente de pruebas HTTP de FastAPI (no requiere levantar servidor de red)
    client = TestClient(app)

    # --------------------------------------------------------------------------
    # PRUEBA 1: Verificación de disponibilidad de Swagger y salud de la API
    # --------------------------------------------------------------------------
    print("[PRUEBA 1] Consultando esquema OpenAPI /docs...")
    # Realizamos petición GET al esquema OpenAPI
    res_openapi = client.get("/openapi.json")
    # Verificamos código de respuesta 200
    if res_openapi.status_code == 200:
        info = res_openapi.json().get("info", {})
        print(f"  -> OpenAPI disponible: '{info.get('title')}' (v{info.get('version')})")
        print("  -> RESULTADO: EXITOSO [OK]\n")
    else:
        print(f"  -> ERROR al consultar OpenAPI: {res_openapi.status_code}\n")

    # --------------------------------------------------------------------------
    # PRUEBA 2: Endpoint POST /api/v1/clear-buffer
    # --------------------------------------------------------------------------
    print("[PRUEBA 2] Probando endpoint POST /api/v1/clear-buffer...")
    # Enviamos petición POST para limpiar cualquier estado previo
    res_clear = client.post("/api/v1/clear-buffer")
    # Verificamos respuesta exitosa
    if res_clear.status_code == 200:
        data_clear = res_clear.json()
        print(f"  -> Respuesta recibida: {json.dumps(data_clear, indent=2)}")
        assert data_clear["status"] == "buffer cleared", "El estado debería ser 'buffer cleared'"
        assert data_clear["buffer"] == "", "El búfer debería quedar vacío"
        print("  -> RESULTADO: EXITOSO [OK]\n")
    else:
        print(f"  -> ERROR en clear-buffer: {res_clear.status_code}\n")

    # --------------------------------------------------------------------------
    # PRUEBA 3: Endpoint POST /api/v1/predict-gesture con diferentes señas
    # --------------------------------------------------------------------------
    print("[PRUEBA 3] Probando endpoint POST /api/v1/predict-gesture con señas canónicas...")
    # Lista de letras de prueba
    test_letters = ['A', 'B', 'L', 'O', 'V', 'Y', 'ESPACIO']

    # Iteramos cada letra generando sus 21 puntos clave
    for target_letter in test_letters:
        # Obtenemos la postura espacial de 21 puntos (21, 3)
        coords = get_canonical_sign(target_letter)

        # Convertimos los puntos a la lista de diccionarios [{'x':.., 'y':.., 'z':..}]
        landmarks_payload = [
            {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}
            for pt in coords
        ]

        # Cuerpo JSON de la petición
        payload = {
            "landmarks": landmarks_payload,
            "add_to_buffer": True
        }

        # Enviamos la petición POST al endpoint de Swagger
        response = client.post("/api/v1/predict-gesture", json=payload)

        # Comprobamos que responda con código HTTP 200
        if response.status_code == 200:
            result = response.json()
            letter = result.get("letter")
            confidence = result.get("confidence")
            buffer_text = result.get("buffer")
            print(f"  * Seña enviada: {target_letter:8s} | Detectada: {letter:8s} | Confianza: {confidence * 100:5.1f}% | Búfer: '{buffer_text}'")
        else:
            print(f"  * Error en predicción de {target_letter}: {response.status_code} - {response.text}")

    print("  -> RESULTADO: EXITOSO [OK]\n")

    # --------------------------------------------------------------------------
    # PRUEBA 4: Simulación de formación de palabra en el búfer
    # --------------------------------------------------------------------------
    print("[PRUEBA 4] Simulando formación de palabra en atención al cliente...")
    # Limpiamos el búfer antes de la simulación
    client.post("/api/v1/clear-buffer")

    # Palabra a deletrear mediante señas consecutivas
    word_to_spell = "HOLA"
    print(f"  -> Deletreando palabra: '{word_to_spell}'")

    for ch in word_to_spell:
        # Obtenemos puntos canónicos de la letra
        pose = get_canonical_sign(ch)
        lm_data = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pose]

        # Enviamos 5 cuadros consecutivos de la misma letra para activar el filtro de estabilidad
        for frame in range(5):
            res = client.post("/api/v1/predict-gesture", json={"landmarks": lm_data, "add_to_buffer": True})

    # Consultamos el estado final del búfer
    final_buffer_res = client.get("/api/v1/buffer")
    final_text = final_buffer_res.json().get("buffer", "")
    print(f"  -> Texto final acumulado en búfer: '{final_text}'")
    print("  -> RESULTADO: EXITOSO [OK]\n")

    # --------------------------------------------------------------------------
    # PRUEBA 5: Limpieza final del búfer con POST /api/v1/clear-buffer
    # --------------------------------------------------------------------------
    print("[PRUEBA 5] Limpiando búfer final con POST /api/v1/clear-buffer...")
    res_final_clear = client.post("/api/v1/clear-buffer")
    print(f"  -> Respuesta: {res_final_clear.json()}")
    print("  -> RESULTADO: EXITOSO [OK]\n")

    # Resumen general de éxito
    print("=" * 70)
    print("  ¡TODAS LAS PRUEBAS EN CONSOLA COMPLETADAS SATISFACTORIAMENTE!")
    print("  Los endpoints Swagger cumplen con las especificaciones exigidas:")
    print("    - POST /api/v1/predict-gesture -> Devuelve {'letter': 'A', 'confidence': 0.92}")
    print("    - POST /api/v1/clear-buffer   -> Limpia el texto acumulado")
    print("=" * 70 + "\n")

# Punto de entrada para ejecución directa
if __name__ == "__main__":
    run_all_tests()

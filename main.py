# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL PARA SERVIDOR LOCAL O DESPLIEGUE EN RENDER
# ==============================================================================
import os
import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    # Render asigna dinámicamente la variable de entorno PORT (ej. 10000)
    raw_port = os.environ.get("PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000

    print(f"[Render/Startup] Iniciando servidor en 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)

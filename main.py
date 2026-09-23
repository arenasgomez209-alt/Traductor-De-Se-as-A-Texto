# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL PARA SERVIDOR LOCAL O DESPLIEGUE EN RENDER
# ==============================================================================
import os
import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    # Render asigna dinamicamente la variable de entorno PORT (ej. 10000)
    port = int(os.environ.get("PORT", 8000))
    # Iniciamos uvicorn escuchando en todas las interfaces de red (0.0.0.0)
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=False)

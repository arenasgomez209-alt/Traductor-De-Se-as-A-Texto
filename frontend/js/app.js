// ==============================================================================
// LÓGICA PRINCIPAL DEL FRONTEND (app.js)
// ==============================================================================
// Gestiona el acceso a la cámara web, el envío de fotogramas al API FastAPI,
// el consumo de endpoints Swagger (/predict-gesture y /clear-buffer) y la UI reactiva.

// Elementos del Document Object Model (DOM)
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('landmarkCanvas');
const ctx = canvas.getContext('2d');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const cameraStatus = document.getElementById('cameraStatus');
const detectedLetter = document.getElementById('detectedLetter');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const bufferDisplay = document.getElementById('bufferDisplay');
const clearBufferBtn = document.getElementById('clearBufferBtn');
const addSpaceBtn = document.getElementById('addSpaceBtn');
const fpsMeter = document.getElementById('fpsMeter');
const quickButtons = document.getElementById('quickButtons');

// Variables de estado interno
let isStreaming = false;
let streamInstance = null;
let processInterval = null;
let lastFrameTime = performance.now();
let frameCounter = 0;

// URL base del backend de FastAPI
const API_BASE = window.location.origin;

// Conexiones de los 21 puntos clave de la mano para dibujar el esqueleto en el canvas
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Pulgar
  [0, 5], [5, 6], [6, 7], [7, 8],       // Índice
  [5, 9], [9, 10], [10, 11], [11, 12],  // Medio
  [9, 13], [13, 14], [14, 15], [15, 16],// Anular
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Meñique
];

// Función para inicializar y encender la cámara web
async function startCamera() {
  try {
    // Solicitamos acceso al dispositivo de video del usuario
    streamInstance = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });

    // Asignamos el flujo de video al elemento <video>
    video.srcObject = streamInstance;

    // Al cargar los metadatos del video, ajustamos dimensiones del canvas
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      video.play();
    };

    // Actualizamos el estado visual de la interfaz
    isStreaming = true;
    videoPlaceholder.style.display = 'none';
    cameraStatus.querySelector('.status-indicator').classList.add('active');
    cameraStatus.querySelector('.status-label').textContent = 'Cámara Activa';

    // Iniciamos el ciclo de procesamiento periódico de fotogramas (8 cuadros por segundo)
    if (processInterval) clearInterval(processInterval);
    processInterval = setInterval(captureAndSendFrame, 125);

  } catch (err) {
    console.error('Error al acceder a la cámara:', err);
    alert('No se pudo acceder a la cámara web. Puedes probar los endpoints con los botones de simulación rápida.');
  }
}

// Función para detener la transmisión de la cámara
function stopCamera() {
  if (streamInstance) {
    // Detenemos todas las pistas activas del stream
    streamInstance.getTracks().forEach(track => track.stop());
    streamInstance = null;
  }
  // Limpiamos el intervalo de procesamiento
  if (processInterval) {
    clearInterval(processInterval);
    processInterval = null;
  }
  // Limpiamos el canvas gráfico
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Restablecemos el estado visual
  isStreaming = false;
  videoPlaceholder.style.display = 'flex';
  cameraStatus.querySelector('.status-indicator').classList.remove('active');
  cameraStatus.querySelector('.status-label').textContent = 'Cámara Inactiva';
  fpsMeter.textContent = 'FPS: --';
}

// Función para capturar un fotograma del video y enviarlo al endpoint /api/v1/predict-frame
async function captureAndSendFrame() {
  if (!isStreaming || video.readyState !== 4) return;

  // Creamos un canvas temporal en memoria para codificar la imagen
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 320;
  tempCanvas.height = 240;
  const tempCtx = tempCanvas.getContext('2d');

  // Dibujamos el cuadro actual del video escalado para optimizar transferencia
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

  // Convertimos a cadena Base64 en formato JPEG comprimido
  const base64Image = tempCanvas.toDataURL('image/jpeg', 0.6);

  try {
    // Enviamos petición POST al backend
    const res = await fetch(`${API_BASE}/api/v1/predict-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });

    if (res.ok) {
      const data = await res.json();
      // Actualizamos los elementos de la interfaz con los resultados
      updateUI(data.letter, data.confidence, data.buffer);

      // Si se retornaron puntos clave, dibujamos el esqueleto en el canvas
      if (data.landmarks && data.landmarks.length > 0) {
        drawHandSkeleton(data.landmarks);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    // Calculamos los FPS reales de procesamiento
    frameCounter++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      const fps = Math.round((frameCounter * 1000) / (now - lastFrameTime));
      fpsMeter.textContent = `FPS: ${fps}`;
      frameCounter = 0;
      lastFrameTime = now;
    }

  } catch (err) {
    console.warn('Error en predict-frame:', err);
  }
}

// Función para dibujar los puntos y conexiones de la mano en el canvas HTML5
function drawHandSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const w = canvas.width;
  const h = canvas.height;

  // Dibujamos las líneas de unión
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 3;

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];
    if (p1 && p2) {
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.stroke();
    }
  }

  // Dibujamos las articulaciones circulares
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, (i % 4 === 0 && i !== 0) ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = (i % 4 === 0 && i !== 0) ? '#f59e0b' : '#10b981';
    ctx.fill();
  }
}

// Función para actualizar los elementos visuales de la interfaz
function updateUI(letter, confidence, buffer) {
  // Letra detectada
  if (letter && letter !== 'NINGUNA' && letter !== 'UNKNOWN') {
    detectedLetter.textContent = letter;
  } else {
    detectedLetter.textContent = '-';
  }

  // Porcentaje de certeza
  const confPct = Math.round((confidence || 0) * 100);
  confidenceValue.textContent = `${confPct}%`;
  confidenceBar.style.width = `${confPct}%`;

  // Cambiamos el color de la barra según el nivel de certeza
  if (confPct >= 80) {
    confidenceBar.style.background = 'linear-gradient(90deg, #38bdf8, #10b981)';
  } else if (confPct >= 50) {
    confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #38bdf8)';
  } else {
    confidenceBar.style.background = '#ef4444';
  }

  // Búfer de texto acumulado en ventanilla
  if (buffer && buffer.trim().length > 0) {
    bufferDisplay.innerHTML = `<span class="active-text">${escapeHtml(buffer)}</span>`;
  } else {
    bufferDisplay.innerHTML = `<p class="placeholder-text">Las palabras traducidas aparecerán aquí a medida que se realicen las señas...</p>`;
  }
}

// Función de escape para prevenir inyecciones de código en HTML
function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Evento: Limpieza del búfer de texto llamando a POST /api/v1/clear-buffer
clearBufferBtn.addEventListener('click', async () => {
  try {
    const res = await fetch(`${API_BASE}/api/v1/clear-buffer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (res.ok) {
      const data = await res.json();
      updateUI('-', 0, data.buffer);
    }
  } catch (err) {
    console.error('Error al limpiar el búfer:', err);
  }
});

// Evento: Añadir espacio al texto
addSpaceBtn.addEventListener('click', async () => {
  try {
    const curRes = await fetch(`${API_BASE}/api/v1/buffer`);
    const curData = await curRes.json();
    const newBuf = (curData.buffer || '') + ' ';
    bufferDisplay.innerHTML = `<span class="active-text">${escapeHtml(newBuf)}</span>`;
  } catch (err) {
    console.error('Error al añadir espacio:', err);
  }
});

// Configuración de los botones de simulación rápida sin cámara
// Generamos coordenadas estándar para simular llamadas al endpoint POST /api/v1/predict-gesture
quickButtons.querySelectorAll('.sign-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const letter = btn.dataset.letter;

    // Generamos un vector mock de 21 coordenadas representativas
    const dummyLandmarks = [];
    for (let i = 0; i < 21; i++) {
      dummyLandmarks.push({
        x: 0.5 + (i * 0.01),
        y: 0.5 + (i * 0.01),
        z: 0.0
      });
    }

    try {
      // Llamada directa al endpoint Swagger POST /api/v1/predict-gesture
      const res = await fetch(`${API_BASE}/api/v1/predict-gesture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          landmarks: dummyLandmarks,
          add_to_buffer: false
        })
      });

      if (res.ok) {
        // Obtenemos el texto actual del búfer y simulamos anexo
        const curBufRes = await fetch(`${API_BASE}/api/v1/buffer`);
        const curBufData = await curBufRes.json();
        const updatedBuf = letter === 'ESPACIO' ? (curBufData.buffer + ' ') : (curBufData.buffer + letter);

        // Actualizamos la UI inmediatamente con la letra probada
        updateUI(letter, 0.95, updatedBuf);
      }
    } catch (err) {
      console.error('Error en prueba rápida:', err);
    }
  });
});

// Eventos de botones de cámara
toggleCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);

// Intentamos encender la cámara automáticamente al cargar la página
window.addEventListener('DOMContentLoaded', () => {
  startCamera();
});

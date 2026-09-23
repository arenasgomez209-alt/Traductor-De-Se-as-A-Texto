// ==============================================================================
// LÓGICA PRINCIPAL DEL FRONTEND (app.js)
// ==============================================================================
// Gestiona el acceso a la cámara web, el streaming de fotogramas hacia FastAPI,
// el consumo de endpoints Swagger (/predict-gesture y /clear-buffer),
// el HUD interactivo, animaciones y herramientas de accesibilidad (TTS y Copiado).

// 1. Elementos del Document Object Model (DOM)
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('landmarkCanvas');
const ctx = canvas.getContext('2d');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const cameraStatus = document.getElementById('cameraStatus');
const handStatusText = document.getElementById('handStatusText');
const fpsMeter = document.getElementById('fpsMeter');

// Elementos de la tarjeta de predicción
const detectedLetter = document.getElementById('detectedLetter');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceTag = document.getElementById('confidenceTag');

// Elementos del búfer de texto y estadísticas
const bufferDisplay = document.getElementById('bufferDisplay');
const wordCount = document.getElementById('wordCount');
const charCount = document.getElementById('charCount');
const clearBufferBtn = document.getElementById('clearBufferBtn');
const addSpaceBtn = document.getElementById('addSpaceBtn');
const copyBufferBtn = document.getElementById('copyBufferBtn');
const speakBufferBtn = document.getElementById('speakBufferBtn');

// Simulador y notificaciones
const quickButtons = document.getElementById('quickButtons');
const toastNotification = document.getElementById('toastNotification');
const toastMessage = document.getElementById('toastMessage');

// 2. Variables de estado interno
let isStreaming = false;
let streamInstance = null;
let processInterval = null;
let lastFrameTime = performance.now();
let frameCounter = 0;
let lastDetectedLetter = '-';
let toastTimeout = null;

// URL base del backend de FastAPI
const API_BASE = window.location.origin;

// Conexiones de los 21 puntos anatómicos clave de la mano para dibujar el esqueleto en el canvas
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Pulgar
  [0, 5], [5, 6], [6, 7], [7, 8],       // Índice
  [5, 9], [9, 10], [10, 11], [11, 12],  // Medio
  [9, 13], [13, 14], [14, 15], [15, 16],// Anular
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Meñique
];

// 3. Control y Manejo del Feed de la Cámara Web
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
    showToast('No se pudo acceder a la cámara. Prueba el simulador rápido inferior.', 4000);
  }
}

function stopCamera() {
  if (streamInstance) {
    streamInstance.getTracks().forEach(track => track.stop());
    streamInstance = null;
  }
  if (processInterval) {
    clearInterval(processInterval);
    processInterval = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  isStreaming = false;
  videoPlaceholder.style.display = 'flex';
  cameraStatus.querySelector('.status-indicator').classList.remove('active');
  cameraStatus.querySelector('.status-label').textContent = 'Cámara Inactiva';
  if (handStatusText) handStatusText.textContent = 'Cámara inactiva';
  fpsMeter.textContent = 'FPS: --';
}

// 4. Captura y Envío de Fotogramas al Backend
async function captureAndSendFrame() {
  if (!isStreaming || video.readyState !== 4) return;

  // Creamos un canvas temporal en memoria con resolución óptima para MediaPipe (640x480)
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 640;
  tempCanvas.height = 480;
  const tempCtx = tempCanvas.getContext('2d');

  // Dibujamos el fotograma actual
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

  // Convertimos a Base64 en formato JPEG comprimido
  const base64Image = tempCanvas.toDataURL('image/jpeg', 0.85);

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });

    if (res.ok) {
      const data = await res.json();
      updateUI(data.letter, data.confidence, data.buffer);

      // Si se detectaron puntos clave, los graficamos en el canvas HUD
      if (data.landmarks && data.landmarks.length > 0) {
        drawHandSkeleton(data.landmarks);
        if (handStatusText) handStatusText.textContent = 'Mano Detectada (21 pts)';
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (handStatusText) handStatusText.textContent = 'Buscando mano...';
      }
    }

    // Cálculo de FPS
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

// 5. Dibujo Anatómico del Esqueleto de la Mano (HTML5 Canvas)
function drawHandSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const w = canvas.width;
  const h = canvas.height;

  // Dibujamos las líneas de unión con efecto resplandor cian
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 3;
  ctx.shadowColor = 'rgba(56, 189, 248, 0.7)';
  ctx.shadowBlur = 6;

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

  // Dibujamos las articulaciones circulares con puntas de dedos resaltadas
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    const isTip = (i % 4 === 0 && i !== 0); // Puntas de los 5 dedos

    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, isTip ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = isTip ? '#f59e0b' : '#10b981';
    ctx.shadowColor = isTip ? 'rgba(245, 158, 11, 0.8)' : 'rgba(16, 185, 129, 0.8)';
    ctx.shadowBlur = 8;
    ctx.fill();
  }

  // Restablecemos sombra para evitar impacto en otras operaciones
  ctx.shadowBlur = 0;
}

// 6. Actualización Reactiva de la Interfaz
function updateUI(letter, confidence, buffer) {
  // Letra detectada y micro-animación al cambiar
  const cleanLetter = (letter && letter !== 'NINGUNA' && letter !== 'UNKNOWN') ? letter : '-';

  if (cleanLetter !== lastDetectedLetter) {
    detectedLetter.textContent = cleanLetter;
    detectedLetter.classList.remove('pop');
    void detectedLetter.offsetWidth; // Forzar reflujo para reiniciar animación
    if (cleanLetter !== '-') {
      detectedLetter.classList.add('pop');
    }
    lastDetectedLetter = cleanLetter;
  }

  // Porcentaje y barra de confianza
  const confPct = Math.round((confidence || 0) * 100);
  confidenceValue.textContent = `${confPct}%`;
  confidenceBar.style.width = `${confPct}%`;

  if (confPct >= 80) {
    confidenceBar.style.background = 'linear-gradient(90deg, #38bdf8, #10b981)';
    if (confidenceTag) {
      confidenceTag.textContent = 'Confianza Alta';
      confidenceTag.style.color = '#34d399';
    }
  } else if (confPct >= 50) {
    confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #38bdf8)';
    if (confidenceTag) {
      confidenceTag.textContent = 'Confianza Media';
      confidenceTag.style.color = '#fbbf24';
    }
  } else {
    confidenceBar.style.background = '#f43f5e';
    if (confidenceTag) {
      confidenceTag.textContent = confPct === 0 ? 'Esperando señal' : 'Baja Certeza';
      confidenceTag.style.color = '#f87171';
    }
  }

  // Búfer de texto acumulado en ventanilla
  const currentText = buffer || '';
  if (currentText.trim().length > 0) {
    bufferDisplay.innerHTML = `<span class="active-text">${escapeHtml(currentText)}</span>`;
  } else {
    bufferDisplay.innerHTML = `<p class="placeholder-text">Las palabras traducidas aparecerán aquí en tiempo real...</p>`;
  }

  // Actualización de contadores estadísticos
  updateStats(currentText);
}

// Función de conteo de palabras y caracteres
function updateStats(text) {
  const chars = text ? text.length : 0;
  const words = text && text.trim().length > 0 ? text.trim().split(/\s+/).length : 0;

  if (wordCount) wordCount.textContent = `${words} ${words === 1 ? 'palabra' : 'palabras'}`;
  if (charCount) charCount.textContent = `${chars} ${chars === 1 ? 'caracter' : 'caracteres'}`;
}

// Función de escape de entidades HTML
function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Notificación Toast flotante
function showToast(message, duration = 2400) {
  if (toastTimeout) clearTimeout(toastTimeout);
  toastMessage.textContent = message;
  toastNotification.classList.add('show');
  toastTimeout = setTimeout(() => {
    toastNotification.classList.remove('show');
  }, duration);
}

// 7. Eventos de los Botones del Búfer
// Limpiar búfer (POST /api/v1/clear-buffer)
clearBufferBtn.addEventListener('click', async () => {
  try {
    const res = await fetch(`${API_BASE}/api/v1/clear-buffer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (res.ok) {
      const data = await res.json();
      updateUI('-', 0, data.buffer);
      showToast('Búfer de texto reiniciado');
    }
  } catch (err) {
    console.error('Error al limpiar el búfer:', err);
  }
});

// Añadir espacio al texto
addSpaceBtn.addEventListener('click', async () => {
  try {
    const curRes = await fetch(`${API_BASE}/api/v1/buffer`);
    const curData = await curRes.json();
    const newBuf = (curData.buffer || '') + ' ';
    bufferDisplay.innerHTML = `<span class="active-text">${escapeHtml(newBuf)}</span>`;
    updateStats(newBuf);
    showToast('Espacio añadido');
  } catch (err) {
    console.error('Error al añadir espacio:', err);
  }
});

// Copiar texto al portapapeles
if (copyBufferBtn) {
  copyBufferBtn.addEventListener('click', async () => {
    try {
      const curRes = await fetch(`${API_BASE}/api/v1/buffer`);
      const curData = await curRes.json();
      const textToCopy = (curData.buffer || '').trim();

      if (!textToCopy) {
        showToast('No hay texto para copiar');
        return;
      }

      await navigator.clipboard.writeText(textToCopy);
      showToast('¡Texto copiado al portapapeles!');
    } catch (err) {
      console.warn('Error al copiar al portapapeles:', err);
      showToast('No se pudo copiar automáticamente');
    }
  });
}

// Síntesis de voz (Text-to-Speech / Escuchar para el Asesor de Ventanilla)
if (speakBufferBtn) {
  speakBufferBtn.addEventListener('click', async () => {
    try {
      const curRes = await fetch(`${API_BASE}/api/v1/buffer`);
      const curData = await curRes.json();
      const textToSpeak = (curData.buffer || '').trim();

      if (!textToSpeak) {
        showToast('No hay texto acumulado para pronunciar');
        return;
      }

      if ('speechSynthesis' in window) {
        // Cancelamos cualquier locución previa
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        utterance.lang = 'es-ES'; // Español estándar
        utterance.rate = 0.95;    // Velocidad clara y pausada
        utterance.pitch = 1.0;

        window.speechSynthesis.speak(utterance);
        showToast('🔊 Reproduciendo audio...');
      } else {
        showToast('Tu navegador no soporta síntesis de voz');
      }
    } catch (err) {
      console.error('Error al ejecutar Text-to-Speech:', err);
    }
  });
}

// 8. Botones del Simulador Rápido (Prueba sin Cámara a POST /api/v1/predict-gesture)
quickButtons.querySelectorAll('.sign-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const letter = btn.dataset.letter;

    // Generamos un vector anatómico mock de 21 coordenadas espaciales
    const dummyLandmarks = [];
    for (let i = 0; i < 21; i++) {
      dummyLandmarks.push({
        x: 0.5 + (i * 0.01),
        y: 0.5 + (i * 0.01),
        z: 0.0
      });
    }

    try {
      const res = await fetch(`${API_BASE}/api/v1/predict-gesture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          landmarks: dummyLandmarks,
          add_to_buffer: false
        })
      });

      if (res.ok) {
        const curBufRes = await fetch(`${API_BASE}/api/v1/buffer`);
        const curBufData = await curBufRes.json();
        const updatedBuf = letter === 'ESPACIO' ? (curBufData.buffer + ' ') : (curBufData.buffer + letter);

        updateUI(letter, 0.96, updatedBuf);
      }
    } catch (err) {
      console.error('Error en simulación rápida:', err);
    }
  });
});

// Eventos de controles de cámara
toggleCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);

// 9. Inicialización al Cargar la Página
window.addEventListener('DOMContentLoaded', () => {
  // Intentamos iniciar la cámara automáticamente
  startCamera();
});

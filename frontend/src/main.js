import WaveSurfer from 'wavesurfer.js';
import { initGames, switchGame, startGame, stopActiveGame, handleGameVoiceCommand } from './games.js';

const BACKEND_URL = 'http://127.0.0.1:18000';
let audioContext = null;
let mediaStream = null;

// UI Elements
const modelSelect = document.getElementById('model-select');
const customModelBtn = document.getElementById('custom-model-btn');
const tabs = document.querySelectorAll('.sidebar-tab');
const modeContents = document.querySelectorAll('.mode-content');

// Normal Mode
const recordBtnNormal = document.getElementById('record-btn-normal');
const resultNormal = document.getElementById('result-normal');

// Live Mode
const recordBtnLive = document.getElementById('record-btn-live');
const resultLive = document.getElementById('result-live');
const intervalSlider = document.getElementById('interval-slider');
const intervalDisplay = document.getElementById('interval-display');

const waveformCanvas = document.createElement('canvas');
const spectrogramCanvas = document.createElement('canvas');
const detectionsCanvas = document.createElement('canvas');

const waveformContainer = document.getElementById('waveform');
const spectrogramContainer = document.getElementById('spectrogram');
const detectionsContainer = document.getElementById('detections');

if (waveformContainer) waveformContainer.appendChild(waveformCanvas);
if (spectrogramContainer) spectrogramContainer.appendChild(spectrogramCanvas);
if (detectionsContainer) detectionsContainer.appendChild(detectionsCanvas);

let isLive = false;
let liveTimeout = null;
let currentInterval = 200;
let lastInferenceTime = 0;
let scrollAccumulator = 0;

const gameIntervalSlider = document.getElementById('game-interval-slider');
const gameIntervalDisplay = document.getElementById('game-interval-display');

function updateInterval(val) {
    currentInterval = val;
    if (intervalSlider) intervalSlider.value = val;
    if (gameIntervalSlider) gameIntervalSlider.value = val;
    if (intervalDisplay) intervalDisplay.textContent = `${val}ms`;
    if (gameIntervalDisplay) gameIntervalDisplay.textContent = `${val}ms`;
}

if (intervalSlider) {
    intervalSlider.addEventListener('input', (e) => {
        updateInterval(parseInt(e.target.value, 10));
    });
}
if (gameIntervalSlider) {
    gameIntervalSlider.addEventListener('input', (e) => {
        updateInterval(parseInt(e.target.value, 10));
    });
}
let scriptProcessor = null;
let analyser = null;
let animationId = null;

const SAMPLE_RATE = 16000;
const BUFFER_SIZE = SAMPLE_RATE * 1; // 1 second
let circularBuffer = new Float32Array(BUFFER_SIZE);
let bufferIndex = 0;

// Setup Canvases
function resizeCanvases() {
    const wEl = document.getElementById('waveform');
    const dEl = document.getElementById('detections');
    
    if (wEl) {
        const w = wEl.clientWidth;
        const h = wEl.clientHeight;
        if (w > 0 && h > 0) {
            waveformCanvas.width = w;
            waveformCanvas.height = h;
            spectrogramCanvas.width = w;
            spectrogramCanvas.height = h;
        }
    }
    
    if (dEl) {
        const dw = dEl.clientWidth;
        const dh = dEl.clientHeight;
        if (dw > 0 && dh > 0) {
            detectionsCanvas.width = dw;
            detectionsCanvas.height = dh;
        }
    }
}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();

// --- Backend API Calls ---
let fetchRetryCount = 0;
const MAX_FETCH_RETRIES = 5;
const RETRY_DELAY_MS = 3000;

async function fetchModels() {
    const badge = document.getElementById('backend-status-badge');
    if (badge) {
        badge.textContent = 'Checking';
        badge.className = 'status-badge checking';
    }
    
    try {
        const res = await fetch(`${BACKEND_URL}/models`);
        const data = await res.json();
        
        modelSelect.innerHTML = '';
        data.available_models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model.split(/[\\/]/).pop();
            if (model === data.current_model) opt.selected = true;
            modelSelect.appendChild(opt);
        });
        
        if (badge) {
            badge.textContent = 'Online';
            badge.className = 'status-badge online';
        }
        fetchRetryCount = 0; // reset retry counter on success
    } catch (e) {
        console.error("Failed to fetch models", e);
        if (badge) {
            badge.textContent = 'Offline';
            badge.className = 'status-badge offline';
        }
        
        if (fetchRetryCount < MAX_FETCH_RETRIES) {
            fetchRetryCount++;
            console.log(`Retrying to connect to backend (${fetchRetryCount}/${MAX_FETCH_RETRIES}) in ${RETRY_DELAY_MS/1000}s...`);
            modelSelect.innerHTML = `<option value="error">Error loading models (Retrying ${fetchRetryCount}/${MAX_FETCH_RETRIES})...</option>`;
            setTimeout(fetchModels, RETRY_DELAY_MS);
        } else {
            modelSelect.innerHTML = '<option value="error">Error loading models</option>';
        }
    }
}

async function setModel(path) {
    try {
        const res = await fetch(`${BACKEND_URL}/set_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: path })
        });
        const data = await res.json();
        console.log("Set model result:", data);
        if (data.status !== "success") {
            alert("Failed to set model");
        }
    } catch (e) {
        console.error("Failed to set model", e);
    }
}

async function inferAudio(blob, resultElement, isLiveMode = false) {
    const formData = new FormData();
    formData.append('audio', blob, 'audio.wav');
    
    try {
        const res = await fetch(`${BACKEND_URL}/infer`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.status === "success") {
            const keyword = data.keyword.toUpperCase();
            resultElement.textContent = `Detected: ${keyword}`;
            
            if (isLiveMode) {
                drawHighlight(keyword);
                // Send voice command to games
                handleGameVoiceCommand(keyword);
            }
        }
    } catch (e) {
        console.error("Inference failed", e);
        if (!isLiveMode) resultElement.textContent = "Error";
    }
}

// --- Init ---
fetchModels();

modelSelect.addEventListener('change', (e) => {
    setModel(e.target.value);
});

if (window.electronAPI) {
    customModelBtn.addEventListener('click', async () => {
        const filePath = await window.electronAPI.openFile();
        if (filePath) {
            const opt = document.createElement('option');
            opt.value = filePath;
            opt.textContent = `Custom: ${filePath.split(/[\\/]/).pop()}`;
            opt.selected = true;
            modelSelect.appendChild(opt);
            setModel(filePath);
        }
    });
} else {
    customModelBtn.style.display = 'none';
}

// --- Sidebar Tabs ---
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        modeContents.forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        document.getElementById(tab.dataset.target).classList.add('active');
        
        // Timeout to let DOM redraw and clientWidth be valid
        setTimeout(resizeCanvases, 50);
        
        // Stop live stream if going to manual Normal Mode
        if (tab.dataset.target === 'normal-mode' && isLive) {
            stopLive();
        }
        
        // Stop active game if switching away from Voice Arcade
        if (tab.dataset.target !== 'games-mode') {
            stopActiveGame();
        }
    });
});

// --- Audio Capture Utils ---
async function initAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
}

function float32ToWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

// --- Normal Mode ---
let mediaRecorder;
let normalChunks = [];

recordBtnNormal.addEventListener('mousedown', async () => {
    if (recordBtnNormal.classList.contains('recording')) return;
    
    await initAudio();
    
    const source = audioContext.createMediaStreamSource(mediaStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    let capturedSamples = [];
    
    processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        capturedSamples.push(new Float32Array(input));
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
    
    recordBtnNormal.classList.add('recording');
    recordBtnNormal.textContent = 'Recording...';
    resultNormal.textContent = 'Listening...';
    
    let stopped = false;
    let autoCapTimeout;
    
    const stopRecording = () => {
        if (stopped) return;
        stopped = true;
        
        if (autoCapTimeout) clearTimeout(autoCapTimeout);
        
        source.disconnect();
        processor.disconnect();
        recordBtnNormal.classList.remove('recording');
        recordBtnNormal.textContent = 'Hold to Talk';
        
        let totalLen = capturedSamples.reduce((acc, val) => acc + val.length, 0);
        if (totalLen === 0) return;
        
        let flat = new Float32Array(totalLen);
        let offset = 0;
        for (let arr of capturedSamples) {
            flat.set(arr, offset);
            offset += arr.length;
        }
        
        let finalSamples = flat.length > SAMPLE_RATE ? flat.slice(0, SAMPLE_RATE) : flat;
        const wavBlob = float32ToWav(finalSamples, SAMPLE_RATE);
        
        inferAudio(wavBlob, resultNormal, false);
        
        // Remove listeners
        window.removeEventListener('mouseup', stopRecording);
    };
    
    autoCapTimeout = setTimeout(stopRecording, 1000);
    window.addEventListener('mouseup', stopRecording);
});

// For touch devices support
recordBtnNormal.addEventListener('touchstart', async (e) => {
    e.preventDefault();
    if (recordBtnNormal.classList.contains('recording')) return;
    
    await initAudio();
    const source = audioContext.createMediaStreamSource(mediaStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    let capturedSamples = [];
    
    processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        capturedSamples.push(new Float32Array(input));
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
    
    recordBtnNormal.classList.add('recording');
    recordBtnNormal.textContent = 'Recording...';
    resultNormal.textContent = 'Listening...';
    
    let stopped = false;
    let autoCapTimeout;
    
    const stopRecordingTouch = () => {
        if (stopped) return;
        stopped = true;
        
        if (autoCapTimeout) clearTimeout(autoCapTimeout);
        
        source.disconnect();
        processor.disconnect();
        recordBtnNormal.classList.remove('recording');
        recordBtnNormal.textContent = 'Hold to Talk';
        
        let totalLen = capturedSamples.reduce((acc, val) => acc + val.length, 0);
        if (totalLen === 0) return;
        
        let flat = new Float32Array(totalLen);
        let offset = 0;
        for (let arr of capturedSamples) {
            flat.set(arr, offset);
            offset += arr.length;
        }
        
        let finalSamples = flat.length > SAMPLE_RATE ? flat.slice(0, SAMPLE_RATE) : flat;
        const wavBlob = float32ToWav(finalSamples, SAMPLE_RATE);
        
        inferAudio(wavBlob, resultNormal, false);
        
        window.removeEventListener('touchend', stopRecordingTouch);
    };
    
    autoCapTimeout = setTimeout(stopRecordingTouch, 1000);
    window.addEventListener('touchend', stopRecordingTouch);
});


// --- Live Mode & Global Streaming ---
const ctxWave = waveformCanvas.getContext('2d', { willReadFrequently: true });
const ctxSpec = spectrogramCanvas.getContext('2d', { willReadFrequently: true });
const ctxDet = detectionsCanvas.getContext('2d');

const KEYWORD_COLORS = {
    'YES': '#ffffff',
    'NO': '#888888',
    'UP': '#aaaaaa',
    'DOWN': '#666666',
    'OTHER': '#444444'
};

let highlights = [];

function drawHighlight(keyword) {
    if (!lastInferenceTime) return;
    const now = Date.now();
    const actualDelta = now - lastInferenceTime;
    const pixelsPerMs = detectionsCanvas.width / 15000;
    const blockWidth = (actualDelta * pixelsPerMs) - 1; 
    
    highlights.push({ x: detectionsCanvas.width, keyword: keyword, width: Math.max(1, blockWidth) });
}

function renderVisuals() {
    if (!analyser) return;
    
    const pixelsPerFrame = waveformCanvas.width / (15000 / (1000 / 60));
    scrollAccumulator += pixelsPerFrame;
    const speed = Math.floor(scrollAccumulator);
    
    if (speed >= 1) {
        scrollAccumulator -= speed;
        
        // Waveform
        const wData = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteTimeDomainData(wData);
        
        const wImg = ctxWave.getImageData(speed, 0, waveformCanvas.width - speed, waveformCanvas.height);
        ctxWave.putImageData(wImg, 0, 0);
        ctxWave.clearRect(waveformCanvas.width - speed, 0, speed, waveformCanvas.height);
        
        ctxWave.beginPath();
        ctxWave.strokeStyle = '#ffffff';
        ctxWave.lineWidth = 2;
        for (let i = 0; i < wData.length; i++) {
            const v = wData[i] / 128.0;
            const y = (v * waveformCanvas.height) / 2;
            const x = waveformCanvas.width - speed + (i / wData.length) * speed;
            if (i === 0) ctxWave.moveTo(x, y);
            else ctxWave.lineTo(x, y);
        }
        ctxWave.stroke();

        // Spectrogram
        const sData = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(sData);
        
        const sImg = ctxSpec.getImageData(speed, 0, spectrogramCanvas.width - speed, spectrogramCanvas.height);
        ctxSpec.putImageData(sImg, 0, 0);
        
        for (let i = 0; i < sData.length; i++) {
            const val = sData[i];
            const percent = val / 255;
            const gray = Math.floor(percent * 255);
            ctxSpec.fillStyle = `rgb(${gray},${gray},${gray})`;
            const y = spectrogramCanvas.height - (i / sData.length) * spectrogramCanvas.height;
            ctxSpec.fillRect(spectrogramCanvas.width - speed, y, speed, spectrogramCanvas.height / sData.length + 1);
        }
        
        // Detections timeline
        ctxDet.clearRect(0, 0, detectionsCanvas.width, detectionsCanvas.height);
        highlights.forEach((h) => {
            h.x -= speed;
            const color = KEYWORD_COLORS[h.keyword] || KEYWORD_COLORS['OTHER'];
            ctxDet.fillStyle = color;
            ctxDet.fillRect(h.x - h.width, 0, h.width, detectionsCanvas.height);
        });
        highlights = highlights.filter(h => h.x > 0);
    }

    if (isLive) {
        animationId = requestAnimationFrame(renderVisuals);
    }
}

function syncVoiceUI() {
    const gameToggle = document.getElementById('game-voice-toggle');
    const gameStatusDot = document.querySelector('.status-indicator-dot');
    const gameStatusText = document.getElementById('game-voice-status-text');
    
    if (isLive) {
        if (gameToggle) {
            gameToggle.textContent = 'Disable Voice Input';
            gameToggle.classList.add('active');
        }
        if (gameStatusDot) gameStatusDot.classList.add('active');
        if (gameStatusText) {
            gameStatusText.textContent = 'Voice Input: Active';
            gameStatusText.classList.add('active');
        }
        
        recordBtnLive.textContent = 'Stop Live Stream';
        recordBtnLive.classList.add('danger');
        resultLive.textContent = 'Listening...';
        resultLive.className = 'result-display-live';
    } else {
        if (gameToggle) {
            gameToggle.textContent = 'Enable Voice Input';
            gameToggle.classList.remove('active');
        }
        if (gameStatusDot) gameStatusDot.classList.remove('active');
        if (gameStatusText) {
            gameStatusText.textContent = 'Voice Input: Off';
            gameStatusText.classList.remove('active');
        }
        
        recordBtnLive.textContent = 'Start Live Stream';
        recordBtnLive.classList.remove('danger');
        resultLive.textContent = 'Ready';
        resultLive.className = 'result-display-live';
    }
}

async function startLive() {
    await initAudio();
    isLive = true;
    syncVoiceUI();
    
    const source = audioContext.createMediaStreamSource(mediaStream);
    
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    scriptProcessor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < input.length; i++) {
            circularBuffer[bufferIndex] = input[i];
            bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
        }
    };
    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
    
    // Clear canvases
    ctxWave.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    ctxSpec.clearRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
    ctxDet.clearRect(0, 0, detectionsCanvas.width, detectionsCanvas.height);
    highlights = [];
    lastInferenceTime = Date.now();
    scrollAccumulator = 0;
    
    renderVisuals();
    
    function scheduleNextInference() {
        if (!isLive) return;
        liveTimeout = setTimeout(() => {
            let samples = new Float32Array(BUFFER_SIZE);
            for (let i = 0; i < BUFFER_SIZE; i++) {
                samples[i] = circularBuffer[(bufferIndex + i) % BUFFER_SIZE];
            }
            
            const wavBlob = float32ToWav(samples, SAMPLE_RATE);
            inferAudio(wavBlob, resultLive, true).then(() => {
                lastInferenceTime = Date.now();
                scheduleNextInference();
            }).catch(() => {
                lastInferenceTime = Date.now();
                scheduleNextInference();
            });
        }, currentInterval);
    }
    
    scheduleNextInference();
}

function stopLive() {
    isLive = false;
    if (liveTimeout) clearTimeout(liveTimeout);
    if (animationId) cancelAnimationFrame(animationId);
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    syncVoiceUI();
}

recordBtnLive.addEventListener('click', () => {
    if (isLive) {
        stopLive();
    } else {
        startLive();
    }
});


// --- Games/Voice Arcade Setup ---
const gameCanvas = document.getElementById('game-canvas');
const gameOverlay = document.getElementById('game-overlay');
const startGameBtn = document.getElementById('start-game-btn');
const gameScore = document.getElementById('game-score');
const gameHighscore = document.getElementById('game-highscore');
const gameLastCommand = document.getElementById('game-last-command');

if (gameCanvas) {
    initGames(gameCanvas, gameOverlay, startGameBtn, gameScore, gameHighscore, gameLastCommand);
    
    startGameBtn.addEventListener('click', () => {
        startGame();
    });
    
    const gameBtns = document.querySelectorAll('.game-btn');
    gameBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            gameBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            switchGame(btn.dataset.game);
        });
    });
    
    const gameVoiceToggle = document.getElementById('game-voice-toggle');
    if (gameVoiceToggle) {
        gameVoiceToggle.addEventListener('click', () => {
            if (isLive) {
                stopLive();
            } else {
                startLive();
            }
        });
    }
}

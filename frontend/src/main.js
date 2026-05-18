import WaveSurfer from 'wavesurfer.js';

const BACKEND_URL = 'http://127.0.0.1:18000';
let audioContext = null;
let mediaStream = null;

// UI Elements
const modelSelect = document.getElementById('model-select');
const customModelBtn = document.getElementById('custom-model-btn');
const tabs = document.querySelectorAll('.tab');
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

document.getElementById('waveform').appendChild(waveformCanvas);
document.getElementById('spectrogram').appendChild(spectrogramCanvas);
document.getElementById('detections').appendChild(detectionsCanvas);

let isLive = false;
let liveTimeout = null;
let currentInterval = 200;
let lastInferenceTime = 0;
let scrollAccumulator = 0;

intervalSlider.addEventListener('input', (e) => {
    currentInterval = parseInt(e.target.value, 10);
    intervalDisplay.textContent = `${currentInterval}ms`;
});
let scriptProcessor = null;
let analyser = null;
let animationId = null;

const SAMPLE_RATE = 16000;
const BUFFER_SIZE = SAMPLE_RATE * 1; // 1 second
let circularBuffer = new Float32Array(BUFFER_SIZE);
let bufferIndex = 0;

// Setup Canvases
function resizeCanvases() {
    const w = document.getElementById('waveform').clientWidth;
    const h = document.getElementById('waveform').clientHeight;
    waveformCanvas.width = w;
    waveformCanvas.height = h;
    spectrogramCanvas.width = w;
    spectrogramCanvas.height = h;
    
    const dw = document.getElementById('detections').clientWidth;
    const dh = document.getElementById('detections').clientHeight;
    detectionsCanvas.width = dw;
    detectionsCanvas.height = dh;
}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();

// --- Backend API Calls ---
async function fetchModels() {
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
    } catch (e) {
        console.error("Failed to fetch models", e);
        modelSelect.innerHTML = '<option>Error loading models</option>';
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
            resultElement.textContent = `Detected: ${data.keyword.toUpperCase()}`;
            if (isLiveMode) {
                drawHighlight(data.keyword.toUpperCase());
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

// --- Tabs ---
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        modeContents.forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        document.getElementById(tab.dataset.target).classList.add('active');
        resizeCanvases();
        
        if (tab.dataset.target === 'normal-mode' && isLive) {
            stopLive();
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

recordBtnNormal.addEventListener('click', async () => {
    if (recordBtnNormal.classList.contains('recording')) return;
    
    await initAudio();
    mediaRecorder = new MediaRecorder(mediaStream);
    normalChunks = [];
    
    mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) normalChunks.push(e.data);
    };
    
    mediaRecorder.onstop = () => {
        const blob = new Blob(normalChunks, { type: 'audio/webm' });
        // WebM to WAV conversion is tricky without decoding. 
        // We will just send WebM and let torchaudio handle it, 
        // as sf.read supports webm if ffmpeg is around, but wait, torchaudio doesn't support webm easily without ffmpeg.
        // It's safer to use the float32ToWav pipeline for Normal mode too!
    };
    
    // We'll use the Web Audio pipeline for Normal Mode to guarantee 16kHz WAV
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
    
    setTimeout(() => {
        source.disconnect();
        processor.disconnect();
        recordBtnNormal.classList.remove('recording');
        recordBtnNormal.textContent = 'Hold to Talk';
        
        // Flatten array
        let totalLen = capturedSamples.reduce((acc, val) => acc + val.length, 0);
        let flat = new Float32Array(totalLen);
        let offset = 0;
        for (let arr of capturedSamples) {
            flat.set(arr, offset);
            offset += arr.length;
        }
        
        // Take exact 1 second
        let finalSamples = flat.length > SAMPLE_RATE ? flat.slice(0, SAMPLE_RATE) : flat;
        const wavBlob = float32ToWav(finalSamples, SAMPLE_RATE);
        
        inferAudio(wavBlob, resultNormal, false);
    }, 1000); // 1 second auto stop
});


// --- Live Mode ---
const ctxWave = waveformCanvas.getContext('2d');
const ctxSpec = spectrogramCanvas.getContext('2d');
const ctxDet = detectionsCanvas.getContext('2d');

const KEYWORD_COLORS = {
    'YES': '#3b82f6',
    'NO': '#ef4444',
    'UP': '#10b981',
    'DOWN': '#f59e0b',
    'OTHER': '#a1a1a1'
};

let highlights = [];

function drawHighlight(keyword) {
    if (!lastInferenceTime) return;
    const now = Date.now();
    const actualDelta = now - lastInferenceTime;
    
    // Calculate how many pixels the canvas travels per millisecond for a total of 15000ms
    const pixelsPerMs = detectionsCanvas.width / 15000;
    
    // Width of block based on exactly how much time passed since last inference
    // Subtract 1 pixel to leave a thin black line gap
    const blockWidth = (actualDelta * pixelsPerMs) - 1; 
    
    highlights.push({ x: detectionsCanvas.width, keyword: keyword, width: Math.max(1, blockWidth) });
}

function renderVisuals() {
    if (!analyser) return;
    
    // Target duration is 15 seconds (15000ms). At 60fps, we move (width / 15000) * msPerFrame.
    // However, requestAnimationFrame isn't strictly 60fps. We accumulate float pixels and scroll by integers.
    const pixelsPerFrame = waveformCanvas.width / (15000 / (1000 / 60)); // assuming 60fps for accumulator
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
        ctxWave.strokeStyle = '#3b82f6';
        ctxWave.lineWidth = 2;
        for (let i = 0; i < wData.length; i++) {
            const v = wData[i] / 128.0;
            const y = v * waveformCanvas.height / 2;
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
            const r = Math.floor(percent * 52);
            const g = Math.floor(percent * 211);
            const b = Math.floor(percent * 153 + (1-percent)*255);
            ctxSpec.fillStyle = `rgb(${r},${g},${b})`;
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

async function startLive() {
    await initAudio();
    isLive = true;
    recordBtnLive.textContent = 'Stop Live Stream';
    recordBtnLive.classList.add('danger');
    resultLive.textContent = 'Listening...';
    
    const source = audioContext.createMediaStreamSource(mediaStream);
    
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    scriptProcessor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        // Write to circular buffer
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
    
    // Polling loop for inference
    function scheduleNextInference() {
        if (!isLive) return;
        liveTimeout = setTimeout(() => {
            // Extract last 1s from circular buffer
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
    recordBtnLive.textContent = 'Start Live Stream';
    recordBtnLive.classList.remove('danger');
    resultLive.textContent = 'Ready';
}

recordBtnLive.addEventListener('click', () => {
    if (isLive) {
        stopLive();
    } else {
        startLive();
    }
});

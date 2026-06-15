import WaveSurfer from 'wavesurfer.js';
import { initGames, switchGame, startGame, stopActiveGame, handleGameVoiceCommand } from './games.js';
import * as ort from 'onnxruntime-web/wasm';
import { preprocessMelSpectrogram, preprocessMfcc } from './dsp.js';

const basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);
ort.env.wasm.wasmPaths = {
    'ort-wasm-simd-threaded.wasm': basePath + 'wasm/ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.mjs': basePath + 'wasm/ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.jsep.wasm': basePath + 'wasm/ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.mjs': basePath + 'wasm/ort-wasm-simd-threaded.mjs'
};
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = !window.Capacitor;

const AVAILABLE_MODELS = [
    'PyTorch.onnx',
    'TensorFlow.onnx'
];
let activeModelPath = 'PyTorch.onnx';
const loadedSessions = {};

const CLASSES = ["yes", "no", "up", "down", "other"];

async function loadSession(modelPath) {
    if (loadedSessions[modelPath]) {
        return loadedSessions[modelPath];
    }
    
    let session;
    let isV2 = false;
    let name = modelPath;
    
    if (modelPath.startsWith('custom_')) {
        throw new Error(`Custom model session not found for ${modelPath}`);
    } else {
        const filename = modelPath.split(/[\\/]/).pop();
        const url = `./Models/${filename}`;
        console.log(`Loading ONNX model from: ${url}`);
        session = await ort.InferenceSession.create(url, {
            executionProviders: ['wasm']
        });
        name = filename;
    }
    
    const inputNames = session.inputNames;
    const inputShape = (session.inputMetadata && session.inputMetadata[0]) ? session.inputMetadata[0].shape : [];
    console.log(`Model input shape for ${name}:`, inputShape);
    
    let featuresDim = 40;
    if (inputShape && inputShape.length === 4) {
        if (inputShape[3] === 1) {
            featuresDim = inputShape[1];
        } else {
            featuresDim = inputShape[2];
        }
    } else if (inputShape && inputShape.length === 3) {
        featuresDim = inputShape[1];
    }
    
    if (featuresDim === 64 || name.includes('PyTorch2') || name.includes('PyTorch3')) {
        isV2 = true;
    }
    
    const res = { session, isV2, inputShape, inputName: inputNames[0] };
    loadedSessions[modelPath] = res;
    return res;
}
let audioContext = null;
let mediaStream = null;

// UI Elements
const micSelect = document.getElementById('mic-select');
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

// Dual Mode UI Elements & Canvases
const recordBtnDual = document.getElementById('record-btn-dual');
const resultDualA = document.getElementById('result-dual-a');
const resultDualB = document.getElementById('result-dual-b');
const intervalSliderDual = document.getElementById('interval-slider-dual');
const intervalDisplayDual = document.getElementById('interval-display-dual');
const modelSelectA = document.getElementById('model-select-a');
const modelSelectB = document.getElementById('model-select-b');

const detectionsCanvasA = document.createElement('canvas');
const detectionsCanvasB = document.createElement('canvas');
const waveformCanvasDual = document.createElement('canvas');

const detectionsContainerA = document.getElementById('detections-dual-a');
const detectionsContainerB = document.getElementById('detections-dual-b');
const waveformContainerDual = document.getElementById('waveform-dual');

if (detectionsContainerA) detectionsContainerA.appendChild(detectionsCanvasA);
if (detectionsContainerB) detectionsContainerB.appendChild(detectionsCanvasB);
if (waveformContainerDual) waveformContainerDual.appendChild(waveformCanvasDual);

let isDualLive = false;
let dualLiveTimeout = null;
let currentIntervalDual = 200;
let lastInferenceTimeDual = 0;
let scrollAccumulatorDual = 0;
let animationIdDual = null;
let highlightsA = [];
let highlightsB = [];

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
if (intervalSliderDual) {
    intervalSliderDual.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        currentIntervalDual = val;
        if (intervalDisplayDual) {
            intervalDisplayDual.textContent = `${val}ms`;
        }
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
    
    // Dual mode canvases
    const wElDual = document.getElementById('waveform-dual');
    const dElA = document.getElementById('detections-dual-a');
    const dElB = document.getElementById('detections-dual-b');
    
    if (wElDual) {
        const w = wElDual.clientWidth;
        const h = wElDual.clientHeight;
        if (w > 0 && h > 0) {
            waveformCanvasDual.width = w;
            waveformCanvasDual.height = h;
        }
    }
    
    if (dElA) {
        const dw = dElA.clientWidth;
        const dh = dElA.clientHeight;
        if (dw > 0 && dh > 0) {
            detectionsCanvasA.width = dw;
            detectionsCanvasA.height = dh;
        }
    }
    
    if (dElB) {
        const dw = dElB.clientWidth;
        const dh = dElB.clientHeight;
        if (dw > 0 && dh > 0) {
            detectionsCanvasB.width = dw;
            detectionsCanvasB.height = dh;
        }
    }
}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();

// --- Client-side Model Preprocessing & Inference ---

async function fetchModels() {
    const badge = document.getElementById('backend-status-badge');
    if (badge) {
        badge.textContent = 'Checking';
        badge.className = 'status-badge checking';
    }
    
    // Load config.json dynamically
    try {
        const configRes = await fetch('./config.json');
        const configData = await configRes.json();
        if (configData.keywords) {
            CLASSES.length = 0;
            CLASSES.push(...configData.keywords);
            console.log("Loaded classes dynamically:", CLASSES);
        }
    } catch (err) {
        console.warn("Using default keywords, config.json not found:", err);
    }
    
    try {
        modelSelect.innerHTML = '';
        AVAILABLE_MODELS.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model;
            if (model === activeModelPath) opt.selected = true;
            modelSelect.appendChild(opt);
        });
        
        // Populate Dual Model Selects
        if (modelSelectA && modelSelectB) {
            modelSelectA.innerHTML = '';
            modelSelectB.innerHTML = '';
            AVAILABLE_MODELS.forEach((model, index) => {
                const optA = document.createElement('option');
                optA.value = model;
                optA.textContent = model;
                
                const optB = document.createElement('option');
                optB.value = model;
                optB.textContent = model;
                
                if (index === 0) {
                    optA.selected = true;
                } else if (index === 1 || (AVAILABLE_MODELS.length === 1 && index === 0)) {
                    optB.selected = true;
                }
                
                modelSelectA.appendChild(optA);
                modelSelectB.appendChild(optB);
            });
            updateDualLabels();
        }
        
        if (badge) {
            badge.textContent = 'Active (Local)';
            badge.className = 'status-badge online';
        }
        
        // Load initial model
        await setModel(activeModelPath);
    } catch (e) {
        console.error("Failed to initialize local models", e);
        if (badge) {
            badge.textContent = 'Offline';
            badge.className = 'status-badge offline';
        }
    }
}

async function setModel(path) {
    try {
        activeModelPath = path;
        const badge = document.getElementById('backend-status-badge');
        if (badge) {
            badge.textContent = 'Loading Model...';
            badge.className = 'status-badge checking';
        }
        await loadSession(path);
        if (badge) {
            badge.textContent = 'Active (Local)';
            badge.className = 'status-badge online';
        }
    } catch (e) {
        console.error("Failed to set model", e);
        const badge = document.getElementById('backend-status-badge');
        if (badge) {
            badge.textContent = 'Error Loading';
            badge.className = 'status-badge offline';
        }
    }
}

async function inferAudio(audioData, resultElement, isLiveMode = false, modelPath = null, highlightCallback = null) {
    console.log("[DEBUG] inferAudio called. isLiveMode =", isLiveMode, "modelPath =", modelPath || activeModelPath);
    try {
        const targetModelPath = modelPath || activeModelPath;
        const { session, isV2, inputShape, inputName } = await loadSession(targetModelPath);
        
        let waveform;
        if (audioData instanceof Blob) {
            // Decode WAV blob to Float32Array
            const arrayBuffer = await audioData.arrayBuffer();
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            let samples = audioBuffer.getChannelData(0);
            
            // Resample to 16000Hz if needed
            waveform = samples;
            if (audioBuffer.sampleRate !== SAMPLE_RATE) {
                const ratio = SAMPLE_RATE / audioBuffer.sampleRate;
                const newLength = Math.round(samples.length * ratio);
                const resampled = new Float32Array(newLength);
                for (let i = 0; i < newLength; i++) {
                    const srcIdx = i / ratio;
                    const baseIdx = Math.floor(srcIdx);
                    const nextIdx = Math.min(baseIdx + 1, samples.length - 1);
                    const frac = srcIdx - baseIdx;
                    resampled[i] = samples[baseIdx] * (1.0 - frac) + samples[nextIdx] * frac;
                }
                waveform = resampled;
            }
        } else {
            // Already raw Float32Array at 16000Hz from live audio context
            waveform = audioData;
        }
        
        // Pad or truncate to exactly 16000 samples
        const NUM_SAMPLES = 16000;
        if (waveform.length > NUM_SAMPLES) {
            const start = Math.floor((waveform.length - NUM_SAMPLES) / 2);
            waveform = waveform.slice(start, start + NUM_SAMPLES);
        } else if (waveform.length < NUM_SAMPLES) {
            const padded = new Float32Array(NUM_SAMPLES);
            padded.set(waveform);
            waveform = padded;
        }
        
        // Preprocess features using the JS DSP functions
        let preprocessed;
        if (isV2) {
            preprocessed = preprocessMelSpectrogram(waveform, SAMPLE_RATE, 400, 160, 64);
        } else {
            preprocessed = preprocessMfcc(waveform, SAMPLE_RATE, 40, 64);
        }
        
        // Shape mapping
        let shape;
        const h = preprocessed.rows;
        const w = preprocessed.cols;
        
        if (inputShape.length === 4) {
            if (inputShape[3] === 1 || inputShape[3] === '1') {
                shape = [1, h, w, 1];
            } else {
                shape = [1, 1, h, w];
            }
        } else if (inputShape.length === 3) {
            shape = [1, h, w];
        } else {
            shape = [1, 1, h, w];
        }
        
        const inputTensor = new ort.Tensor('float32', preprocessed.data, shape);
        const feeds = {};
        feeds[inputName] = inputTensor;
        
        console.log("[DEBUG] session.run about to execute");
        const results = await session.run(feeds);
        console.log("[DEBUG] session.run completed successfully");
        const outputName = session.outputNames[0];
        const logits = results[outputName].data;
        
        // Find argmax
        let maxVal = -Infinity;
        let predictedIdx = 0;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                predictedIdx = i;
            }
        }
        
        const keyword = CLASSES[predictedIdx].toUpperCase();
        resultElement.textContent = `Detected: ${keyword}`;
        
        // Remove previous classes
        resultElement.classList.remove('detected-yes', 'detected-no', 'detected-up', 'detected-down', 'detected-other');
        
        // Add class for coloring + pulsating animation
        const kwLower = keyword.toLowerCase();
        if (['yes', 'no', 'up', 'down', 'other'].includes(kwLower)) {
            resultElement.classList.add(`detected-${kwLower}`);
        } else {
            resultElement.classList.add('detected-other');
        }
        
        if (highlightCallback) {
            highlightCallback(keyword);
        } else if (isLiveMode) {
            drawHighlight(keyword);
            handleGameVoiceCommand(keyword);
        }
    } catch (e) {
        console.error("Inference failed", e);
        if (!isLiveMode) {
            resultElement.textContent = "Error";
            resultElement.classList.remove('detected-yes', 'detected-no', 'detected-up', 'detected-down', 'detected-other');
        }
    }
}

async function populateMicSelect() {
    if (!micSelect) return;
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioDevices = devices.filter(device => device.kind === 'audioinput');
        console.log("[DEBUG] populateMicSelect called. Devices count:", audioDevices.length);
        console.log("[DEBUG] populateMicSelect devices list:", JSON.stringify(audioDevices.map(d => ({id: d.deviceId, label: d.label}))));
        
        const currentVal = micSelect.value;
        micSelect.innerHTML = '';
        
        if (audioDevices.length === 0) {
            const opt = document.createElement('option');
            opt.value = 'default';
            opt.textContent = 'No microphones found';
            micSelect.appendChild(opt);
            return;
        }
        
        audioDevices.forEach(device => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            opt.textContent = device.label || `Microphone ${micSelect.length + 1}`;
            micSelect.appendChild(opt);
        });
        
        // Restore value if it still exists
        if (audioDevices.some(d => d.deviceId === currentVal)) {
            micSelect.value = currentVal;
        }
    } catch (e) {
        console.error("Error enumerating devices:", e);
    }
}

function checkSecureContext() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const warning = document.createElement('div');
        warning.className = 'secure-context-warning';
        warning.innerHTML = `
            <div class="warning-card">
                <div class="warning-icon">⚠️</div>
                <h3>Microphone Access Blocked</h3>
                <p>Modern mobile browsers (iOS Safari, Android Chrome) block microphone access over local IP addresses on insecure HTTP.</p>
                <p>To use your mobile phone's microphone, you must use a secure context. Here are your options:</p>
                <ul>
                    <li><strong>Option A (Android Chrome):</strong> Go to <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>, enable the flag, add <code>http://${window.location.hostname}:5173</code> to the list, and relaunch Chrome.</li>
                    <li><strong>Option B (iOS & Android):</strong> Run a free secure tunnel on your computer using <code>npx localtunnel --port 5173</code> (or <code>ngrok http 5173</code>) to get a secure <code>https://...</code> link you can open on your phone.</li>
                    <li><strong>Option C:</strong> Use the application on your computer via <code>http://localhost:5173</code> (which browsers always treat as secure).</li>
                </ul>
                <button class="warning-close-btn" id="warning-close-btn">Dismiss</button>
            </div>
        `;
        document.body.appendChild(warning);
        
        const closeBtn = warning.querySelector('#warning-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => warning.remove());
        }
    }
}

// --- Init ---
checkSecureContext();
fetchModels();
populateMicSelect();

// Listen for device changes
try {
    navigator.mediaDevices.addEventListener('devicechange', populateMicSelect);
} catch (e) {
    console.warn("Failed to add devicechange listener", e);
}

if (micSelect) {
    micSelect.addEventListener('change', async () => {
        if (isLive) {
            stopLive();
            await startLive();
        } else {
            try {
                await initAudio();
            } catch (e) {
                console.error("Failed to change microphone:", e);
            }
        }
    });
}

// Game Speed Slider Setup
const gameSpeedSlider = document.getElementById('game-speed-slider');
const gameSpeedDisplay = document.getElementById('game-speed-display');

window.gameSpeedMultiplier = 1.0;

if (gameSpeedSlider) {
    gameSpeedSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        window.gameSpeedMultiplier = val;
        if (gameSpeedDisplay) {
            gameSpeedDisplay.textContent = `${val.toFixed(1)}x`;
        }
    });
}

modelSelect.addEventListener('change', (e) => {
    setModel(e.target.value);
});

if (window.electronAPI) {
    customModelBtn.addEventListener('click', async () => {
        const fileInfo = await window.electronAPI.openFile();
        if (fileInfo) {
            const name = fileInfo.name;
            const buffer = fileInfo.data; // Uint8Array
            
            try {
                console.log(`Loading custom model client-side: ${name}`);
                const session = await ort.InferenceSession.create(buffer, {
                    executionProviders: ['wasm']
                });
                
                const inputNames = session.inputNames;
                const inputShape = (session.inputMetadata && session.inputMetadata[0]) ? session.inputMetadata[0].shape : [];
                
                let isV2 = false;
                let featuresDim = 40;
                if (inputShape && inputShape.length === 4) {
                    if (inputShape[3] === 1) featuresDim = inputShape[1];
                    else featuresDim = inputShape[2];
                } else if (inputShape && inputShape.length === 3) {
                    featuresDim = inputShape[1];
                }
                if (featuresDim === 64 || name.includes('PyTorch2') || name.includes('PyTorch3')) {
                    isV2 = true;
                }
                
                const res = { session, isV2, inputShape, inputName: inputNames[0] };
                const key = `custom_${name}`;
                loadedSessions[key] = res;
                
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = `Custom: ${name}`;
                opt.selected = true;
                modelSelect.appendChild(opt);
                
                await setModel(key);
            } catch (err) {
                console.error("Failed to load custom model", err);
                alert(`Failed to load custom model: ${err.message}`);
            }
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
        
        // Stop live streams if switching away from their modes
        if (tab.dataset.target !== 'live-mode' && isLive) {
            stopLive();
        }
        if (tab.dataset.target !== 'dual-mode' && isDualLive) {
            stopDualLive();
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
    }
    
    const selectedDeviceId = micSelect ? micSelect.value : 'default';
    console.log("[DEBUG] initAudio called. DeviceId selected:", selectedDeviceId);
    
    // Reuse existing stream if it's active and has the correct deviceId
    if (mediaStream && mediaStream.active) {
        const tracks = mediaStream.getAudioTracks();
        if (tracks.length > 0 && tracks[0].readyState === 'live') {
            const settings = tracks[0].getSettings ? tracks[0].getSettings() : {};
            if (selectedDeviceId === 'default' || !selectedDeviceId || settings.deviceId === selectedDeviceId) {
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }
                return;
            }
        }
    }
    
    const constraints = {
        audio: (selectedDeviceId === 'default' || !selectedDeviceId || selectedDeviceId === 'loading') 
            ? true 
            : { deviceId: { exact: selectedDeviceId } },
        video: false
    };
    
    // Stop previous track if any since we are switching to a new device or stream
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    
    try {
        console.log("[DEBUG] getUserMedia called with constraints:", JSON.stringify(constraints));
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("[DEBUG] getUserMedia succeeded. Stream active:", mediaStream.active);
    } catch (err) {
        if (err.name === 'OverconstrainedError' || err.name === 'NotFoundError') {
            console.warn("Requested microphone constraints failed, falling back to default microphone", err);
            console.log("[DEBUG] falling back getUserMedia");
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            console.log("[DEBUG] fallback getUserMedia succeeded. Stream active:", mediaStream.active);
        } else {
            throw err;
        }
    }
    
    // Populate mic dropdown labels once permission is granted
    const hasPlaceholders = micSelect ? Array.from(micSelect.options).some(opt => opt.textContent.startsWith('Microphone ')) : false;
    if (micSelect && (micSelect.options.length <= 1 || hasPlaceholders)) {
        await populateMicSelect();
        const tracks = mediaStream.getAudioTracks();
        const activeTrackDeviceId = (tracks.length > 0 && tracks[0].getSettings) ? tracks[0].getSettings().deviceId : null;
        if (activeTrackDeviceId && Array.from(micSelect.options).some(opt => opt.value === activeTrackDeviceId)) {
            micSelect.value = activeTrackDeviceId;
        } else if (selectedDeviceId && selectedDeviceId !== 'default' && selectedDeviceId !== 'loading') {
            micSelect.value = selectedDeviceId;
        }
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
    resultNormal.className = 'result-display';
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
    resultNormal.className = 'result-display';
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
    'YES': '#97DCAE',
    'NO': '#DC8D82',
    'UP': '#8BB8D7',
    'DOWN': '#8290BA',
    'OTHER': '#F6E7B9'
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
    console.log("[DEBUG] startLive called");
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
        console.log("[DEBUG] scheduleNextInference scheduled with interval:", currentInterval);
        liveTimeout = setTimeout(() => {
            let samples = new Float32Array(BUFFER_SIZE);
            for (let i = 0; i < BUFFER_SIZE; i++) {
                samples[i] = circularBuffer[(bufferIndex + i) % BUFFER_SIZE];
            }
            
            console.log("[DEBUG] scheduleNextInference: calling inferAudio");
            inferAudio(samples, resultLive, true).then(() => {
                console.log("[DEBUG] scheduleNextInference: inferAudio resolved successfully");
                lastInferenceTime = Date.now();
                scheduleNextInference();
            }).catch((err) => {
                console.error("[DEBUG] scheduleNextInference: inferAudio failed:", err);
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

// --- Dual Model Live Drawing and Inference ---
const ctxDetA = detectionsCanvasA.getContext('2d');
const ctxDetB = detectionsCanvasB.getContext('2d');
const ctxWaveDual = waveformCanvasDual.getContext('2d', { willReadFrequently: true });

function updateDualLabels() {
    const labelResultA = document.getElementById('label-result-a');
    const labelResultB = document.getElementById('label-result-b');
    const legendLabelA = document.getElementById('legend-label-a');
    const legendLabelB = document.getElementById('legend-label-b');
    
    if (modelSelectA && modelSelectA.selectedOptions.length > 0) {
        const nameA = modelSelectA.selectedOptions[0].textContent;
        if (labelResultA) labelResultA.textContent = `${nameA} Detection`;
        if (legendLabelA) legendLabelA.textContent = `${nameA} Timeline (15s)`;
    }
    if (modelSelectB && modelSelectB.selectedOptions.length > 0) {
        const nameB = modelSelectB.selectedOptions[0].textContent;
        if (labelResultB) labelResultB.textContent = `${nameB} Detection`;
        if (legendLabelB) legendLabelB.textContent = `${nameB} Timeline (15s)`;
    }
}

if (modelSelectA) modelSelectA.addEventListener('change', updateDualLabels);
if (modelSelectB) modelSelectB.addEventListener('change', updateDualLabels);

function drawHighlightDual(modelType, keyword) {
    if (!lastInferenceTimeDual) return;
    const now = Date.now();
    const actualDelta = now - lastInferenceTimeDual;
    const pixelsPerMs = detectionsCanvasA.width / 15000;
    const blockWidth = (actualDelta * pixelsPerMs) - 1;
    
    if (modelType === 'A') {
        highlightsA.push({ x: detectionsCanvasA.width, keyword: keyword, width: Math.max(1, blockWidth) });
    } else {
        highlightsB.push({ x: detectionsCanvasB.width, keyword: keyword, width: Math.max(1, blockWidth) });
    }
}

function renderDualVisuals() {
    if (!analyser) return;
    
    const pixelsPerFrame = waveformCanvasDual.width / (15000 / (1000 / 60));
    scrollAccumulatorDual += pixelsPerFrame;
    const speed = Math.floor(scrollAccumulatorDual);
    
    if (speed >= 1) {
        scrollAccumulatorDual -= speed;
        
        // Waveform
        const wData = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteTimeDomainData(wData);
        
        const wImg = ctxWaveDual.getImageData(speed, 0, waveformCanvasDual.width - speed, waveformCanvasDual.height);
        ctxWaveDual.putImageData(wImg, 0, 0);
        ctxWaveDual.clearRect(waveformCanvasDual.width - speed, 0, speed, waveformCanvasDual.height);
        
        ctxWaveDual.beginPath();
        ctxWaveDual.strokeStyle = '#ffffff';
        ctxWaveDual.lineWidth = 2;
        for (let i = 0; i < wData.length; i++) {
            const v = wData[i] / 128.0;
            const y = (v * waveformCanvasDual.height) / 2;
            const x = waveformCanvasDual.width - speed + (i / wData.length) * speed;
            if (i === 0) ctxWaveDual.moveTo(x, y);
            else ctxWaveDual.lineTo(x, y);
        }
        ctxWaveDual.stroke();
        
        // Detections A
        ctxDetA.clearRect(0, 0, detectionsCanvasA.width, detectionsCanvasA.height);
        highlightsA.forEach((h) => {
            h.x -= speed;
            const color = KEYWORD_COLORS[h.keyword] || KEYWORD_COLORS['OTHER'];
            ctxDetA.fillStyle = color;
            ctxDetA.fillRect(h.x - h.width, 0, h.width, detectionsCanvasA.height);
        });
        highlightsA = highlightsA.filter(h => h.x > 0);

        // Detections B
        ctxDetB.clearRect(0, 0, detectionsCanvasB.width, detectionsCanvasB.height);
        highlightsB.forEach((h) => {
            h.x -= speed;
            const color = KEYWORD_COLORS[h.keyword] || KEYWORD_COLORS['OTHER'];
            ctxDetB.fillStyle = color;
            ctxDetB.fillRect(h.x - h.width, 0, h.width, detectionsCanvasB.height);
        });
        highlightsB = highlightsB.filter(h => h.x > 0);
    }

    if (isDualLive) {
        animationIdDual = requestAnimationFrame(renderDualVisuals);
    }
}

function syncDualLiveUI() {
    if (isDualLive) {
        recordBtnDual.textContent = 'Stop Dual Stream';
        recordBtnDual.classList.add('danger');
        resultDualA.textContent = 'Listening...';
        resultDualA.className = 'result-display-live';
        resultDualB.textContent = 'Listening...';
        resultDualB.className = 'result-display-live';
    } else {
        recordBtnDual.textContent = 'Start Dual Stream';
        recordBtnDual.classList.remove('danger');
        resultDualA.textContent = 'Ready';
        resultDualA.className = 'result-display-live';
        resultDualB.textContent = 'Ready';
        resultDualB.className = 'result-display-live';
    }
}

async function startDualLive() {
    await initAudio();
    isDualLive = true;
    syncDualLiveUI();
    
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
    ctxWaveDual.clearRect(0, 0, waveformCanvasDual.width, waveformCanvasDual.height);
    ctxDetA.clearRect(0, 0, detectionsCanvasA.width, detectionsCanvasA.height);
    ctxDetB.clearRect(0, 0, detectionsCanvasB.width, detectionsCanvasB.height);
    highlightsA = [];
    highlightsB = [];
    lastInferenceTimeDual = Date.now();
    scrollAccumulatorDual = 0;
    
    renderDualVisuals();
    
    function scheduleNextInferenceDual() {
        if (!isDualLive) return;
        dualLiveTimeout = setTimeout(() => {
            let samples = new Float32Array(BUFFER_SIZE);
            for (let i = 0; i < BUFFER_SIZE; i++) {
                samples[i] = circularBuffer[(bufferIndex + i) % BUFFER_SIZE];
            }
            
            const modelA = modelSelectA ? modelSelectA.value : '';
            const modelB = modelSelectB ? modelSelectB.value : '';
            
            Promise.all([
                inferAudio(samples, resultDualA, true, modelA, (keyword) => drawHighlightDual('A', keyword)),
                inferAudio(samples, resultDualB, true, modelB, (keyword) => drawHighlightDual('B', keyword))
            ]).then(() => {
                lastInferenceTimeDual = Date.now();
                scheduleNextInferenceDual();
            }).catch(() => {
                lastInferenceTimeDual = Date.now();
                scheduleNextInferenceDual();
            });
        }, currentIntervalDual);
    }
    
    scheduleNextInferenceDual();
}

function stopDualLive() {
    isDualLive = false;
    if (dualLiveTimeout) clearTimeout(dualLiveTimeout);
    if (animationIdDual) cancelAnimationFrame(animationIdDual);
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    syncDualLiveUI();
}

if (recordBtnDual) {
    recordBtnDual.addEventListener('click', () => {
        if (isDualLive) {
            stopDualLive();
        } else {
            startDualLive();
        }
    });
}

// Mobile Sidebar Foldable Controls
const sidebar = document.getElementById('app-sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarClose = document.getElementById('sidebar-close');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const sidebarTabs = document.querySelectorAll('.sidebar-tab');

if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.add('open');
        if (sidebarOverlay) sidebarOverlay.classList.add('visible');
    });
}

const closeSidebar = () => {
    if (sidebar) sidebar.classList.remove('open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('visible');
};

if (sidebarClose) {
    sidebarClose.addEventListener('click', closeSidebar);
}
if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebar);
}
sidebarTabs.forEach(tab => {
    tab.addEventListener('click', closeSidebar);
});

// Set dynamic version and release date in About screen
const aboutVersion = document.getElementById('about-version');
const aboutReleaseDate = document.getElementById('about-release-date');
if (aboutVersion && typeof __APP_VERSION__ !== 'undefined') {
  aboutVersion.textContent = __APP_VERSION__;
}
if (aboutReleaseDate && typeof __RELEASE_DATE__ !== 'undefined') {
  aboutReleaseDate.textContent = __RELEASE_DATE__;
}



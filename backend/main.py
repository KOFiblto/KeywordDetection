import os
import json
import io
import wave
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory logic to support pyinstaller and dev
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    base_dir = exe_dir
    for _ in range(5):
        if os.path.exists(os.path.join(base_dir, "config.json")):
            break
        base_dir = os.path.dirname(base_dir)
    BASE_DIR = os.path.abspath(base_dir)
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
        CLASSES = config_data.get("keywords", ["yes", "no", "up", "down", "other"])
        TARGET_SAMPLE_RATE = config_data.get("target_sample_rate", 16000)
        NUM_SAMPLES = config_data.get("num_samples", 16000)
except Exception as e:
    print(f"Error loading config.json: {e}")
    CLASSES = ["yes", "no", "up", "down", "other"]
    TARGET_SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000

# ----------------- Pure NumPy Preprocessing -----------------
def numpy_hz_to_mel(freq, mel_scale='htk'):
    if mel_scale == 'htk':
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    if freq < 1000.0:
        return freq * 3.0 / 200.0
    return 15.0 + np.log(freq / 1000.0) / (np.log(6.4) / 27.0)

def numpy_mel_to_hz(mels, mel_scale='htk'):
    if mel_scale == 'htk':
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    freqs = mels * 200.0 / 3.0
    min_log_mel = 15.0
    log_t = mels >= min_log_mel
    freqs[log_t] = 1000.0 * np.exp((mels[log_t] - min_log_mel) * (np.log(6.4) / 27.0))
    return freqs

def numpy_melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=None, mel_scale='htk'):
    all_freqs = np.linspace(0, sample_rate / 2, n_freqs)
    m_min = numpy_hz_to_mel(f_min, mel_scale)
    m_max = numpy_hz_to_mel(f_max, mel_scale)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = numpy_mel_to_hz(m_pts, mel_scale)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    if norm == 'slaney':
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm[None, :]
    return fb

def numpy_stft(x, n_fft, hop_length):
    n = np.arange(n_fft)
    window = np.sin(np.pi * n / n_fft) ** 2
    pad_len = n_fft // 2
    x_padded = np.pad(x, pad_len, mode='reflect')
    n_frames = 1 + (len(x_padded) - n_fft) // hop_length
    stft_matrix = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = x_padded[start:start + n_fft] * window
        stft_matrix[:, i] = np.fft.rfft(frame, n_fft)
    return stft_matrix

def compute_dct_ii(x):
    N = x.shape[0]
    n = np.arange(N)
    k = np.arange(N)[:, None]
    cos_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    y = 2.0 * np.dot(cos_matrix, x)
    y[0] *= 0.5 * np.sqrt(1.0 / N)
    y[1:] *= 0.5 * np.sqrt(2.0 / N)
    return y

def preprocess_mel_spectrogram(waveform_np, sample_rate, n_fft=400, hop_length=160, n_mels=64):
    stft = numpy_stft(waveform_np, n_fft, hop_length)
    power = np.abs(stft) ** 2
    fb = numpy_melscale_fbanks(n_fft // 2 + 1, 0.0, sample_rate / 2.0, n_mels, sample_rate)
    mel = np.dot(fb.T, power)
    return np.log(mel + 1e-6)

def preprocess_mfcc(waveform_np, sample_rate, n_mfcc=40, n_mels=64):
    stft = numpy_stft(waveform_np, n_fft=400, hop_length=200)
    power = np.abs(stft) ** 2
    fb = numpy_melscale_fbanks(201, 0.0, sample_rate / 2.0, n_mels, sample_rate)
    mel = np.dot(fb.T, power)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mfcc = compute_dct_ii(log_mel)[:n_mfcc]
    return mfcc

# State
model_cache = {}

def get_or_load_model(path: str):
    path = os.path.abspath(path)
    if path in model_cache:
        return model_cache[path]["session"], model_cache[path]["is_v2"]
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    is_v2 = False
    
    if path.endswith(".onnx"):
        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        session = ort.InferenceSession(path, providers=providers)
        
        # Check expected input shape to dynamically identify version
        input_shape = session.get_inputs()[0].shape
        
        features_dim = 40
        if len(input_shape) == 4:
            if input_shape[3] == 1 or input_shape[3] == '1':
                features_dim = input_shape[1]
            else:
                features_dim = input_shape[2]
        elif len(input_shape) == 3:
            features_dim = input_shape[1]

        if features_dim == 64 or 'PyTorch2' in path or 'PyTorch3' in path:
            is_v2 = True
    else:
        raise ValueError("Unsupported model extension. Only self-sustaining .onnx models are supported.")
        
    model_cache[path] = {"session": session, "is_v2": is_v2}
    return session, is_v2

# State
current_model_path = os.path.abspath(os.path.join(BASE_DIR, "PyTorch", "Models", "PyTorch.onnx"))
current_model_type = "onnx"
ort_session = None
is_v2_model = False

def load_model(path: str):
    global current_model_path, current_model_type, ort_session, is_v2_model
    path = os.path.abspath(path)
    session, is_v2 = get_or_load_model(path)
    ort_session = session
    is_v2_model = is_v2
    current_model_path = path
    current_model_type = "onnx"
    print(f"Loaded active model: {path} (is_v2: {is_v2})")

# Initial load
try:
    load_model(current_model_path)
except Exception as e:
    print(f"Initial model load failed: {e}")

class SetModelRequest(BaseModel):
    model_path: str

@app.post("/set_model")
def set_model(req: SetModelRequest):
    try:
        load_model(req.model_path)
        return {"status": "success", "model_path": req.model_path, "type": current_model_type}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
def get_models():
    pytorch_models_dir = os.path.join(BASE_DIR, "PyTorch", "Models")
    tensorflow_models_dir = os.path.join(BASE_DIR, "Tensorflow", "Models")
    
    models = []
    
    # scan PyTorch/Models
    if os.path.exists(pytorch_models_dir):
        for f in os.listdir(pytorch_models_dir):
            if f.endswith(".onnx"):
                models.append(os.path.join(pytorch_models_dir, f))
                
    # scan Tensorflow/Models
    if os.path.exists(tensorflow_models_dir):
        for f in os.listdir(tensorflow_models_dir):
            if f.endswith(".onnx"):
                models.append(os.path.join(tensorflow_models_dir, f))
                
    models.sort(key=lambda x: os.path.basename(x))
    existing = [m for m in models if os.path.exists(m)]
    return {"available_models": existing, "current_model": current_model_path}

@app.post("/infer")
async def infer(audio: UploadFile = File(...), model_path: str = Form(None)):
    audio_bytes = await audio.read()
    
    try:
        if model_path:
            session, is_v2 = get_or_load_model(model_path)
        else:
            if not ort_session:
                raise HTTPException(status_code=500, detail="No model loaded")
            session, is_v2 = ort_session, is_v2_model

        waveform_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        
        # Ensure correct channel dimension [time] (mono conversion)
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)
            
        # Resample if needed using linear interpolation
        if sr != TARGET_SAMPLE_RATE:
            num_samples = int(len(waveform_np) * TARGET_SAMPLE_RATE / sr)
            x_old = np.linspace(0, len(waveform_np) - 1, len(waveform_np))
            x_new = np.linspace(0, len(waveform_np) - 1, num_samples)
            waveform_np = np.interp(x_new, x_old, waveform_np).astype(np.float32)

        # Truncate or pad to exactly NUM_SAMPLES
        if len(waveform_np) > NUM_SAMPLES:
            start = (len(waveform_np) - NUM_SAMPLES) // 2
            waveform_np = waveform_np[start:start + NUM_SAMPLES]
        elif len(waveform_np) < NUM_SAMPLES:
            waveform_np = np.pad(waveform_np, (0, NUM_SAMPLES - len(waveform_np)), mode='constant')

        # Compute preprocessed features
        if is_v2:
            features = preprocess_mel_spectrogram(
                waveform_np, sample_rate=TARGET_SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64
            )
        else:
            features = preprocess_mfcc(
                waveform_np, sample_rate=TARGET_SAMPLE_RATE, n_mfcc=40, n_mels=64
            )

        # Add batch and channel dimensions: shape will be [1, 1, height, width]
        features_np = features[np.newaxis, np.newaxis, :, :].astype(np.float32)
        
        # Check expected input shape from ONNX model
        input_details = session.get_inputs()[0]
        expected_shape = input_details.shape
        
        # Adjust features_np shape to match expected_shape dimensions dynamically
        if len(expected_shape) == 4:
            if expected_shape[3] == 1 or expected_shape[3] == '1':
                # Transpose from [batch, channel, height, width] to [batch, height, width, channel]
                features_np = features_np.transpose(0, 2, 3, 1)
        elif len(expected_shape) == 3:
            # Squeeze channel dimension if model expects 3D input [batch, height, width]
            features_np = features_np.squeeze(1)

        ort_inputs = {input_details.name: features_np}
        logits = session.run(None, ort_inputs)[0]
        predicted_idx = np.argmax(logits, axis=1)[0]

        return {
            "status": "success",
            "keyword": CLASSES[predicted_idx],
            "class_index": int(predicted_idx),
            "classes": CLASSES
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18000)


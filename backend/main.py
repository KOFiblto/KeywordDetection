import os
import json
import io
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))
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

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MFCC Transform exactly like PyTorch.ipynb
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=TARGET_SAMPLE_RATE, n_mfcc=40, melkwargs={"n_mels": 64}
).to(DEVICE)

# Models are loaded and run using ONNX Runtime directly.

# Log-Mel Spectrogram for PyTorch2 models
class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_fft=400, hop_length=160, n_mels=64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, x):
        x = self.mel(x)
        return torch.log(x + 1e-6)

# State
current_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PyTorch", "Models", "PyTorch.onnx"))
current_model_type = "onnx"
ort_session = None

def load_model(path: str):
    global current_model_path, current_model_type, ort_session, mfcc_transform
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    is_v2 = False
    
    if path.endswith(".onnx"):
        # providers options, use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(path, providers=providers)
        
        # Check expected input shape to dynamically identify version
        input_shape = session.get_inputs()[0].shape
        # input_shape[2] represents features dim (64 for v2, 40 for v1)
        if len(input_shape) > 2 and (input_shape[2] == 64 or 'PyTorch2' in path):
            is_v2 = True
            
        ort_session = session
        current_model_type = "onnx"
    else:
        raise ValueError("Unsupported model extension. Only self-sustaining .onnx models are supported.")
        
    # Instantiate and select correct transform
    if is_v2:
        mfcc_transform = LogMelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64
        ).to(DEVICE)
        print("Using LogMelSpectrogram (v2) transform.")
    else:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SAMPLE_RATE, n_mfcc=40, melkwargs={"n_mels": 64}
        ).to(DEVICE)
        print("Using MFCC (v1) transform.")
        
    current_model_path = path
    print(f"Loaded model: {path}")

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
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pytorch_models_dir = os.path.join(base_dir, "PyTorch", "Models")
    tensorflow_models_dir = os.path.join(base_dir, "Tensorflow", "Models")
    
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
                
    # sort models to make order consistent/clean
    models.sort(key=lambda x: os.path.basename(x))
    
    # filter out non-existent
    existing = [m for m in models if os.path.exists(m)]
    return {"available_models": existing, "current_model": current_model_path}

@app.post("/infer")
async def infer(audio: UploadFile = File(...)):
    if not ort_session:
        raise HTTPException(status_code=500, detail="No model loaded")

    audio_bytes = await audio.read()
    
    try:
        waveform_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        waveform = torch.from_numpy(waveform_np)
        
        # Ensure correct channel dimension [1, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        # Convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Truncate or pad to exactly NUM_SAMPLES
        if waveform.shape[1] > NUM_SAMPLES:
            start = (waveform.shape[1] - NUM_SAMPLES) // 2
            waveform = waveform[:, start:start + NUM_SAMPLES]
        elif waveform.shape[1] < NUM_SAMPLES:
            waveform = F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))

        # Add batch dim -> [1, 1, 16000]
        waveform = waveform.unsqueeze(0).to(DEVICE)
        
        # Compute MFCC -> [1, 1, 40, 81]
        with torch.no_grad():
            mfcc = mfcc_transform(waveform)

        # Run ONNX inference
        mfcc_np = mfcc.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: mfcc_np}
        logits = ort_session.run(None, ort_inputs)[0]
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
    uvicorn.run(app, host="127.0.0.1", port=18000)

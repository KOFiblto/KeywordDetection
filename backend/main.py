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
# Cache for loaded models: {path: {"session": session, "transform": transform}}
model_cache = {}

def get_or_load_model(path: str):
    path = os.path.abspath(path)
    if path in model_cache:
        return model_cache[path]["session"], model_cache[path]["transform"]
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    is_v2 = False
    
    if path.endswith(".onnx"):
        # providers options, use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(path, providers=providers)
        
        # Check expected input shape to dynamically identify version
        input_shape = session.get_inputs()[0].shape
        
        # Determine features dimension based on channel layout
        # PyTorch format: [batch, channel, features, time] -> features is at index 2
        # TensorFlow format: [batch, features, time, channel] -> features is at index 1
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
        
    # Instantiate and select correct transform
    if is_v2:
        transform = LogMelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64
        ).to(DEVICE)
        print(f"Using LogMelSpectrogram (v2) transform for {path}.")
    else:
        transform = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SAMPLE_RATE, n_mfcc=40, melkwargs={"n_mels": 64}
        ).to(DEVICE)
        print(f"Using MFCC (v1) transform for {path}.")
        
    model_cache[path] = {"session": session, "transform": transform}
    return session, transform

# State
current_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PyTorch", "Models", "PyTorch.onnx"))
current_model_type = "onnx"
ort_session = None

def load_model(path: str):
    global current_model_path, current_model_type, ort_session, mfcc_transform
    path = os.path.abspath(path)
    session, transform = get_or_load_model(path)
    ort_session = session
    mfcc_transform = transform
    current_model_path = path
    current_model_type = "onnx"
    print(f"Loaded active model: {path}")

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
async def infer(audio: UploadFile = File(...), model_path: str = Form(None)):
    audio_bytes = await audio.read()
    
    try:
        if model_path:
            session, transform = get_or_load_model(model_path)
        else:
            if not ort_session:
                raise HTTPException(status_code=500, detail="No model loaded")
            session, transform = ort_session, mfcc_transform

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
        
        # Compute MFCC -> [1, 1, 40, 81] or [1, 1, 64, 101]
        with torch.no_grad():
            mfcc = transform(waveform)

        # Run ONNX inference
        mfcc_np = mfcc.cpu().numpy()
        
        # Check expected input shape from ONNX model
        input_details = session.get_inputs()[0]
        expected_shape = input_details.shape
        
        # Adjust mfcc_np shape to match expected_shape dimensions dynamically
        # Normally mfcc_np is [batch, channel, height, width] e.g. [1, 1, 40, 81] or [1, 1, 64, 101]
        if len(expected_shape) == 4:
            # If the last dimension is 1 or '1' (representing channel-last format like TensorFlow [batch, height, width, channels])
            if expected_shape[3] == 1 or expected_shape[3] == '1':
                # Transpose from [batch, channel, height, width] to [batch, height, width, channel]
                mfcc_np = mfcc_np.transpose(0, 2, 3, 1)
        elif len(expected_shape) == 3:
            # Squeeze channel dimension if model expects 3D input [batch, height, width]
            mfcc_np = mfcc_np.squeeze(1)

        ort_inputs = {input_details.name: mfcc_np}
        logits = session.run(None, ort_inputs)[0]
        predicted_idx = np.argmax(logits, axis=1)[0]

        return {
            "status": "success",
            "keyword": CLASSES[predicted_idx],
            "class_index": int(predicted_idx),
            "classes": CLASSES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18000)

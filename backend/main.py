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

# Model definition for .pth
class KeywordCNN(nn.Module):
    def __init__(self, num_classes):
        super(KeywordCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits

# State
current_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PyTorch", "Models", "PyTorch.onnx"))
current_model_type = "onnx" # "onnx" or "pth"
torch_model = None
ort_session = None

def load_model(path: str):
    global current_model_path, current_model_type, torch_model, ort_session
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if path.endswith(".pth"):
        model = KeywordCNN(num_classes=len(CLASSES)).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        torch_model = model
        ort_session = None
        current_model_type = "pth"
    elif path.endswith(".onnx"):
        # providers options, use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(path, providers=providers)
        torch_model = None
        current_model_type = "onnx"
    else:
        raise ValueError("Unsupported model extension. Use .pth or .onnx")
    
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
    models = [
        os.path.join(base_dir, "PyTorch", "Models", "PyTorch.onnx"),
        os.path.join(base_dir, "PyTorch", "Models", "PyTorch.pth"),
        os.path.join(base_dir, "Tensorflow", "Models", "Tensorflow.onnx")
    ]
    # filter out non-existent
    existing = [m for m in models if os.path.exists(m)]
    return {"available_models": existing, "current_model": current_model_path}

@app.post("/infer")
async def infer(audio: UploadFile = File(...)):
    if not (torch_model or ort_session):
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

        if current_model_type == "pth":
            with torch.no_grad():
                logits = torch_model(mfcc)
                _, predicted = logits.max(1)
                predicted_idx = predicted.item()
        else:
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
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18000)

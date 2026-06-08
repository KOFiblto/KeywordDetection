# Webapp Keyword Detection Application Documentation

This document provides a comprehensive historical and technical overview of the web application built to interface with the Keyword Detection model. It details the system architecture, FastAPI backend, Electron frontend, real-time visualization pipelines, the Voice Arcade game suite, and startup utilities.

---

## 1. System Architecture

The application is structured as a hybrid desktop application combining a FastAPI Python backend for high-performance deep learning inference and an Electron desktop frontend for real-time user interaction and audio visualization.

```
                    ┌────────────────────────┐
                    │  Electron Application  │
                    │      (Desktop UI)      │
                    └───────────┬────────────┘
                                │
        Enumerate Mics          │  HTTP POST /infer (WAV Blobs)
        Render Visualizations   │  HTTP GET /models
        Circular Audio Buffer   │  HTTP POST /set_model
                                ▼
                    ┌────────────────────────┐
                    │    FastAPI Backend     │
                    │   (Python/ONNX Run)    │
                    └───────────┬────────────┘
                                │
                    Loads PyTorch/Models/*.onnx
                    Converts WAV to Mono 16kHz
                    Runs Transform & ONNX Inference
                                ▼
                        Model Predictions
```

* **FastAPI Backend (Port 18000):** Hosts model loading APIs and inference pipelines, executing predictions using ONNX Runtime.
* **Electron Frontend (Port 5173 / Main Process):** Manages the application window, captures audio from the system's microphone, displays real-time canvas visualizations, runs HTML5 games, and routes user commands.

---

## 2. FastAPI Backend (`backend/main.py`)

The backend is built using FastAPI and run via Uvicorn. It is configured to allow CORS requests from all origins to facilitate frontend communication.

### 2.1 API Endpoints
* **`GET /models`**
  Scans `PyTorch/Models/` and `Tensorflow/Models/` directories for `.onnx` files, returning a list of available models and the current active model.
* **`POST /set_model`**
  Dynamically switches the active model file. Receives a JSON request body:
  ```json
  { "model_path": "C:/_school/KeywordDetection/PyTorch/Models/PyTorch.onnx" }
  ```
  It validates the path and reinstantiates the ONNX Runtime session.
* **`POST /infer`**
  Accepts a `.wav` file upload via form data, performs preprocessing, and runs inference. Returns:
  ```json
  {
    "status": "success",
    "keyword": "up",
    "class_index": 2,
    "classes": ["yes", "no", "up", "down", "other"]
  }
  ```

### 2.2 Preprocessing & Feature Extraction Pipeline
When a WAV file is uploaded to `/infer`:
1. **Decoding:** Decoded using `soundfile` into a float32 NumPy array.
2. **Channel Reduction:** Converted from stereo to mono by averaging channels.
3. **Resampling:** If the sample rate differs from the target ($16\text{ kHz}$), a `torchaudio.transforms.Resample` transform is applied.
4. **Length Standardization:** Truncates or zero-pads the audio to exactly $16,000$ samples ($1\text{ second}$).
5. **Feature Transform:** Runs the transform associated with the loaded model and sends the resulting features to the ONNX session.

### 2.3 Dynamic Model Versioning
To support legacy models and newer architectures, the backend dynamically determines which transform to apply based on the model file's input layer dimensions:
* **v1 Models (MFCC):** If the ONNX input layer features dimension is 40, the backend applies `torchaudio.transforms.MFCC` (40 coefficients, 64 mel channels).
* **v2 Models (Log-Mel Spectrogram):** If the input features dimension is 64, it switches to `LogMelSpectrogram` (n_fft=400, hop_length=160, n_mels=64).

---

## 3. Electron Frontend (`frontend/`)

The frontend application is built using vanilla JavaScript and CSS, run via Electron, and compiled/served using Vite.

### 3.1 Window Lifecycle & Custom Model Selector
* **IPC Bridge:** Electron's `main.cjs` and `preload.cjs` expose `window.electronAPI.openFile()`, allowing the frontend to trigger a native Windows file selection dialog.
* **Dynamic Loading:** If a user selects a custom `.onnx` model, its absolute path is appended to the selection dropdown and sent to the backend's `/set_model` endpoint.

### 3.2 Dual Recording Modes

#### Normal Mode (Manual Triggering)
* **Behavior:** The user presses and holds the **Hold to Talk** button to record.
* **Technical Flow:** Audio is captured using a `ScriptProcessorNode`. When the mouse is released, the captured samples are converted to a WAV file blob and sent to the backend's `/infer` endpoint.
* **Timeout Safeguard:** The recording automatically stops after **1.0 second** to keep inputs consistent with model training.

#### Live Mode (Continuous Streaming)
* **Behavior:** Captures audio continuously from the selected microphone device.
* **Technical Flow:** Audio is captured using a `ScriptProcessorNode` and stored in a circular float32 buffer of size $16,000$ ($1\text{ second}$ of audio at $16\text{ kHz}$).
* **Sliding Window Inference:** A recurring timer triggers inference at a configurable interval (typically every $200\text{ ms}$). It reads the last $1\text{ second}$ of samples from the circular buffer, packages them into a WAV blob, and sends them to `/infer`. This overlapping sliding window approach reduces detection latency.

### 3.3 Microphone Selection & Dynamic Hot-Plugging
* **Device Enumeration:** The frontend queries available input hardware via `navigator.mediaDevices.enumerateDevices()` and populates the **Select Microphone** dropdown.
* **Hot-Plugging Support:** To handle microphones being plugged or unplugged at runtime, the application registers a listener on `navigator.mediaDevices` for the `devicechange` event, which automatically calls `populateMicSelect()` to refresh the available options dynamically without reloading the app.
* **Input Transitioning:** Changing the microphone choice automatically terminates the existing `MediaStreamTrack` audio stream (to free hardware resources) and calls `startLive()` or `initAudio()` using the new `deviceId` parameter constraint.

---

## 4. Real-Time Canvas Visualizations

Live Mode features three HTML5 Canvas visualization displays rendered at 60 FPS using `requestAnimationFrame`:

1. **Oscilloscope Waveform Canvas:**
   Displays time-domain audio data captured via an `AnalyserNode` (`getByteTimeDomainData`). The waveform scrolls horizontally from right to left in real-time.
2. **Scrolling Spectrogram Canvas:**
   Displays frequency-domain data using `getByteFrequencyData`. Frequency magnitudes are mapped to grayscale values, creating a scrolling spectrogram that represents pitch and intensity over time.
3. **Detections Timeline Canvas:**
   Displays a scrolling timeline of model predictions. When the backend detects a keyword, a colored block corresponding to the classification is added to the timeline, scrolling left in sync with the waveform and spectrogram.

---

## 5. UI Theme & Styling

The user interface features a clean, responsive black-and-white theme with matte pastel colors for active keywords.

### 5.1 CSS Color Palette
The CSS styling is defined in [style.css](file:///c:/_school/KeywordDetection/frontend/src/style.css):
* **Grayscale Base:**
  * `--bg-base`: `#000000` (Main background)
  * `--bg-sidebar`: `#050505` (Sidebar background)
  * `--bg-panel`: `#0a0a0a` (Container panels)
  * `--bg-card`: `#101010` (Game cards)
  * `--border-color`: `#222222` (Borders)
  * `--text-main`: `#ffffff` (Text color)
  * `--text-muted`: `#888888` (Muted details)
* **Keyword Color Indicators:**
  * `YES` (Mint Green): `#97DCAE` / `--color-yes`
  * `NO` (Soft Coral): `#DC8D82` / `--color-no`
  * `UP` (Soft Blue): `#8BB8D7` / `--color-up`
  * `DOWN` (Slate Blue): `#8290BA` / `--color-down`
  * `OTHER` (Pastel Yellow): `#F6E7B9` / `--color-other`

### 5.2 Animations & Indicators
When a keyword is detected in manual or live mode, the UI adds a corresponding `.detected-<keyword>` class. This applies the keyword's color indicator and triggers a CSS scale-and-pulsate animation.

---

## 6. Voice Arcade (Games)

The **Voice Arcade** contains five HTML5 Canvas games designed to be controlled using voice commands.

```
                    ┌────────────────────────────┐
                    │    Voice Arcade Router     │
                    │   (games.js: activeGame)   │
                    └─────────────┬──────────────┘
                                  │
      ┌───────────────┬───────────┼───────────────┬──────────────┐
      ▼               ▼           ▼               ▼              ▼
┌───────────┐   ┌───────────┐┌───────────┐  ┌───────────┐  ┌───────────┐
│Flappy Bird│   │Grid Runner││Space Def. │  │Simon Mem. │  │Cyber HiLo │
└───────────┘   └───────────┘└───────────┘  └───────────┘  └───────────┘
   "UP"         "UP" / "DOWN"  "YES" / "NO"  "UP" / "DOWN"  "UP" / "DOWN"
   Jumps        "YES" / "NO"   "UP" / "DOWN" "YES" / "NO"   Guesses
```

### 6.1 Flappy Bird
* **Command:** `UP`
* **Mechanics:** The player controls a bird flying through gaps in moving pipes. Saying `UP` applies upward velocity to jump.
* **Optimizations:** The gravity and jump parameters were adjusted for voice control latency, and a speed slider controls pipe speed.

### 6.2 Grid Runner
* **Commands:** `UP` (Up), `DOWN` (Down), `YES` (Right), `NO` (Left)
* **Mechanics:** The player navigates a 15x10 grid to collect green gems while avoiding red enemy blocks that move at regular intervals.

### 6.3 Space Defender
* **Commands:** `YES` (Move Right), `NO` (Move Left), `UP` (Shoot Laser), `DOWN` (Activate Shield)
* **Mechanics:** The player moves a ship left or right to shoot down incoming invaders. Saying `DOWN` activates a protective shield for 1.6 seconds if the cooldown has elapsed. Invader dimensions are scaled up, and maximum concurrent spawns are limited to maintain readability.

### 6.4 Simon Memory (Turn-based)
* **Commands:** `UP`, `DOWN`, `YES`, `NO`
* **Mechanics:** A classic sequence repeating game. The game displays a sequence of directions, and the player must repeat it back in order using voice commands.

### 6.5 Cyber Hi-Lo (Turn-based)
* **Commands:** `UP` (Higher), `DOWN` (Lower)
* **Mechanics:** A card-guessing game. The game displays a card, and the player guesses if the next card will be higher (`UP`) or lower (`DOWN`).

### 6.6 Game Speed & Processing Interval Controls
To accommodate model inference latency, two controls are included:
* **Inference Window Interval Slider:** Controls how frequently the sliding window sends data to the backend (from $100\text{ ms}$ to $1000\text{ ms}$).
* **Game Speed Slider:** Modifies the tick rates, gravity forces, and translation values in `games.js` via `window.gameSpeedMultiplier` to adjust game speed based on vocal response times.

### 6.7 Real-Time Retro Audio Sound Effects Synthesizer
* **Web Audio Synthesis:** Rather than loading heavy external audio assets (MP3 or WAV files), the games feature a built-in retro synthesizer in [games.js](file:///c:/_school/KeywordDetection/frontend/src/games.js) that generates sound effects in real-time.
* **Oscillators & Envelopes:** The sound synthesis uses the Web Audio API's `OscillatorNode` (sine, square, triangle, and sawtooth types) and `GainNode` to construct custom pitch sweep envelopes:
  * *Flap:* Exponential sine sweep from 150 Hz to 380 Hz over 0.1s.
  * *Score:* A dual-note arpeggio (C5 to E5) using a triangle wave over 0.22s.
  * *Hit:* A descending sawtooth sweep from 120 Hz to 30 Hz over 0.35s.
  * *Laser:* A fast descending sawtooth sweep from 700 Hz to 150 Hz over 0.12s.
  * *Shield:* A frequency modulation loop (250 Hz -> 320 Hz -> 250 Hz) using a sine wave over 0.3s.
  * *Shield Blocked:* A harsh dual-step square wave (280 Hz to 220 Hz) over 0.15s.

---

## 7. Startup & Stop Scripts

### 7.1 Application Startup (`start.bat`)
Launches the application services:
1. **Inference Server:** Starts FastAPI in a command window titled `KeywordDetectionBackend` using the virtual environment interpreter:
   `start "KeywordDetectionBackend" cmd /c "cd /d "%~dp0backend" && "..\.venv\Scripts\python.exe" main.py"`
2. **Launch Delay:** Pauses for 3 seconds (`ping 127.0.0.1 -n 3`) to allow the FastAPI server to initialize and load the model.
3. **Frontend Host:** Launches Electron in a command window titled `KeywordDetectionFrontend`:
   `start "KeywordDetectionFrontend" cmd /c "cd /d "%~dp0frontend" && npm start"`

### 7.2 Application Shutdown (`stop.bat`)
Gracefully terminates services:
1. **Targeted Terminations:** Finds and terminates windows titled `KeywordDetectionBackend*` and `KeywordDetectionFrontend*`.
2. **Failsafe Port Clearance:** Queries for processes listening on port 18000 and terminates them by PID to prevent port lockouts:
   ```batch
   for /f "tokens=5" %%a in ('netstat -aon ^| findstr :18000') do (
       taskkill /F /PID %%a > nul 2>&1
   )
   ```

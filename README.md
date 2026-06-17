# Keyword Detector

A real-time deep learning speech recognition application that detects specific spoken commands (`yes`, `no`, `up`, `down`) and uses them to control interactive retro games (Voice Arcade) or perform model comparisons. 

Optimized to over **98.5% validation accuracy** utilizing PyTorch, ONNX, Electron, and FastAPI.

---

## Repository Branches

The repository is structured into the following branches to manage releases, development, and different framework implementations:

* **`main`**: Contains stable releases.
* **`develop`**: The primary development branch, branching off from `main`.
* **`PyTorch`**: The PyTorch implementation of the keyword detector (including model training, optimization, and progressive test suites), branching off from `develop`.
* **`TensorFlow`**: The TensorFlow/Keras implementation of the keyword detector, branching off from `develop`.

---

## Complete Documentation

The complete project documentation, installation instructions, system architecture, project logs, and lessons learned are compiled in a single LaTeX document, alongside a standalone markdown development log:
*   **[Documentation.tex](file:///c:/Users/nikna/Desktop/KeywordDetection/Documentation/Documentation.tex)**
*   **[Project_Log.md](file:///c:/Users/nikna/Desktop/KeywordDetection/Documentation/Project_Log.md)**

---

<details>
<summary>Project Structure</summary>

Here is the visual structure of the project, focusing on the machine learning models and dataset:

```text
.
├── config.json                     # Global configuration (classes, sample rates)
├── start.bat                       # Launch script for both services (Frontend + Backend)
├── stop.bat                        # Shutdown script to terminate running processes
│
├── PyTorch/                        # PyTorch model training and optimization
│   ├── PyTorch.ipynb               # Model training & ONNX export notebook
│   ├── Models/                     # Exported PyTorch ONNX model
│   └── Testing/                    # Progressive test suites (01_specaugment to 11_combined_stable)
│       └── Results/                # Training metrics & evaluation logs
│
├── TensorFlow/                     # TensorFlow/Keras alternate model implementations
│   ├── tensorflow.ipynb            # Model training & ONNX export notebook
│   └── Models/                     # Exported TensorFlow ONNX model
│
├── backend/                        # FastAPI REST API serving model inference via ONNX Runtime
│   └── main.py                     # Main server entrypoint (Port 18000)
│
├── frontend/                       # Electron desktop application
│   ├── index.html                  # Core HTML5 layout & Canvas viewports
│   ├── main.cjs                    # Electron main controller process
│   └── src/
│       ├── main.js                 # Circular audio buffer capture & visualizer
│       ├── games.js                # Retro Voice Arcade game engines (Flappy Bird, Space Defender, etc.)
│       └── style.css               # Glassmorphic UI styling
│
├── install/                        # Dataset installers and guide
│   ├── Download_Dataset.py         # Automatic Kaggle dataset fetcher & slicer
│   └── pytorch/                    # Python environment requirements
│
└── Utils/                          # Helper scripts for dataset analysis and cleanup
    ├── analyze_wavs.py             # Script to extract audio duration and sample statistics
    └── dataset_statistics.txt      # Distribution statistics of audio files
```

---

## Quick Start

1.  **System Requirements:** Install [Node.js](https://nodejs.org/), [Python 3.10+](https://www.python.org/) and [FFmpeg](https://www.ffmpeg.org/) (ensure FFmpeg is added to your environment `PATH`).
2.  **Install & Setup:**
    *   Create a virtual environment: `python -m venv .venv`
    *   Install Python packages: `.\.venv\Scripts\pip install -r install/pytorch/pytorch-requirements.txt`
    *   Install Node packages: Run `npm install` inside the `frontend/` directory.
    *   Fetch and prepare dataset: `.\.venv\Scripts\python install/Download_Dataset.py`
3.  **Run Application:**
    *   Double-click `start.bat` in the root folder to start both the Python Backend and Electron Frontend.
    *   Double-click `stop.bat` to terminate all services cleanly.

---

## Advanced Usage (Scripts)

### Analyze Dataset statistics
```bash
.\.venv\Scripts\python Utils/analyze_wavs.py
```

### Run Model Test Suites
Execute all progressive model architectures to compare accuracies (output logged to `PyTorch/Testing/Results/Results.txt`):
```bash
.\.venv\Scripts\python PyTorch/Testing/run_all.py
```

Run a specific architecture:
```bash
.\.venv\Scripts\python PyTorch/Testing/run_all.py 11_combined_stable.py
```

---

## Repository
GitHub remote origin: [https://github.com/KOFiblto/KeywordDetection](https://github.com/KOFiblto/KeywordDetection)

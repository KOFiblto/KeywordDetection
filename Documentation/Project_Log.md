# Project Log & Commit Chronology

This document provides a chronological overview of the development phases, associated Git commits, and key technical solutions implemented during the lifecycle of the Keyword Detection System.

| Commits | Author(s) | Phase / Milestone | Key Changes & Technical Solutions |
| :--- | :--- | :--- | :--- |
| `10ff1c0`, `843bee5` | Dominik Praja | **PyQt6 Dataset Cleanup** | • Sourced the raw Google Speech Commands dataset and cleaned it using a custom-built GUI tool (`Wav_File_Cleanup.py`) which computes the RMS signal energy of WAV files, enabling developers to quickly filter, reclassify, or delete corrupt, empty, or silent audio files via keyboard shortcuts. |
| `6b7c505`, `8f67e95` | Mathias Kornschober | **Dataset & I/O Tuning** | • Resolved massive class imbalance by downsampling the negative class (`other`) to 10,000 files and oversampling target keywords (`yes`, `no`, `up`, `down`) to 8,000 files in the training split.<br>• Fixed severe CPU/disk bottleneck (which kept the GPU starved and idle ~70% of the time) by implementing a RAM-caching mechanism in the PyTorch Dataset loader, reducing epoch times from minutes to under 5 seconds. |
| `f285a57`, `7ae26e2` | Mathias Kornschober | **ONNX & GUI Integration** | • Decoupled production inference from heavy deep learning framework runtimes by compiling PyTorch models to ONNX format (CPU-inference latency $<5$ ms).<br>• Designed and built an Electron desktop app with 60 FPS Canvas-based waveform, scrolling spectrogram, and detection timeline visualizations. |
| `db035fa`, `188f706` | Mathias Kornschober | **Post-Inference Filters** | • Integrated energy-based Voice Activity Detection (VAD) (RMS energy threshold $<0.002$ overrides prediction to `other`).<br>• Implemented a confidence threshold filter (softmax probability $<0.85$ overrides prediction to `other`), decreasing false positives from ambient room noise/breathing by $>50$\% in Live Mode. |
| `cb0e16a`, `950bc6e` | Dominik Praja | **TensorFlow Alignment** | • Refactored the Keras pipeline to extract MFCC features matching the PyTorch model's input dimension of $(40, 81, 1)$.<br>• Reorganized the repository structure and consolidated all ONNX export models into dedicated subfolders. |
| `359ad29`, `0424c58` | Mathias Kornschober | **Dual Mode & JS Synthesizer** | • Integrated a comparative `DualLiveMode` in the GUI allowing real-time side-by-side inference comparing PyTorch and TensorFlow models on the same microphone stream.<br>• Designed a Web Audio API-based retro synthesizer directly in JavaScript (`games.js`) to procedurally generate sound effects (Flap, Score, Laser, Shield) without loading external audio assets. |
| `d4fbbc4` | Mathias Kornschober | **Service Consolidation** | • Consolidated local development services by introducing an integrated Node.js startup script (`start-services.js`) to concurrently launch the FastAPI Python backend, the Vite dev server, and Electron, while automatically opening a browser window and cleaning up background processes on exit.<br>• Resolved CSS styling issues to ensure correct mobile responsive scrolling in the UI. |
| `09fe383` to `fa83120` | Dominik Praja | **Keras Tuning & Retraining** | • Retrained the TensorFlow/Keras model over 100 epochs, embedding GPU-based data augmentations (time-shift, noise) and SpecAugment directly inside the model graph.<br>• Ported VAD and confidence evaluation filters to TensorFlow, yielding a final test set accuracy of 97.46%. |

***

### Summary of Major Improvements
* **Accuracy:** Reached **98.52%** test set accuracy on PyTorch and **97.46%** on TensorFlow.
* **Latency:** Low-latency inference ($<5$ ms) with continuous 200 ms sliding window capture.
* **Robustness:** Ambient noise false triggers reduced by over 50% using VAD and confidence limits.
* **Liveliness:** Retro sound synthesizers and Canvas renderers operate natively at 60 FPS in Electron.

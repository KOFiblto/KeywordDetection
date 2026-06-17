# Project Log & Commit Chronology

This document provides a chronological overview of the development phases, associated Git commits, and key technical solutions implemented during the lifecycle of the Keyword Detection System.

| Commits | Author(s) | Phase / Milestone | Key Changes & Technical Solutions |
| :--- | :--- | :--- | :--- |
| `ac1a563` | Mathias Kornschober | **Dataset Directory Re-ordering** | • Re-ordered directories for improved raw asset file structure. |
| `2bd9488` | Mathias Kornschober | **Dataset Ingestion Script** | • Sourced Speech Commands dataset and wrote download verification tools. |
| `459ab8b` | Mathias Kornschober | **Initial PyTorch Baseline** | • Implemented the baseline PyTorch training pipeline (`Main.ipynb`) and defined initial model requirements. |
| `d2e3d2f` | Mathias Kornschober | **Developer Installation Guide** | • Wrote the first version of the dependencies and setup guides (`InstallGuide.md`). |
| `7894459` | Dominik Praja | **Initial TensorFlow Prototype** | • Setup the initial sequential prototype script (`main.py`) for TensorFlow audio ingestion. |
| `630b632` | Dominik Praja | **Keras Audio Ingestion** | • Refactored feature extraction to use standard wave headers directly, eliminating dependencies on librosa. |
| `f70c693` | Dominik Praja | **GPU Status Checks** | • Added diagnostics (`check_gpu_status`) to verify CUDA GPU acceleration within TensorFlow. |
| `bb644c3` | Dominik Praja | **WAV Metadata Verification** | • Programmed checks (`check_wav_specs_tf`) to enforce sampling rates and clip lengths. |
| `d5180b0` | Dominik Praja | **Centralized Parameter Config** | • Created `config.json` containing sample rates and model dimensions to unify Python scripts. |
| `9b7c558` | Dominik Praja | **Relative Path Configurations** | • Patched absolute path limitations, making setups relative and cross-platform compatible. |
| `244d7ed` | Dominik Praja | **Keras Sequential Training** | • Programmed the core sequential training loop in TensorFlow, structuring basic evaluation steps. |
| `5b4cefa` | Dominik Praja | **Dynamic Spectrogram Display** | • Mapped training parameters in `config.json` to synchronize real-time spectrogram plots. |
| `9e2ea55` | Dominik Praja | **Keras Stratified Corpus Split** | • Divided data into 70/10/20 train/val/test splits to verify Keras evaluations. |
| `e76d261` | Dominik Praja | **Keras Early Stopping** | • Added `EarlyStopping` callbacks monitoring validation losses to prevent overfitting. |
| `58a3f6d` | Dominik Praja | **Keras Prediction Formatters** | • Developed formatted console tables for displaying validation metrics and test predictions. |
| `343a5a2` | Mathias Kornschober | **SpecAugment Frequency Masking** | • Added time/frequency masking to PyTorch dataset loader to reduce model overfitting. |
| `4d371c0` | Mathias Kornschober | **PyTorch Scheduler Config** | • Configured `ReduceLROnPlateau` and checkpoint saving logic to optimize learning rate decays. |
| `11bb378` | Mathias Kornschober | **Cross-Validation Framework** | • Implemented K-Fold evaluation metrics for PyTorch to assess generalization performance. |
| `10ff1c0` | Dominik Praja | **PyQt6 Audio Reviewer GUI** | • Developed PyQt6 tool (`Wav_File_Cleanup.py`) to visualize audio waveforms and compute RMS energy. |
| `843bee5` | Dominik Praja | **Dataset Sorting Controls** | • Created key bindings to filter corrupted and silent wave files inside the reviewer GUI. |
| `fb0b63e` | Dominik Praja | **Cleaned "Yes" Directory** | • Cleaned the `yes` keyword audio folder using the PyQt6 reviewer GUI. |
| `4fec2c6` | Dominik Praja | **Cleaned "No" Directory** | • Cleaned the `no` keyword audio folder. |
| `16fba2f` | Dominik Praja | **Cleaned "Down" Directory** | • Cleaned the `down` keyword audio folder. |
| `aa7822f` | Dominik Praja | **Directory Clean-up & Organization** | • Re-organized directory structure to ensure balanced classes. |
| `6b7c505, 8f67e95` | Mathias Kornschober | **Dataset Imbalance & RAM-Caching** | • Balanced classes to 10k negatives and 8k keywords, and implemented RAM caching in PyTorch. |
| `f285a57` | Mathias Kornschober | **PyTorch ONNX Export** | • Serialized PyTorch weights to ONNX format, reducing CPU latency to under 5 ms. |
| `7ae26e2` | Mathias Kornschober | **Electron UI & Waveforms** | • Created an Electron app shell using Vite to render 60 FPS real-time Canvas waveforms. |
| `8f67e95` | Mathias Kornschober | **PyTorch Stratified Splitting** | • Refactored dataset splits to guarantee zero validation leakage between training folds. |
| `db035fa` | Mathias Kornschober | **Post-Inference VAD Gate** | • Integrated energy-based RMS filters to bypass input signals below $0.002$. |
| `188f706` | Mathias Kornschober | **Confidence Threshold Gate** | • Added a confidence threshold requiring softmax probabilities $>0.85$ to drop ambient noise false alarms. |
| `cb0e16a, 950bc6e` | Dominik Praja | **TensorFlow Alignment & MFCC** | • Aligned Keras preprocessing to extract $(40, 81, 1)$ MFCC features to match PyTorch shapes. |
| `359ad29, 0424c58` | Mathias Kornschober | **Dual Live Mode & JS Synthesizer** | • Integrated comparative real-time GUI inference and programmed a Web Audio synth. |
| `d4fbbc4` | Mathias Kornschober | **Consolidated Services Script** | • Created `start-services.js` to concurrently spawn the FastAPI backend, Vite dev server, and Electron. |
| `09fe383` | Dominik Praja | **Negative Class Balancing** | • Retrained Keras model to include 'other' negative class, improving open-domain noise robustness. |
| `0e01959` | Dominik Praja | **Keras SpecAugment Graph** | • Embedded time-shifting and SpecAugment transformations directly inside Keras model graph. |
| `e53f2ff` | Dominik Praja | **Keras VAD & Filters Port** | • Integrated VAD filters and confidence limits into TensorFlow to maintain parity. |
| `fa83120` | Dominik Praja | **Final Keras Fit (100 Epochs)** | • Trained Keras model over 100 epochs, reaching a final test accuracy of 97.46%. |
| `edf80e8, 793996d, 3f9d7f9` | Mathias Kornschober | **PyTorch Final Evaluation** | • Reported 98.52% PyTorch test accuracy and compiled the PyTorch final presentation slide deck. |
| `9fd7095, aa6f963, 3cd9709` | Mathias Kornschober, Dominik Praja | **Model Evaluation & Presentation** | • Co-authored final model comparison presentations and TensorFlow slide deck. |
| `f0424bb, 05a3038` | Mathias Kornschober | **Standalone Installers** | • Structured electron-builder configurations to compile standalone Windows application packages. |
| `81dc40b` | Mathias Kornschober | **Client-Side WASM Setup** | • Migrated backend deep learning dependencies to browser-based WASM execution. |
| `3b6ef32` | Mathias Kornschober | **DSP Performance Optimization** | • Optimized client-side STFT and DCT-II algorithms using cached twiddle tables. |
| `416acf4` | Mathias Kornschober | **Active MediaStream Re-use** | • Prevented duplicate mobile microphone permission requests by caching stream states. |
| `32b03b2` | Mathias Kornschober | **Capacitor WebView Loop Fix** | • Forced runtime providers in capacitor context to bypass WebView restarts. |
| `7032bc4, d9a01ed` | Mathias Kornschober | **CI/CD & iOS Deployment** | • Configured GitHub Actions pipelines to compile, package, and archive unsigned iOS application bundles. |
| `8109e56, f3b9df5, 83e58e0, 4fd8943` | Mathias Kornschober, Dominik Praja | **Workflow Alignment** | • Merged develop branches, isolated requirement specs, and documented release workflows. |

***

### Summary of Major Improvements
* **Accuracy:** Reached **98.52%** test set accuracy on PyTorch and **97.46%** on TensorFlow.
* **Latency:** Low-latency inference ($<5$ ms) with continuous 200 ms sliding window capture.
* **Robustness:** Ambient noise false triggers reduced by over 50% using VAD and confidence limits.
* **Liveliness:** Retro sound synthesizers and Canvas renderers operate natively at 60 FPS in Electron.

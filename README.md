# Keyword Detector

A PyTorch-based neural network designed to detect specific spoken commands within audio files. Recently optimized for maximum accuracy and stability, reaching over ~98.5% validation accuracy.

## Supported Keywords

The model classifies audio into one of the following categories:
* yes
* no
* up
* down

## Project Structure

```text
.
├─ dataset/
│   ├─ yes/
│   │   └─ *.wav
│   ├─ no/
│   │   └─ *.wav
│   ├─ up/
│   │   └─ *.wav
│   └─ down/
│       └─ *.wav
└─ src/
    ├─ main.py
    ├─ main.ipynb
    └─ _Antigravity/
        ├─ Analyze Wave Files/    # Dataset statistics and distribution plots
        │   ├─ analyze_wavs.py
        │   ├─ dataset_distribution.png
        │   └─ dataset_statistics.txt
        └─ Testing/               # Progressive model enhancements & final architectures
            ├─ 01_specaugment.py  ...  10_combined_best.py
            ├─ 11_combined_stable.py    # Master model (MFCC, SpecAugment, DA, LR Scheduler)
            ├─ 12_confusion_matrix.py   # Generates heatmaps
            ├─ 13_kfold_cross_validation.py # Evaluates strictly using 5-Fold CV
            ├─ run_all.py         # Batch evaluator runner
            └─ Results/           # Final evaluation logs, checkpoints, and figures
```

## Features and Improvements

The core PyTorch model includes the following deep learning optimizations dynamically tested in the `Testing` directory:
- **Audio Data Augmentation:** Random time-shifting and noise injection.
- **SpecAugment:** `FrequencyMasking` and `TimeMasking` on the raw spectrogram transforms to construct robust features.
- **MFCC:** Replacing standard MelSpectrograms with 40-channel MFCC generation for isolated vocal context.
- **Network Depth & Capacity:** 4-layer CNN with expanding filter capacity (16 -> 128) and aggressive dropout.
- **Optimization Strategy:** Adam optimizer coupled with `ReduceLROnPlateau` and intelligent model checkpointing.

## Requirements

Currently built and optimized inside a local virtual environment (`.venv`) utilizing PyTorch, torchaudio, scikit-learn, soundfile, matplotlib, and seaborn.

## Usage

### Analyzing Dataset
Run the data visualization script to measure audio clip lengths across all classes:
```bash
python "src/_Antigravity/Analyze Wave Files/analyze_wavs.py"
```

### Running Model Tests
You can execute and log all progressively built tests automatically via the `run_all.py` suite. All logs output directly to `src/_Antigravity/Testing/Results/Results.txt`:
```bash
python "src/_Antigravity/Testing/run_all.py"
```

To run a specific architecture standalone through the batch runner:
```bash
python "src/_Antigravity/Testing/run_all.py" 11_combined_stable.py
```

### Advanced Evaluation
Generate the definitive testing confusion matrix or compute true variance using K-Fold Validation manually:
```bash
python "src/_Antigravity/Testing/12_confusion_matrix.py"
python "src/_Antigravity/Testing/13_kfold_cross_validation.py"
```

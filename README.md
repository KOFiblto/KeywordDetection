# Keyword Detector

A PyTorch-based neural network designed to detect specific spoken commands within audio files.

## Supported Keywords

The model classifies audio into one of the following categories:
* yes
* no
* up
* down

## Project Structure

```text
.
├─ datasets/
│   └─ speech_dataset/
│       ├─ yes/
│       │   └─ *.wav
│       ├─ no/
│       │   └─ *.wav
│       ├─ up/
│       │   └─ *.wav
│       └─ down/
│           └─ *.wav
└─ src/
    └─ *.ipynb
```

## Requirements

* Python 3.8+
* PyTorch
* Torchaudio
* Jupyter Notebook

## Installation

Install the required dependencies using pip:

```bash
pip install torch torchaudio jupyter
```

## Usage

Start the Jupyter server to access and run the notebooks located in the `src/` directory:

```bash
jupyter notebook
```
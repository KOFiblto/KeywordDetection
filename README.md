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


## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

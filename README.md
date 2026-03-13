# Keyword Detector

A neural network project designed to detect specific spoken commands within audio files. The model classifies audio into one of the following categories:
* yes
* no
* up
* down

These will very likely be expanded in the future, but to shorten model-training time we only used 4 as of now.

## Repository Branches

The repository is structured into the following branches to manage releases, development, and different framework implementations:

* **`main`**: Contains stable releases.
* **`develop`**: The primary development branch, branching off from `main`.
* **`Pytorch`**: The PyTorch implementation of the keyword detector, branching off from `develop`.
* **`Dominik`**: The TensorFlow implementation, branching off from `develop` (Note: This branch is scheduled to be renamed to `Tensorflow`).

## Requirements

*(Framework-specific requirements are managed within the respective implementation branches).*

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

*(Usage details depend on the specific implementation branch).*

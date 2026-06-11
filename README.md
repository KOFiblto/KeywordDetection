# Keyword Detector

A deep learning project designed to recognize specific spoken command keywords (**yes**, **no**, **up**, **down**) and filter out background noise or other words. This system provides high-performance models implemented in both **PyTorch** and **TensorFlow**, ready to compile, export, and use.

This repository contains isolated keyword spotters trained on the Google Speech Commands dataset. The models classify 1-second audio recordings into one of five classes: `yes`, `no`, `up`, `down`, and `other` (representing any other word or background noise). The project is optimized to reach over ~98.5% validation accuracy using Mel-Frequency Cepstral Coefficients (MFCC) feature extraction, audio data augmentation (time-shifting, noise injection), and SpecAugment (frequency and time masking). Both training pipelines export models directly to the ONNX format for efficient deployment.

---

<details>
<summary>Project Structure</summary>

Here is the visual structure of the project, focusing on the machine learning models and dataset:

```text
.
├── dataset/
│   ├── yes/
│   ├── no/
│   ├── up/
│   ├── down/
│   └── other/
├── install/
│   ├── Download_Dataset.bat
│   ├── install_requirements.bat
│   └── pytorch/
│       └── pytorch-requirements.txt
├── PyTorch/
│   ├── PyTorch.ipynb
│   ├── Models/
│   │   └── PyTorch.onnx
│   └── Testing/
├── TensorFlow/
│   ├── tensorflow.ipynb
│   └── Models/
│       └── TensorFlow.onnx
├── start.bat
└── stop.bat
```

* **[PyTorch](file:///c:/_school/KeywordDetection/PyTorch)**: Contains the training notebook [PyTorch/PyTorch.ipynb](file:///c:/_school/KeywordDetection/PyTorch/PyTorch.ipynb), model exports in [PyTorch/Models](file:///c:/_school/KeywordDetection/PyTorch/Models), and progressive enhancement tests in [PyTorch/Testing](file:///c:/_school/KeywordDetection/PyTorch/Testing).
* **[TensorFlow](file:///c:/_school/KeywordDetection/TensorFlow)**: Contains the training notebook [TensorFlow/tensorflow.ipynb](file:///c:/_school/KeywordDetection/TensorFlow/tensorflow.ipynb) and export binaries in [TensorFlow/Models](file:///c:/_school/KeywordDetection/TensorFlow/Models).
* **[dataset](file:///c:/_school/KeywordDetection/dataset)**: The target directory where audio commands are structured after extraction.
* **[install](file:///c:/_school/KeywordDetection/install)**: Shell scripts and dependency definitions.
</details>

<details>
<summary>Installation and Setup</summary>

### 1. Install Dependencies
Run the install batch script to set up Python packages like PyTorch, torchaudio, scikit-learn, and soundfile:

```cmd
.\install\install_requirements.bat
```

*Note: For audio streaming in the local web application, make sure FFmpeg is installed and added to your system path.*

### 2. Download and Restructure Dataset
To download, extract, and clean the Google Speech Commands dataset automatically, run the dataset utility:

```cmd
.\install\Download_Dataset.bat
```

The script will prompt you for two source choices:
* **Option 1 (Local ZIP)**: Select a pre-downloaded dataset ZIP file on your machine.
* **Option 2 (Kaggle API)**: Enter your Kaggle credentials to download the ~1.4GB dataset automatically.

Once downloaded, the utility restructures the folders to retain the target keywords (`yes`, `no`, `up`, `down`), slices background noise into 1-second WAV files, and groups all other vocabulary categories into the `other` directory.
</details>

<details>
<summary>Usage</summary>

### Training Models
To train the neural networks from scratch, open and run the interactive cells in the respective training notebooks:
* For PyTorch: [PyTorch/PyTorch.ipynb](file:///c:/_school/KeywordDetection/PyTorch/PyTorch.ipynb)
* For TensorFlow: [TensorFlow/tensorflow.ipynb](file:///c:/_school/KeywordDetection/TensorFlow/tensorflow.ipynb)

### Running the Application
The project includes a local backend inference service and frontend GUI. To start the application, execute the start script:

```cmd
.\start.bat
```

To stop all active application processes, run:

```cmd
.\stop.bat
```
</details>

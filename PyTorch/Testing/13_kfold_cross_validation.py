import os
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import soundfile as sf
import time
import numpy as np

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataset"))
import json
def load_keywords():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        p = os.path.join(d, "config.json")
        if os.path.exists(p):
            with open(p, "r") as f: return json.load(f)["keywords"]
        d = os.path.dirname(d)
    return ["yes", "no", "up", "down"]

CLASSES = get_keywords()
TARGET_SAMPLE_RATE = get_config_value('target_sample_rate', 16000)
NUM_SAMPLES = get_config_value('num_samples', 16000)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Preparation
def get_files_and_labels():
    file_paths = []
    labels = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}

    for cls_name in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for file in os.listdir(cls_dir):
            if file.endswith(".wav"):
                file_paths.append(os.path.join(cls_dir, file))
                labels.append(class_to_idx[cls_name])

    return np.array(file_paths), np.array(labels)

class KeywordDataset(Dataset):
    def __init__(self, paths, labels, is_training=False):
        self.paths = paths
        self.labels = labels
        self.is_training = is_training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clean_path = os.path.normpath(self.paths[idx])
        waveform_np, sr = sf.read(clean_path, dtype="float32")
        waveform = torch.from_numpy(waveform_np)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.is_training:
            shift = random.randint(-1600, 1600)
            waveform = torch.roll(waveform, shift, dims=-1)

        if waveform.shape[1] > NUM_SAMPLES:
            waveform = waveform[:, :NUM_SAMPLES]
        elif waveform.shape[1] < NUM_SAMPLES:
            waveform = F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))

        return waveform, torch.tensor(self.labels[idx], dtype=torch.long)

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

def train_kfold():
    file_paths, labels = get_files_and_labels()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    print(f"Starting 5-Fold Cross Validation on {DEVICE}...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n--- Fold {fold + 1} ---")
        
        train_paths, val_paths = file_paths[train_idx], file_paths[test_idx]
        train_labels, val_labels = labels[train_idx], labels[test_idx]

        train_loader = DataLoader(KeywordDataset(train_paths, train_labels, is_training=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(KeywordDataset(val_paths, val_labels, is_training=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        model = KeywordCNN(num_classes=len(CLASSES)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SAMPLE_RATE, n_mfcc=40, melkwargs={"n_mels": 64}
        ).to(DEVICE)
        
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10).to(DEVICE)
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20).to(DEVICE)

        best_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                with torch.no_grad():
                    inputs = mfcc_transform(inputs)
                    inputs = freq_masking(inputs)
                    inputs = time_masking(inputs)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total

            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    inputs = mfcc_transform(inputs)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = 100. * test_correct / test_total

            if test_acc > best_acc:
                best_acc = test_acc

            scheduler.step(test_acc)
            
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
               print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Acc: {train_acc:.2f}%, Val Acc: {test_acc:.2f}%")

        print(f"Best Test Accuracy for Fold {fold + 1}: {best_acc:.2f}%")
        fold_accuracies.append(best_acc)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print("\n==============================")
    print("5-Fold Cross Validation Results:")
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Deviation: {std_acc:.2f}%")
    print("==============================")

if __name__ == '__main__':
    train_kfold()

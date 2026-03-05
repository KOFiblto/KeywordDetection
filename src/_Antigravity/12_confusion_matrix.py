import os
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import soundfile as sf
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Configuration
DATA_DIR = "../dataset/"
CLASSES = ["yes", "no", "up", "down"]
TARGET_SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
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

    return file_paths, labels

file_paths, labels = get_files_and_labels()
train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

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

def train_model():
    train_loader = DataLoader(KeywordDataset(train_paths, train_labels, is_training=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(KeywordDataset(test_paths, test_labels, is_training=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

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

    print(f"Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        train_acc = 100. * correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                inputs = mfcc_transform(inputs)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100. * test_correct / test_total
        dur = time.time() - start_time

        print(f"=== Epoch {epoch+1}/{EPOCHS} [{dur:.2f}s] Summary | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% ===")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_keyword_model.pth")
            print(f"--> Saved new best model with accuracy: {best_acc:.2f}%")

        scheduler.step(test_acc)
        print()

    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load("best_keyword_model.pth"))

    # Confusion Matrix Evaluation
    print("Evaluating best model to generate confusion matrix...")
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = mfcc_transform(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == '__main__':
    train_model()

from pathlib import Path
import json
import struct
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "archive (2) (1)"
MODEL_PATH = ROOT / "mnist_model.pth"
METRICS_PATH = ROOT / "metrics.json"
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
SEED = 42

def read_idx_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != count * rows * cols:
        raise ValueError(f"Unexpected image data size in {path}")
    return data.reshape(count, rows, cols)

def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Unexpected label data size in {path}")
    return labels

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x_train = read_idx_images(DATA / "train-images.idx3-ubyte")
    y_train = read_idx_labels(DATA / "train-labels.idx1-ubyte")
    x_test = read_idx_images(DATA / "t10k-images.idx3-ubyte")
    y_test = read_idx_labels(DATA / "t10k-labels.idx1-ubyte")

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=512, shuffle=False
    )

    model = MNISTMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {running_loss / total:.4f} | "
            f"training accuracy: {100 * correct / total:.2f}%"
        )

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            predictions = probabilities.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = 100.0 * test_correct / test_total
    model_cpu = model.to("cpu")
    torch.save({
        "model_state_dict": model_cpu.state_dict(),
        "architecture": "784-256-128-10",
        "class_names": [str(i) for i in range(10)]
    }, MODEL_PATH)

    metrics = {
        "dataset": "MNIST",
        "training_images": int(len(y_train)),
        "testing_images": int(len(y_test)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "training_accuracy_last_epoch_percent": round(100 * correct / total, 2),
        "testing_accuracy_percent": round(test_accuracy, 2),
        "model_file": MODEL_PATH.name
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nFinal test accuracy: {test_accuracy:.2f}%")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()


'''PS C:\Users\bhara\OneDrive\Desktop\AIML-PY\DeepLearning_test> python train_model.py
Device: cpu
Epoch 1/5 | loss: 0.3530 | training accuracy: 90.04%
Epoch 2/5 | loss: 0.1334 | training accuracy: 96.00%
Epoch 3/5 | loss: 0.0889 | training accuracy: 97.37%
Epoch 4/5 | loss: 0.0643 | training accuracy: 98.03%
Epoch 5/5 | loss: 0.0487 | training accuracy: 98.48%

Final test accuracy: 97.56%
Saved model: C:\Users\bhara\OneDrive\Desktop\AIML-PY\DeepLearning_test\mnist_model.pth
Saved metrics: C:\Users\bhara\OneDrive\Desktop\AIML-PY\DeepLearning_test\metrics.json
PS C:\Users\bhara\OneDrive\Desktop\AIML-PY\DeepLearning_test> '''

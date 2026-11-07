
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_digit.py path/to/image.png")
        raise SystemExit(1)
    image_path = Path(sys.argv[1])
    image = Image.open(image_path).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32)
    if array.mean() > 127:
        array = 255.0 - array
    tensor = torch.tensor(array / 255.0).unsqueeze(0).unsqueeze(0)

    checkpoint = torch.load(
        Path(__file__).resolve().parent / "mnist_model.pth",
        map_location="cpu",
        weights_only=True
    )
    model = MNISTMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    digit = int(probabilities.argmax())
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {100 * probabilities[digit].item():.2f}%")
    print("All class probabilities:")
    for i, p in enumerate(probabilities.tolist()):
        print(f"  {i}: {100*p:.2f}%")

if __name__ == "__main__":
    main()

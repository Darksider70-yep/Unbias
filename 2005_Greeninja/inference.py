from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


IMAGE_SIZE = 128
MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pth"
DEVICE = torch.device("cpu")
_MODEL: nn.Module | None = None


class GenderClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = _center_square_crop(image)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def _load_model() -> nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = GenderClassifier().to(DEVICE)

    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _MODEL = model
    return _MODEL


def predict(image_path: str) -> Tuple[int, float]:
    """
    Returns:
    label: int (0 = Male, 1 = Female)
    confidence: float (0-1)
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_file) as image:
        inputs = _preprocess(image).to(DEVICE)

    model = _load_model()
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[0]
        label = int(torch.argmax(probs).item())
        confidence = float(probs[label].item())

    return label, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gender classification inference")
    parser.add_argument("image_path", type=str, help="Path to one input image")
    args = parser.parse_args()

    out_label, out_conf = predict(args.image_path)
    print(f"label={out_label} confidence={out_conf:.6f}")

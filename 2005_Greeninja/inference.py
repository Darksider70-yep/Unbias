from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import torch


MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pth"
DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
RESIZE_SHORT_EDGE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL: torch.jit.ScriptModule | None = None


def _load_model() -> torch.jit.ScriptModule:
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _MODEL = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
        _MODEL.eval()
    return _MODEL


def _resize_keep_aspect(image: Image.Image, short_edge: int) -> Image.Image:
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("Invalid image with zero width/height.")

    if width < height:
        new_width = short_edge
        new_height = int(round(height * (short_edge / width)))
    else:
        new_height = short_edge
        new_width = int(round(width * (short_edge / height)))

    return image.resize((new_width, new_height), Image.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return image.crop((left, top, left + size, top + size))


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = _resize_keep_aspect(image, RESIZE_SHORT_EDGE)
    image = _center_crop(image, IMAGE_SIZE)

    original = _to_tensor(image)
    flipped = _to_tensor(ImageOps.mirror(image))
    batch = torch.stack([original, flipped], dim=0)
    return batch


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
        probs = torch.softmax(logits, dim=1)
        mean_probs = probs.mean(dim=0)

    label = int(torch.argmax(mean_probs).item())
    confidence = float(mean_probs[label].item())
    return label, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gender classification inference")
    parser.add_argument("image_path", type=str, help="Path to one input image")
    args = parser.parse_args()

    out_label, out_conf = predict(args.image_path)
    print(f"label={out_label} confidence={out_conf:.6f}")

# scripts/calibrate_temperature.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from core.classifier import PresentationSignalModel


# ----------------------------
# Synthetic validation data
# ----------------------------
def generate_validation_data(n=600, dim=128):
    X = []
    y = []

    for _ in range(n):
        cls = np.random.choice([0, 1, 2])
        vec = np.random.normal(0.5, 0.2, dim)

        if cls == 0:      # femininity signal
            vec[:20] += 0.15
        elif cls == 1:    # masculinity signal
            vec[20:40] += 0.15
        else:             # ambiguity
            vec += np.random.normal(0, 0.25, dim)

        X.append(vec.astype(np.float32))
        y.append(cls)

    return torch.tensor(X), torch.tensor(y)


# ----------------------------
# Temperature scaling module
# ----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


# ----------------------------
# Calibration routine
# ----------------------------
def calibrate():
    model = PresentationSignalModel()
    scaler = TemperatureScaler()

    X_val, y_val = generate_validation_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    # Collect ensemble logits
    with torch.no_grad():
        logits = []
        for x in X_val:
            x = x.unsqueeze(0)
            model_outputs = []
            for m in model.models:
                raw = m[:-1](x)  # remove softmax
                model_outputs.append(raw)
            mean_logits = torch.mean(torch.stack(model_outputs), dim=0)
            logits.append(mean_logits)

        logits = torch.cat(logits)
        labels = y_val

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = scaler.temperature.item()
    print(f"\nCalibrated temperature: {T:.3f}")

    os.makedirs("models/unbias", exist_ok=True)
    torch.save({"temperature": T}, "models/unbias/temperature.pt")


if __name__ == "__main__":
    calibrate()

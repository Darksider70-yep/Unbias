# scripts/train_classifier.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from core.classifier import GenderPresentationClassifier

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIM = 128
EPOCHS = 60
LR = 0.01
SAMPLES_PER_CLASS = 300

SAVE_PATH = "models/unbias/classifier.pt"

# -----------------------------
# Synthetic Dataset Generator
# -----------------------------
def generate_synthetic_data():
    X = []
    y = []

    # Class 0: female-presenting (slightly higher vertical texture)
    for _ in range(SAMPLES_PER_CLASS):
        vec = np.random.normal(0.5, 0.15, INPUT_DIM)
        vec[-3:] += [0.0, 0.2, -0.1]   # structured bias
        X.append(vec)
        y.append(0)

    # Class 1: male-presenting (slightly higher horizontal texture)
    for _ in range(SAMPLES_PER_CLASS):
        vec = np.random.normal(0.5, 0.15, INPUT_DIM)
        vec[-3:] += [0.2, -0.1, 0.0]
        X.append(vec)
        y.append(1)

    # Class 2: uncertain (high noise, no structure)
    for _ in range(SAMPLES_PER_CLASS):
        vec = np.random.normal(0.5, 0.30, INPUT_DIM)
        vec[-3:] += [0.0, 0.0, 0.0]
        X.append(vec)
        y.append(2)

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.long)
    )

# -----------------------------
# Train
# -----------------------------
def train():
    model = GenderPresentationClassifier(INPUT_DIM)
    net = model.model

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    X, y = generate_synthetic_data()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = net(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # Save trained weights
    torch.save(net.state_dict(), SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()

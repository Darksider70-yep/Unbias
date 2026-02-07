# scripts/train_synthetic.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from core.classifier import PresentationSignalModel


# -----------------------------
# Synthetic data generator
# -----------------------------
def generate_synthetic_data(n=3000, dim=128):
    X, y = [], []

    for _ in range(n):
        cls = np.random.choice([0, 1, 2], p=[0.33, 0.33, 0.34])
        vec = np.random.normal(0.5, 0.15, dim)

        if cls == 0:      # femininity signal
            vec[:30] += 0.20
            target = [1.0, 0.0, 0.0]
        elif cls == 1:    # masculinity signal
            vec[30:60] += 0.20
            target = [0.0, 1.0, 0.0]
        else:             # ambiguity
            vec += np.random.normal(0, 0.25, dim)
            target = [0.33, 0.33, 0.34]

        X.append(vec.astype(np.float32))
        y.append(np.array(target, dtype=np.float32))

    return np.stack(X), np.stack(y)


# -----------------------------
# Training loop (ensemble-safe)
# -----------------------------
def train():
    X, y = generate_synthetic_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PresentationSignalModel()
    os.makedirs("models/unbias", exist_ok=True)

    for idx, net in enumerate(model.models):
        print(f"\n🔹 Training ensemble member {idx}")

        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(15):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = net(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Model {idx} | Epoch {epoch+1:02d} | "
                f"Loss: {total_loss/len(loader):.4f}"
            )

        path = f"models/unbias/presentation_model_{idx}.pt"
        torch.save(net.state_dict(), path)
        print(f"✅ Saved {path}")

    print("\n🎉 Synthetic ensemble training complete.")


if __name__ == "__main__":
    train()

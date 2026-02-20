# scripts/train_synthetic.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from core.classifier import PresentationSignalModel


# -------------------------------------------------
# Synthetic data generator (uncertainty-aware)
# -------------------------------------------------
def generate_synthetic_data(n=3000, dim=128):
    X, y = [], []

    for _ in range(n):
        cls = np.random.choice([0, 1, 2], p=[0.33, 0.33, 0.34])
        vec = np.random.normal(0.5, 0.15, dim)

        if cls == 0:  # femininity-presenting
            vec[:30] += 0.20
            target = [0.75, 0.05, 0.20]

        elif cls == 1:  # masculinity-presenting
            vec[30:60] += 0.20
            target = [0.05, 0.75, 0.20]

        else:  # ambiguous presentation
            vec += np.random.normal(0, 0.25, dim)
            target = [0.33, 0.33, 0.34]

        X.append(vec.astype(np.float32))
        y.append(np.array(target, dtype=np.float32))

    return np.stack(X), np.stack(y)


# -------------------------------------------------
# Training routine (ensemble-safe + uncertainty-safe)
# -------------------------------------------------
def train():
    X, y = generate_synthetic_data()
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PresentationSignalModel()
    os.makedirs("models/unbias", exist_ok=True)

    for idx, net in enumerate(model.models):
        print(f"\n🔹 Training ensemble member {idx}")
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(20):
            total_loss = 0.0

            for xb, yb in loader:
                optimizer.zero_grad()

                preds = net(xb)
                loss = loss_fn(preds, yb)

                # ---------------------------------------------
                # 1) Ambiguity floor (never collapse to zero)
                # ---------------------------------------------
                ambiguity_floor = 0.15
                ambiguity_penalty = torch.mean(
                    torch.relu(ambiguity_floor - preds[:, 2])
                )

                # ---------------------------------------------
                # 2) Competition-based ambiguity regularization
                #    If fem & masc compete, ambiguity must rise
                # ---------------------------------------------
                competition = torch.abs(preds[:, 0] - preds[:, 1])
                ambiguity_boost = torch.mean(
                    torch.relu(competition - preds[:, 2])
                )

                # Final loss
                loss = loss + 0.2 * ambiguity_penalty + 0.3 * ambiguity_boost

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"Model {idx} | Epoch {epoch+1:02d} | "
                f"Loss: {total_loss / len(loader):.4f}"
            )

        path = f"models/unbias/presentation_model_{idx}.pt"
        torch.save(net.state_dict(), path)
        print(f"✅ Saved {path}")

    print("\n🎉 Synthetic ensemble training complete (uncertainty preserved).")


if __name__ == "__main__":
    train()

# core/classifier.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class GenderPresentationClassifier:
    """
    Lightweight probabilistic classifier for gender presentation.
    Outputs probabilities for:
    - female
    - male
    - uncertain
    """

    def __init__(self, input_dim: int = 128):
        self.device = torch.device("cpu")

        # Simple, interpretable MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # female, male, uncertain
        ).to(self.device)

        try:
            self.model.load_state_dict(
                torch.load("models/unbias/classifier.pt", map_location=self.device)
            )
            print("Loaded trained classifier weights.")
        except FileNotFoundError:
            print("No trained weights found. Using untrained model.")

        self.model.eval()

        self.labels = ["female", "male", "uncertain"]

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Takes a feature vector and returns probabilistic predictions.
        """

        if features is None or features.size == 0:
            return {"female": 0.0, "male": 0.0, "uncertain": 1.0}

        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)
            temperature = 0.7  # < 1 sharpens distribution, demo-safe
            probs = torch.softmax(logits / temperature, dim=0).cpu().numpy()

        # Convert to dict
        output = {label: float(prob) for label, prob in zip(self.labels, probs)}
        # Gentle neutral prior to avoid perfect uniformity
        prior = {"female": 0.34, "male": 0.33, "uncertain": 0.33}

        output = {
            k: 0.9 * output[k] + 0.1 * prior[k]
            for k in output
        }
        # Numerical safety
        total = sum(output.values())
        if total > 0:
            output = {k: v / total for k, v in output.items()}

        return output

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

        # We are NOT training yet — demo-safe initialization
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
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Convert to dict
        output = {label: float(prob) for label, prob in zip(self.labels, probs)}

        # Numerical safety
        total = sum(output.values())
        if total > 0:
            output = {k: v / total for k, v in output.items()}

        return output

# core/classifier.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import os

class PresentationSignalModel:
    """
    Estimates gender presentation signals:
    femininity, masculinity, ambiguity
    """

    def __init__(self, input_dim: int = 128):
        self.device = torch.device("cpu")

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=0)
        ).to(self.device)

        path = "models/unbias/presentation_model.pt"
        if os.path.exists(path):
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            print("Loaded calibrated presentation model.")

        self.model.eval()
        self.labels = ["femininity", "masculinity", "ambiguity"]

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            probs = self.model(x).numpy()

        return dict(zip(self.labels, probs))

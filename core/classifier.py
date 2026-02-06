# core/classifier.py

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict


class PresentationSignalModel:
    """
    Ensemble-based presentation signal estimator (v1.1)
    """

    def __init__(self, input_dim: int = 128):
        self.device = torch.device("cpu")
        self.labels = ["femininity", "masculinity", "ambiguity"]
        self.models = []

        self.temperature = 1.0
        temp_path = "models/unbias/temperature.pt"
        if os.path.exists(temp_path):
            self.temperature = torch.load(temp_path)["temperature"]

        for i in range(3):
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Softmax(dim=0)
            ).to(self.device)

            path = f"models/unbias/presentation_model_{i}.pt"
            if os.path.exists(path):
                model.load_state_dict(
                    torch.load(path, map_location=self.device)
                )

            model.eval()
            self.models.append(model)

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        x = torch.tensor(features, dtype=torch.float32)

        outputs = []
        for model in self.models:
            with torch.no_grad():
                with torch.no_grad():
                    logits = model[:-1](x)          # remove softmax
                    scaled = logits / self.temperature
                    probs = torch.softmax(scaled, dim=0).numpy()
                    outputs.append(probs)
                    
        mean_output = np.mean(outputs, axis=0)
        return dict(zip(self.labels, mean_output))

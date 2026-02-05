# core/classifier.py

import numpy as np
from typing import Dict

class GenderPresentationClassifier:
    def __init__(self):
        pass

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Returns probabilistic gender presentation.
        """
        # TEMP: mock probabilities
        probs = {
            "female": 0.4,
            "male": 0.4,
            "uncertain": 0.2
        }

        # Normalize (safety)
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

# core/uncertainty.py

import math
from typing import Dict

class UncertaintyHandler:
    def __init__(self, entropy_threshold: float = 0.9):
        self.entropy_threshold = entropy_threshold

    def entropy(self, probs: Dict[str, float]) -> float:
        return -sum(p * math.log(p + 1e-9) for p in probs.values())

    def should_abstain(self, probs: Dict[str, float]) -> bool:
        return self.entropy(probs) > self.entropy_threshold

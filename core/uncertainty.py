# core/uncertainty.py

from typing import Dict

class UncertaintyHandler:
    def __init__(self, min_confidence: float = 0.45):
        """
        Abstain only if no class is confident enough.
        """
        self.min_confidence = min_confidence

    def should_abstain(self, probs: Dict[str, float]) -> bool:
        """
        Abstain if the highest non-uncertain probability is too low.
        """
        confident_classes = {
            k: v for k, v in probs.items() if k != "uncertain"
        }

        max_conf = max(confident_classes.values())
        return max_conf < self.min_confidence

# core/aggregator.py

from typing import List, Dict

class Aggregator:
    def aggregate(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        totals = {
            "female": 0.0,
            "male": 0.0,
            "uncertain": 0.0
        }

        for pred in predictions:
            for key in totals:
                totals[key] += pred.get(key, 0.0)

        totals["total"] = sum(totals.values())
        return totals

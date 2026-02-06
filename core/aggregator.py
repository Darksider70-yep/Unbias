# core/aggregator.py

from typing import List, Dict

class Aggregator:
    """
    Aggregates presentation signals across individuals.
    """

    def aggregate(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        totals = {
            "femininity": 0.0,
            "masculinity": 0.0,
            "ambiguity": 0.0
        }

        for pred in predictions:
            for k in totals:
                totals[k] += float(pred.get(k, 0.0))

        totals["total"] = sum(totals.values())
        return totals

# pipelines/inference_pipeline.py

import numpy as np

from core.detector import PersonDetector
from core.feature_extractor import FeatureExtractor
from core.classifier import PresentationSignalModel
from core.aggregator import Aggregator


class InferencePipeline:
    def __init__(self):
        self.detector = PersonDetector(conf=0.35)
        self.extractor = FeatureExtractor()
        self.model = PresentationSignalModel()
        self.aggregator = Aggregator()

    def jitter_crops(self, image, box, n=3):
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        crops = []

        for _ in range(n):
            dx = np.random.randint(-5, 6)
            dy = np.random.randint(-5, 6)

            nx1 = max(0, x1 + dx)
            ny1 = max(0, y1 + dy)
            nx2 = min(w, x2 + dx)
            ny2 = min(h, y2 + dy)

            crop = image[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crops.append(crop)

        return crops

    def run(self, image: np.ndarray):
        boxes = self.detector.detect(image)
        predictions = []

        for box in boxes:
            crops = self.jitter_crops(image, box, n=3)
            signals = []

            for crop in crops:
                features = self.extractor.extract(crop)
                signals.append(self.model.predict(features))

            avg_signal = {
                k: np.mean([s[k] for s in signals])
                for k in signals[0]
            }

            predictions.append(avg_signal)

        return self.aggregator.aggregate(predictions)

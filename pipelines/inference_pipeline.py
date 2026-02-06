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

    def run(self, image: np.ndarray):
        if image is None:
            raise ValueError("Input image is None")

        boxes = self.detector.detect(image)
        predictions = []

        for (x1, y1, x2, y2) in boxes:
            crop = image[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            features = self.extractor.extract(crop)
            signals = self.model.predict(features)
            predictions.append(signals)

        return self.aggregator.aggregate(predictions)

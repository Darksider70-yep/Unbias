# pipelines/inference_pipeline.py

import numpy as np

from core.detector import PersonDetector
from core.feature_extractor import FeatureExtractor
from core.classifier import GenderPresentationClassifier
from core.uncertainty import UncertaintyHandler
from core.aggregator import Aggregator


class InferencePipeline:
    def __init__(self):
        self.detector = PersonDetector()
        self.extractor = FeatureExtractor()
        self.classifier = GenderPresentationClassifier()
        self.uncertainty = UncertaintyHandler()
        self.aggregator = Aggregator()

    def run(self, image: np.ndarray):
        if image is None:
            raise ValueError("Input image is None")

        boxes = self.detector.detect(image)
        predictions = []

        for (x1, y1, x2, y2) in boxes:
            person_crop = image[y1:y2, x1:x2]

            if person_crop is None or person_crop.size == 0:
                continue

            features = self.extractor.extract(person_crop)
            probs = self.classifier.predict(features)

            if self.uncertainty.should_abstain(probs):
                probs = {"female": 0.0, "male": 0.0, "uncertain": 1.0}

            predictions.append(probs)

        return self.aggregator.aggregate(predictions)

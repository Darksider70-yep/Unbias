# pipelines/inference_pipeline.py

import numpy as np

from core.detector import PersonDetector
from core.cropper import Cropper
from core.feature_extractor import FeatureExtractor
from core.classifier import PresentationSignalModel
from core.aggregator import Aggregator


class InferencePipeline:
    """
    End-to-end inference pipeline for Unbias v1.1
    Detector → Cropper → FeatureExtractor → Ensemble Model → Aggregator
    """

    def __init__(self):
        self.detector = PersonDetector(conf=0.35)
        self.cropper = Cropper(jitter_pixels=5, num_crops=3)
        self.extractor = FeatureExtractor()
        self.model = PresentationSignalModel()
        self.aggregator = Aggregator()

    def run(self, image: np.ndarray):
        if image is None:
            raise ValueError("Input image is None")

        boxes = self.detector.detect(image)
        predictions = []

        for box in boxes:
            crops = self.cropper.jitter_crops(image, box)

            if not crops:
                continue

            signals = []
            for crop in crops:
                features = self.extractor.extract(crop)
                signals.append(self.model.predict(features))

            # Average signals for this person
            avg_signal = {
                key: float(np.mean([s[key] for s in signals]))
                for key in signals[0]
            }

            predictions.append(avg_signal)

        return self.aggregator.aggregate(predictions)

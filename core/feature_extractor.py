# core/feature_extractor.py

import cv2
import numpy as np

class FeatureExtractor:
    """
    Extracts non-identifying visual features from a person crop.
    """

    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Returns a fixed-size feature vector.
        """

        # Safety check
        if person_crop is None or person_crop.size == 0:
            return np.zeros(self.output_dim)

        # Resize to normalize scale
        resized = cv2.resize(person_crop, (64, 128), interpolation=cv2.INTER_AREA)

        # Convert to grayscale (removes color identity cues)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Normalize
        gray = gray.astype(np.float32) / 255.0

        # Simple texture + shape signal
        flattened = gray.flatten()

        # Reduce dimensionality safely
        if flattened.shape[0] >= self.output_dim:
            features = flattened[:self.output_dim]
        else:
            pad = np.zeros(self.output_dim - flattened.shape[0])
            features = np.concatenate([flattened, pad])

        return features.astype(np.float32)

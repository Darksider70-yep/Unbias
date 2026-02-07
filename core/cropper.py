# core/cropper.py

import numpy as np


class Cropper:
    """
    Handles person cropping and augmentation for robustness.
    Designed for Unbias v1.1 (multi-crop consistency).
    """

    def __init__(
        self,
        jitter_pixels: int = 5,
        num_crops: int = 3
    ):
        self.jitter_pixels = jitter_pixels
        self.num_crops = num_crops

    def _clamp_box(self, x1, y1, x2, y2, h, w):
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        return x1, y1, x2, y2

    def crop(self, image: np.ndarray, box):
        """
        Returns a single tight crop.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        x1, y1, x2, y2 = self._clamp_box(x1, y1, x2, y2, h, w)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    def jitter_crops(self, image: np.ndarray, box):
        """
        Returns multiple jittered crops around a bounding box.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        crops = []

        for _ in range(self.num_crops):
            dx = np.random.randint(-self.jitter_pixels, self.jitter_pixels + 1)
            dy = np.random.randint(-self.jitter_pixels, self.jitter_pixels + 1)

            nx1 = x1 + dx
            ny1 = y1 + dy
            nx2 = x2 + dx
            ny2 = y2 + dy

            nx1, ny1, nx2, ny2 = self._clamp_box(
                nx1, ny1, nx2, ny2, h, w
            )

            crop = image[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crops.append(crop)

        return crops

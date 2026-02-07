# core/detector.py

import os
from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        conf: float = 0.35,
        model_path: str = "models/detector/yolov8n.pt"
    ):
        self.conf = conf

        # Resolve absolute path (critical)
        self.model_path = os.path.abspath(model_path)

        # Fail fast if missing (prevents auto-download)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {self.model_path}.\n"
                "Please place yolov8n.pt in models/detector/"
            )

        self.model = YOLO(self.model_path)

    def detect(self, image):
        results = self.model(
            image,
            conf=self.conf,
            classes=[0],  # person class only
            verbose=False
        )

        boxes = []
        for r in results:
            for b in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, b.tolist())
                boxes.append((x1, y1, x2, y2))

        return boxes

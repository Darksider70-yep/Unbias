# core/detector.py

from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

BoundingBox = Tuple[int, int, int, int]  # x1, y1, x2, y2

class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        """
        YOLO-based person detector.
        Detects only the 'person' class.
        """
        self.model = YOLO("models/detector/yolov8n.pt")
        self.conf = conf
        self.person_class_id = 0  # YOLO class index for 'person'

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            verbose=False
        )

        boxes: List[BoundingBox] = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != self.person_class_id:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((
                    int(x1), int(y1),
                    int(x2), int(y2)
                ))

        return boxes

# core/feature_extractor.py

import cv2
import numpy as np
import mediapipe as mp

class FeatureExtractor:
    """
    Extracts non-identifying visual + pose-based features
    for presentation signal estimation.
    """

    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def extract_pose_features(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return np.zeros(6, dtype=np.float32)

        lm = result.pose_landmarks.landmark

        shoulder_width = abs(lm[11].x - lm[12].x)
        hip_width = abs(lm[23].x - lm[24].x)
        torso_length = abs(lm[11].y - lm[23].y)
        ratio_shoulder_hip = shoulder_width / (hip_width + 1e-6)
        arm_spread = abs(lm[11].x - lm[15].x) + abs(lm[12].x - lm[16].x)

        return np.array(
            [
                shoulder_width,
                hip_width,
                ratio_shoulder_hip,
                torso_length,
                arm_spread,
                1.0  # bias stabilizer
            ],
            dtype=np.float32
        )

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        if person_crop is None or person_crop.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        resized = cv2.resize(person_crop, (64, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        flattened = gray.flatten()
        pose_features = self.extract_pose_features(resized)

        usable_len = self.output_dim - len(pose_features)
        base_features = flattened[:usable_len]

        features = np.concatenate([base_features, pose_features])
        return features.astype(np.float32)

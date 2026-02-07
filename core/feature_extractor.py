# core/feature_extractor.py

import cv2
import numpy as np

# Optional MediaPipe (graceful fallback)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions")
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from skimage.feature import hog


class FeatureExtractor:
    """
    Extracts non-identifying structured visual features
    for presentation-signal estimation (v1.1).
    """

    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim
        self.pose = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.pose = mp.solutions.pose.Pose(static_image_mode=True)
            except Exception:
                self.pose = None

    def extract_pose_features(self, image: np.ndarray) -> np.ndarray:
        if self.pose is None:
            return np.zeros(6, dtype=np.float32)

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
                1.0
            ],
            dtype=np.float32
        )

    def extract_structured_visual_features(self, gray: np.ndarray) -> np.ndarray:
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

        edge_h = np.mean(np.abs(sobelx))
        edge_v = np.mean(np.abs(sobely))

        mid = gray.shape[1] // 2
        left = gray[:, :mid]
        right = np.fliplr(gray[:, mid:])

        symmetry = np.mean(np.abs(left - right[:, :left.shape[1]]))

        return np.concatenate([
            hog_feat[:80],
            np.array([edge_h, edge_v, symmetry], dtype=np.float32)
        ])

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        if person_crop is None or person_crop.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        resized = cv2.resize(person_crop, (64, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        visual_features = self.extract_structured_visual_features(gray)
        pose_features = self.extract_pose_features(resized)

        features = np.concatenate([visual_features, pose_features])

        if len(features) >= self.output_dim:
            features = features[:self.output_dim]
        else:
            features = np.pad(features, (0, self.output_dim - len(features)))

        return features.astype(np.float32)

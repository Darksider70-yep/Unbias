# core/feature_extractor.py

import cv2
import numpy as np

# -----------------------------
# Optional MediaPipe Support
# -----------------------------
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions")
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FeatureExtractor:
    """
    Extracts non-identifying visual features.
    Pose-based features are used only if MediaPipe is available.
    """

    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim

        self.pose = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.pose = mp.solutions.pose.Pose(static_image_mode=True)
            except Exception:
                # Safety net: even partial MediaPipe installs won't crash
                self.pose = None

    def extract_pose_features(self, image: np.ndarray) -> np.ndarray:
        """
        Returns pose-based geometry features if available,
        otherwise returns zeros (graceful degradation).
        """
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
                1.0  # bias stabilizer
            ],
            dtype=np.float32
        )

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        if person_crop is None or person_crop.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        # Normalize scale and color
        resized = cv2.resize(person_crop, (64, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        flattened = gray.flatten()
        pose_features = self.extract_pose_features(resized)

        usable_len = self.output_dim - len(pose_features)
        base_features = flattened[:usable_len]

        features = np.concatenate([base_features, pose_features])
        return features.astype(np.float32)

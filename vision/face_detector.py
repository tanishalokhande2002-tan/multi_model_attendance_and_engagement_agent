"""
Privacy-Conscious Face Detection Module.

Uses practical, locally runnable OpenCV computer vision detectors with contrast-adaptive
preprocessing to handle illumination variations. Does NOT perform facial recognition,
feature encoding, or identity matching.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Locally executable face detector using OpenCV Haar Cascade or DNN.
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination robustness.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (30, 30),
        use_clahe: bool = True,
    ):
        """
        Initialize the face detector.

        Args:
            cascade_path: Optional custom path to Haar XML. If None, uses OpenCV's default.
            scale_factor: Image pyramid scale factor (>1.0).
            min_neighbors: Neighborhood detection threshold for candidate bounding boxes.
            min_face_size: Minimum (width, height) bounding box dimensions in pixels.
            use_clahe: Whether to apply CLAHE preprocessing to normalize lighting variations.
        """
        self.scale_factor = max(1.01, scale_factor)
        self.min_neighbors = max(1, min_neighbors)
        self.min_face_size = min_face_size
        self.use_clahe = use_clahe

        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.cascade_path = Path(cascade_path)
        if not self.cascade_path.exists():
            raise FileNotFoundError(f"Face detector model not found at {self.cascade_path}")

        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load OpenCV CascadeClassifier from {self.cascade_path}")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        logger.info("Initialized FaceDetector with model: %s", self.cascade_path.name)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and adaptively equalize contrast to mitigate shadows and glare.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray = frame.copy()
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        if self.clahe is not None:
            gray = self.clahe.apply(gray)
        else:
            gray = cv2.equalizeHist(gray)

        return gray

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in a single frame.

        Args:
            frame: BGR or grayscale numpy image array.

        Returns:
            List of bounding boxes as (x, y, w, h) coordinates.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            gray = self.preprocess_frame(frame)
            detections = self.classifier.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(detections) == 0:
                return []

            # Return list of integer coordinates
            boxes: List[Tuple[int, int, int, int]] = []
            for (x, y, w, h) in detections:
                boxes.append((int(x), int(y), int(w), int(h)))

            return boxes

        except Exception as e:
            logger.error("Error during face detection: %s", e)
            return []

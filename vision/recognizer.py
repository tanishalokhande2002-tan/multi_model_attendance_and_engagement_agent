"""
Face Recognition and Feature Encoding Module.
Extracts normalized feature representations from face crops and performs
consent-checked cosine similarity matching against enrolled student rosters.
Includes a self-contained OpenCV-based feature encoder with optional DeepFace backend.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Extracts face feature embeddings and matches candidate faces
    against authorized student representations with strict consent enforcement.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        target_size: Tuple[int, int] = (96, 96),
        use_deepface: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.target_size = target_size
        self.use_deepface = use_deepface
        self.deepface_available = False

        if use_deepface:
            try:
                import deepface  # type: ignore
                self.deepface_available = True
                logger.info("DeepFace is enabled and available.")
            except ImportError:
                logger.warning("DeepFace requested but not installed. Falling back to built-in OpenCV feature encoder.")

    def extract_embedding(self, face_img: np.ndarray) -> Optional[List[float]]:
        """
        Extract a normalized 128-dimensional feature embedding from a cropped face image.
        """
        if face_img is None or face_img.size == 0:
            return None

        # Check minimum viable size
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            return None

        try:
            # If DeepFace is requested and available
            if self.use_deepface and self.deepface_available:
                try:
                    from deepface import DeepFace  # type: ignore
                    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) if len(face_img.shape) == 3 else face_img
                    embeddings = DeepFace.represent(
                        img_path=rgb_face,
                        model_name="Facenet512",
                        enforce_detection=False,
                    )
                    if embeddings and len(embeddings) > 0:
                        vec = embeddings[0]["embedding"]
                        vec_np = np.array(vec, dtype=np.float32)
                        norm = np.linalg.norm(vec_np)
                        if norm > 1e-6:
                            vec_np = vec_np / norm
                        return vec_np.tolist()
                except Exception as e:
                    logger.debug("DeepFace representation fallback: %s", e)

            # Built-in robust OpenCV multi-scale normalized descriptor
            # Preprocess: convert to grayscale, resize, equalize
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img.copy()

            resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            eq = clahe.apply(resized)

            # Spatial grid cell pooling (8x8 cells across 96x96 = 12x12 cells)
            # Compute gradient magnitudes and orientations (HOG-like lightweight descriptor)
            gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            # Quantize into 8 orientation bins across 4x4 spatial blocks
            h, w = eq.shape
            block_h, block_w = h // 4, w // 4
            descriptor = []

            for r in range(4):
                for c in range(4):
                    block_mag = mag[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
                    block_ang = ang[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
                    hist, _ = np.histogram(block_ang, bins=8, range=(0, 360), weights=block_mag)
                    descriptor.extend(hist)

            # 4 * 4 * 8 = 128 dimensional feature vector
            vec = np.array(descriptor, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            else:
                vec = np.zeros(128, dtype=np.float32)

            return [round(float(x), 6) for x in vec]

        except Exception as e:
            logger.error("Failed to extract embedding: %s", e)
            return None

    @staticmethod
    def compute_cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two unit vectors: dot(u, v)."""
        if not emb1 or not emb2 or len(emb1) != len(emb2):
            return 0.0

        u = np.array(emb1, dtype=np.float32)
        v = np.array(emb2, dtype=np.float32)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0

        cos_sim = float(np.dot(u, v) / (norm_u * norm_v))
        return max(0.0, min(1.0, cos_sim))

    def match_face_to_roster(
        self,
        candidate_embedding: List[float],
        student_roster: List[Dict[str, Any]],
        strict_consent: bool = True,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Compare candidate embedding against consented enrolled students.

        Returns:
            (student_id, student_name, confidence_score)
        """
        if not candidate_embedding or not student_roster:
            return None, None, 0.0

        best_score = 0.0
        best_student: Optional[Dict[str, Any]] = None

        for student in student_roster:
            # Enforce consent safeguard: never match unconsented students
            if strict_consent and not student.get("consent_enrolled", 0):
                continue

            ref_encoding = student.get("face_encoding")
            if not ref_encoding:
                continue

            sim = self.compute_cosine_similarity(candidate_embedding, ref_encoding)
            if sim > best_score:
                best_score = sim
                best_student = student

        if best_student and best_score >= self.similarity_threshold:
            return (
                best_student["student_id"],
                best_student.get("name", "Unknown"),
                round(best_score, 4),
            )

        return None, None, round(best_score, 4)

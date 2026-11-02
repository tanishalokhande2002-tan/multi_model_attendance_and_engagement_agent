"""
Vision Module for Multimodal Attendance & Engagement Agent.
Provides face detection, persistent centroid tracking, face recognition,
and temporal presence evaluation.
"""

from .face_detector import FaceDetector
from .tracker import CentroidTracker, TrackedFace
from .recognizer import FaceRecognizer
from .attendance import AttendanceManager, AttendanceRecord
from .pipeline import VisionPipeline

__all__ = [
    "FaceDetector",
    "CentroidTracker",
    "TrackedFace",
    "FaceRecognizer",
    "AttendanceManager",
    "AttendanceRecord",
    "VisionPipeline",
]

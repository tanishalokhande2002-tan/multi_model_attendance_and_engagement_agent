"""
Unit tests for Computer Vision, Centroid Tracking, and Face Matching.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.face_detector import FaceDetector
from vision.tracker import CentroidTracker, TrackedFace
from vision.recognizer import FaceRecognizer
from vision.attendance import AttendanceManager


def test_face_detector_initialization():
    """Verify face detector loads default Haar model without errors."""
    detector = FaceDetector()
    assert detector.classifier is not None
    assert not detector.classifier.empty()

    # Test processing a dummy image
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = detector.detect_faces(dummy_frame)
    assert isinstance(boxes, list)


def test_centroid_tracker_lifecycle():
    """Verify tracker registers tracks, matches centroid movements, and deregisters lost tracks."""
    tracker = CentroidTracker(max_disappeared_frames=3, max_centroid_distance=50.0)

    # Frame 1: Register box at (50, 50, 40, 40)
    tracks_f1 = tracker.update([(50, 50, 40, 40)], timestamp_sec=0.0)
    assert len(tracks_f1) == 1
    t1 = tracks_f1[0]
    assert t1.track_id == 1
    assert t1.centroid == (70, 70)

    # Frame 2: Slight displacement to (55, 52, 40, 40) -> Centroid ~ (75, 72)
    tracks_f2 = tracker.update([(55, 52, 40, 40)], timestamp_sec=1.0)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == 1
    assert tracks_f2[0].movement_energy > 0.0

    # Frame 3-6: Disappeared
    tracker.update([], timestamp_sec=2.0)
    tracker.update([], timestamp_sec=3.0)
    tracker.update([], timestamp_sec=4.0)
    tracks_lost = tracker.update([], timestamp_sec=5.0)
    # Track should be deregistered after exceeding max_disappeared_frames (3)
    assert len(tracks_lost) == 0


def test_face_recognizer_embedding_and_consent():
    """Test feature vector extraction and consent-enforced matching."""
    recognizer = FaceRecognizer(similarity_threshold=0.70)

    # Synthetic face patch (96x96 with some pattern)
    patch = np.full((96, 96, 3), 128, dtype=np.uint8)
    patch[30:60, 30:60] = 200  # Inner feature

    emb1 = recognizer.extract_embedding(patch)
    assert emb1 is not None
    assert len(emb1) == 128
    # Embedding must be L2 normalized (unit length)
    norm = np.linalg.norm(emb1)
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Same patch should have cosine similarity ~ 1.0
    sim_self = recognizer.compute_cosine_similarity(emb1, emb1)
    assert pytest.approx(sim_self, rel=1e-3) == 1.0

    # Test consent enforcement: Student without consent MUST NOT be matched
    roster = [
        {"student_id": "STU-CONSENT", "name": "Consented Student", "consent_enrolled": 1, "face_encoding": emb1},
        {"student_id": "STU-NOCONSENT", "name": "Unconsented Student", "consent_enrolled": 0, "face_encoding": emb1},
    ]

    # Strict consent enabled:
    mid, mname, score = recognizer.match_face_to_roster(emb1, roster, strict_consent=True)
    assert mid == "STU-CONSENT"
    assert mname == "Consented Student"


def test_attendance_manager_threshold():
    """Test that minimum observation duration threshold gates presence qualification."""
    att_mgr = AttendanceManager(min_presence_duration_sec=5.0, confidence_threshold=0.65)

    # 1. Under-threshold track (seen for 2.0s)
    short_track = TrackedFace(
        track_id=1,
        bbox=(0, 0, 50, 50),
        centroid=(25, 25),
        first_seen_sec=0.0,
        last_seen_sec=2.0,
        total_visible_sec=2.0,
        matched_student_id="STU-001",
        match_confidence=0.88,
    )
    rec_short = att_mgr.evaluate_track(short_track)
    assert rec_short.qualified is False
    assert rec_short.status == "Candidate"

    # 2. Over-threshold track (seen for 6.5s)
    long_track = TrackedFace(
        track_id=2,
        bbox=(0, 0, 50, 50),
        centroid=(25, 25),
        first_seen_sec=0.0,
        last_seen_sec=6.5,
        total_visible_sec=6.5,
        matched_student_id="STU-001",
        match_confidence=0.88,
    )
    rec_long = att_mgr.evaluate_track(long_track)
    assert rec_long.qualified is True
    assert rec_long.status == "Present"

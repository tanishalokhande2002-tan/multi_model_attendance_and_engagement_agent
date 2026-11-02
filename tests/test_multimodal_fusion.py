"""
Unit tests for Multimodal Fusion Engine and Explainable AI reasoning.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.multimodal_fusion import MultimodalFusionEngine
from vision.attendance import AttendanceRecord


def test_fusion_attendance_evidence_and_explainability():
    """Verify fusion generates verified presence with detailed explanation cards."""
    engine = MultimodalFusionEngine(min_presence_sec=5.0, face_confidence_thresh=0.65)

    # Mock video attendance records
    rec1 = AttendanceRecord(
        student_id="STU-001",
        track_id=1,
        name="Alex Chen",
        status="Present",
        observation_duration_sec=14.5,
        first_seen_sec=1.0,
        last_seen_sec=15.5,
        frames_observed=29,
        confidence_score=0.88,
        qualified=True,
        explanation="Qualified: Observed for 14.5s",
    )
    video_res = {"attendance_records": [rec1], "temporal_bins": []}

    audio_res = {
        "speech_ratio": 0.52,
        "discussion_intensity_score": 0.70,
    }

    roster = [
        {"student_id": "STU-001", "name": "Alex Chen", "grade": "10", "section": "A", "consent_enrolled": 1},
        {"student_id": "STU-002", "name": "Maya Patel", "grade": "10", "section": "A", "consent_enrolled": 1},
    ]

    evidence_list = engine.fuse_session_attendance(video_res, audio_res, roster)
    assert len(evidence_list) == 2

    # Alex Chen was seen > 5.0s with high confidence -> Present
    alex = next(e for e in evidence_list if e.student_id == "STU-001")
    assert alex.status == "Present"
    assert alex.observation_duration_sec == 14.5
    assert alex.visual_match_pct == 88.0
    assert "Verified Present" in alex.explanation
    assert "High Interaction" in alex.classroom_activity_level

    # Maya Patel was not seen -> Absent
    maya = next(e for e in evidence_list if e.student_id == "STU-002")
    assert maya.status == "Absent"
    assert maya.observation_duration_sec == 0.0
    assert "Not Detected" in maya.explanation


def test_fusion_temporal_engagement_and_pedagogy():
    """Verify second-by-second temporal alignment and pedagogical advisory generation."""
    engine = MultimodalFusionEngine()

    video_bins = [
        {"timestamp_sec": 0.0, "faces_detected": 8, "movement_energy": 12.0},
        {"timestamp_sec": 1.0, "faces_detected": 8, "movement_energy": 10.0},
        {"timestamp_sec": 2.0, "faces_detected": 8, "movement_energy": 8.0},
    ]
    audio_bins = [
        {"timestamp_sec": 0.0, "speech_active": 1, "rms_energy": 0.04},
        {"timestamp_sec": 1.0, "speech_active": 1, "rms_energy": 0.05},
        {"timestamp_sec": 2.0, "speech_active": 0, "rms_energy": 0.01},
    ]

    fused = engine.fuse_temporal_engagement(video_bins, audio_bins, expected_class_size=10)
    assert len(fused) == 3

    # Frame 0: active speech + 8 faces -> "Interactive Discussion"
    assert fused[0]["pedagogical_state"] == "Interactive Discussion"
    assert 0.0 <= fused[0]["aggregate_engagement_score"] <= 1.0

    # Frame 2: no speech + 8 faces + low motion -> "Attentive Lecture / Focus"
    assert fused[2]["pedagogical_state"] == "Attentive Lecture / Focus"

    # Test advisory for sustained quiet
    quiet_timeline = [{"speech_active": 0, "aggregate_engagement_score": 0.6} for _ in range(30)]
    advisories = engine.generate_pedagogical_advisory(quiet_timeline)
    assert len(advisories) > 0
    assert any("Discussion Prompt" in a["type"] for a in advisories)

"""
Vision-based Attendance Evaluation Logic.
Evaluates continuous tracked observations against temporal thresholds to establish
presence qualification without premature false-positives.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .tracker import TrackedFace


@dataclass
class AttendanceRecord:
    """Evaluated attendance candidate evidence record."""
    student_id: Optional[str]
    track_id: int
    name: str
    status: str  # 'Present', 'Candidate', 'Unqualified'
    observation_duration_sec: float
    first_seen_sec: float
    last_seen_sec: float
    frames_observed: int
    confidence_score: float
    qualified: bool
    explanation: str


class AttendanceManager:
    """
    Applies configurable temporal thresholds to tracked face observations
    to determine if a student meets the criteria for confirmed presence.
    """

    def __init__(
        self,
        min_presence_duration_sec: float = 5.0,
        confidence_threshold: float = 0.65,
    ):
        self.min_presence_duration_sec = min_presence_duration_sec
        self.confidence_threshold = confidence_threshold

    def evaluate_track(self, track: TrackedFace) -> AttendanceRecord:
        """
        Evaluate an individual tracked face for presence qualification.
        """
        # Duration is calculated from continuous visibility or total observed interval
        duration = max(track.total_visible_sec, (track.last_seen_sec - track.first_seen_sec))
        qualified = (duration >= self.min_presence_duration_sec)

        # Check identification
        student_id = track.matched_student_id
        student_name = track.matched_student_name or f"Track #{track.track_id}"
        conf = track.match_confidence

        if qualified and student_id:
            status = "Present"
            explanation = (
                f"Qualified: Observed for {duration:.1f}s across {track.frame_count} frames "
                f"(Threshold: {self.min_presence_duration_sec:.1f}s). Match confidence: {conf * 100:.1f}%."
            )
        elif not qualified and student_id:
            status = "Candidate"
            explanation = (
                f"Candidate under observation: Seen for {duration:.1f}s ({track.frame_count} frames). "
                f"Requires {self.min_presence_duration_sec:.1f}s to qualify."
            )
        else:
            status = "Unidentified"
            explanation = (
                f"Unenrolled or anonymous face observed for {duration:.1f}s. "
                "No authorized consented student match found."
            )

        return AttendanceRecord(
            student_id=student_id,
            track_id=track.track_id,
            name=student_name,
            status=status,
            observation_duration_sec=round(duration, 2),
            first_seen_sec=round(track.first_seen_sec, 2),
            last_seen_sec=round(track.last_seen_sec, 2),
            frames_observed=track.frame_count,
            confidence_score=round(conf, 4),
            qualified=qualified,
            explanation=explanation,
        )

    def evaluate_all(self, tracks: List[TrackedFace]) -> List[AttendanceRecord]:
        """Evaluate a collection of tracks."""
        return [self.evaluate_track(t) for t in tracks]

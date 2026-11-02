"""
Multimodal Fusion Engine for Attendance and Classroom Engagement.
Integrates visual presence observations with acoustic participation indicators.
Provides explainable AI reasoning and cautious pedagogical analytics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AttendanceEvidence:
    """Detailed explainability record for a single student."""
    student_id: str
    name: str
    grade: str
    section: str
    status: str  # 'Present', 'Absent', 'Late', 'Excused'
    confidence_score: float
    observation_duration_sec: float
    frames_detected: int
    visual_match_pct: float
    classroom_activity_level: str
    explanation: str
    audit_notes: str


class MultimodalFusionEngine:
    """
    Fuses computer vision and acoustic features to generate verifiable attendance
    records and aggregate classroom engagement timelines.
    """

    def __init__(
        self,
        min_presence_sec: float = 5.0,
        face_confidence_thresh: float = 0.65,
        presence_weight: float = 0.35,
        movement_weight: float = 0.20,
        audio_weight: float = 0.30,
        variation_weight: float = 0.15,
    ):
        self.min_presence_sec = min_presence_sec
        self.face_confidence_thresh = face_confidence_thresh
        self.presence_weight = presence_weight
        self.movement_weight = movement_weight
        self.audio_weight = audio_weight
        self.variation_weight = variation_weight

    def fuse_session_attendance(
        self,
        video_results: Dict[str, Any],
        audio_results: Optional[Dict[str, Any]],
        student_roster: List[Dict[str, Any]],
    ) -> List[AttendanceEvidence]:
        """
        Synthesize video-tracked face matches and acoustic context into final attendance records.
        """
        # Build map of video sightings
        sightings_by_student: Dict[str, Dict[str, Any]] = {}
        for rec in video_results.get("attendance_records", []):
            if rec.student_id:
                sightings_by_student[rec.student_id] = {
                    "duration": rec.observation_duration_sec,
                    "frames": rec.frames_observed,
                    "confidence": rec.confidence_score,
                    "qualified": rec.qualified,
                }

        # Classroom acoustic context description
        if audio_results:
            intensity = audio_results.get("discussion_intensity_score", 0.0)
            speech_pct = audio_results.get("speech_ratio", 0.0) * 100.0
            if intensity >= 0.65:
                classroom_activity = f"High Interaction ({speech_pct:.0f}% speech)"
            elif intensity >= 0.35:
                classroom_activity = f"Balanced Lecture & Q&A ({speech_pct:.0f}% speech)"
            else:
                classroom_activity = f"Quiet / Independent Study ({speech_pct:.0f}% speech)"
        else:
            classroom_activity = "Audio Not Available (Video-Only Mode)"

        final_records: List[AttendanceEvidence] = []

        for student in student_roster:
            sid = student["student_id"]
            name = student.get("name", "Unknown")
            grade = student.get("grade", "")
            section = student.get("section", "")
            consent = bool(student.get("consent_enrolled", 0))

            if sid in sightings_by_student:
                data = sightings_by_student[sid]
                duration = data["duration"]
                frames = data["frames"]
                conf = data["confidence"]

                if duration >= self.min_presence_sec and conf >= self.face_confidence_thresh:
                    status = "Present"
                    explanation = (
                        f"Verified Present: Observed for {duration:.1f}s across {frames} frames "
                        f"(Threshold: {self.min_presence_sec:.1f}s). Facial match confidence: {conf * 100:.1f}%. "
                        f"Class context: {classroom_activity}."
                    )
                    audit = "Autonomous presence verified via biometric match and sustained observation threshold."
                elif conf >= self.face_confidence_thresh:
                    # High confidence but brief duration
                    status = "Absent"
                    explanation = (
                        f"Flagged under observation ({duration:.1f}s, {frames} frames). "
                        f"Did not meet minimum sustained duration requirement of {self.min_presence_sec:.1f}s."
                    )
                    audit = "Failed temporal presence duration threshold. Teacher review recommended."
                else:
                    status = "Absent"
                    explanation = (
                        f"Low confidence sighting ({conf * 100:.1f}% vs threshold {self.face_confidence_thresh * 100:.1f}%). "
                        "Insufficient biometric certainty."
                    )
                    audit = "Sub-threshold match confidence."

                final_records.append(AttendanceEvidence(
                    student_id=sid,
                    name=name,
                    grade=grade,
                    section=section,
                    status=status,
                    confidence_score=round(conf, 4),
                    observation_duration_sec=round(duration, 2),
                    frames_detected=frames,
                    visual_match_pct=round(conf * 100.0, 1),
                    classroom_activity_level=classroom_activity,
                    explanation=explanation,
                    audit_notes=audit,
                ))
            else:
                # Student not observed in video
                final_records.append(AttendanceEvidence(
                    student_id=sid,
                    name=name,
                    grade=grade,
                    section=section,
                    status="Absent",
                    confidence_score=0.0,
                    observation_duration_sec=0.0,
                    frames_detected=0,
                    visual_match_pct=0.0,
                    classroom_activity_level=classroom_activity,
                    explanation=(
                        f"Not Detected: Zero facial occurrences identified during session video. "
                        f"(Consent on file: {'Yes' if consent else 'No'})."
                    ),
                    audit_notes="Student was unobserved throughout session duration.",
                ))

        return final_records

    def fuse_temporal_engagement(
        self,
        video_bins: List[Dict[str, Any]],
        audio_bins: Optional[List[Dict[str, Any]]],
        expected_class_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Align second-by-second video presence and audio activity into an aggregate engagement timeline.
        """
        # Map audio bins by timestamp
        audio_map: Dict[int, Dict[str, Any]] = {}
        if audio_bins:
            for b in audio_bins:
                audio_map[int(b["timestamp_sec"])] = b

        fused_timeline: List[Dict[str, Any]] = []
        max_sec = len(video_bins) if video_bins else (len(audio_bins) if audio_bins else 0)

        for s in range(max_sec):
            v_bin = video_bins[s] if s < len(video_bins) else {"faces_detected": 0, "movement_energy": 0.0}
            a_bin = audio_map.get(s, {"speech_active": 0, "rms_energy": 0.0})

            faces = v_bin.get("faces_detected", 0)
            mov = v_bin.get("movement_energy", 0.0)
            speech = a_bin.get("speech_active", 0)
            rms = a_bin.get("rms_energy", 0.0)

            # Normalized sub-scores
            norm_presence = min(1.0, faces / max(1, expected_class_size))
            norm_stability = max(0.0, 1.0 - min(1.0, mov / 45.0) * 0.6)
            norm_speech = 0.85 if speech == 1 else 0.30
            norm_rms = min(1.0, rms * 15.0)

            # Multimodal composite engagement index (0.0 to 1.0)
            composite_score = (
                self.presence_weight * norm_presence +
                self.movement_weight * norm_stability +
                self.audio_weight * norm_speech +
                self.variation_weight * norm_rms
            )
            composite_score = round(max(0.0, min(1.0, composite_score)), 3)

            # Pedagogical State categorization
            if speech == 1 and faces >= 4:
                ped_state = "Interactive Discussion"
            elif speech == 0 and faces >= 4 and mov < 15.0:
                ped_state = "Attentive Lecture / Focus"
            elif speech == 0 and faces >= 4:
                ped_state = "Silent Activity / Writing"
            elif faces < 3:
                ped_state = "Low Attendance / Transition"
            else:
                ped_state = "General Classroom Flow"

            fused_timeline.append({
                "timestamp_sec": float(s),
                "faces_detected": faces,
                "movement_energy": round(mov, 2),
                "speech_active": speech,
                "aggregate_engagement_score": composite_score,
                "pedagogical_state": ped_state,
            })

        return fused_timeline

    @staticmethod
    def generate_pedagogical_advisory(timeline: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate constructive pedagogical suggestions based on engagement trend patterns.
        """
        if not timeline:
            return []

        suggestions = []
        dur = len(timeline)

        # 1. Check for extended quiet stretches
        consecutive_quiet = 0
        max_quiet = 0
        for b in timeline:
            if b.get("speech_active") == 0:
                consecutive_quiet += 1
                max_quiet = max(max_quiet, consecutive_quiet)
            else:
                consecutive_quiet = 0

        if max_quiet >= 25:
            suggestions.append({
                "type": "Discussion Prompt",
                "severity": "Info",
                "message": (
                    f"A quiet window of {max_quiet}s was observed. "
                    "Consider initiating a think-pair-share exercise or posing a concept check question."
                ),
            })

        # 2. Check for overall participation balance
        speech_bins = sum(1 for b in timeline if b.get("speech_active", 0) == 1)
        speech_pct = (speech_bins / dur * 100.0) if dur > 0 else 0.0

        if speech_pct > 75.0:
            suggestions.append({
                "type": "Pacing & Synthesis",
                "severity": "Tip",
                "message": (
                    f"High vocal activity across {speech_pct:.0f}% of the session. "
                    "Ensure adequate time for students to consolidate notes and synthesize key takeaways."
                ),
            })
        elif speech_pct < 20.0:
            suggestions.append({
                "type": "Student Interaction",
                "severity": "Actionable",
                "message": (
                    f"Vocal participation was relatively low ({speech_pct:.0f}%). "
                    "Try incorporating more frequent formative polling or student-led explanations."
                ),
            })

        # 3. Overall engagement health
        avg_score = sum(b.get("aggregate_engagement_score", 0.0) for b in timeline) / max(1, dur)
        if avg_score >= 0.70:
            suggestions.append({
                "type": "Session Health",
                "severity": "Positive",
                "message": f"Classroom engagement index remained high ({avg_score * 100:.1f}%) throughout the session.",
            })

        return suggestions

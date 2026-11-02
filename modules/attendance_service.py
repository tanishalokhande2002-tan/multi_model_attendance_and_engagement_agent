"""
Attendance Orchestration Service.
Coordinates automated session execution, vision tracking, audio analysis,
multimodal evidence fusion, anti-duplicate record storage, and manual review overrides.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

from database.db_manager import DatabaseManager
from vision.pipeline import VisionPipeline
from modules.audio_module import AudioProcessor
from modules.multimodal_fusion import MultimodalFusionEngine, AttendanceEvidence

logger = logging.getLogger(__name__)


class AttendanceService:
    """
    Manages end-to-end automated and hybrid attendance tracking workflows.
    """

    def __init__(
        self,
        db: DatabaseManager,
        vision_pipeline: Optional[VisionPipeline] = None,
        audio_processor: Optional[AudioProcessor] = None,
        fusion_engine: Optional[MultimodalFusionEngine] = None,
    ):
        self.db = db
        self.vision = vision_pipeline or VisionPipeline()
        self.audio = audio_processor or AudioProcessor()
        self.fusion = fusion_engine or MultimodalFusionEngine()

    def create_session(
        self,
        session_id: str,
        course_name: str,
        instructor_name: str = "",
        grade: str = "",
        section: str = "",
        session_date: Optional[str] = None,
        start_time: str = "",
        end_time: str = "",
        video_source: str = "",
        audio_source: str = "",
    ) -> bool:
        """Create a new classroom session in the database."""
        return self.db.create_session(
            session_id=session_id,
            course_name=course_name,
            instructor_name=instructor_name,
            grade=grade,
            section=section,
            session_date=session_date,
            start_time=start_time,
            end_time=end_time,
            video_source=video_source,
            audio_source=audio_source,
        )

    def process_automated_session(
        self,
        session_id: str,
        video_path: Optional[str | Path] = None,
        audio_path: Optional[str | Path] = None,
        grade: Optional[str] = None,
        section: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full multimodal processing pipeline on a session's recorded video and audio.
        Guarantees anti-duplicate record creation in SQLite.
        """
        # Fetch relevant student roster
        roster = self.db.get_all_students(grade=grade, section=section)
        if not roster:
            roster = self.db.get_all_students()  # Fallback to entire student body

        video_results: Dict[str, Any] = {"attendance_records": [], "temporal_bins": [], "preview_frames": []}
        audio_results: Optional[Dict[str, Any]] = None

        # 1. Process Video
        if video_path and Path(video_path).exists():
            if progress_callback:
                progress_callback(0.1, "Analyzing video stream and tracking faces...")
            video_results = self.vision.process_video_file(
                video_path=video_path,
                student_roster=roster,
                progress_callback=progress_callback,
            )

        # 2. Process Audio
        if audio_path and Path(audio_path).exists():
            if progress_callback:
                progress_callback(0.7, "Analyzing classroom acoustics and speech activity...")
            audio_results = self.audio.analyze_audio(audio_path)
            # Save audio metrics to database
            self.db.save_audio_metrics(
                session_id=session_id,
                speech_ratio=audio_results["speech_ratio"],
                silence_ratio=audio_results["silence_ratio"],
                avg_rms_energy=audio_results["avg_rms_energy"],
                spectral_centroid_mean=audio_results["spectral_centroid_mean"],
                discussion_intensity_score=audio_results["discussion_intensity_score"],
                turn_taking_events=audio_results["turn_taking_events"],
                audio_duration_sec=audio_results["duration_sec"],
            )

        # 3. Multimodal Fusion for Attendance
        if progress_callback:
            progress_callback(0.85, "Synthesizing multimodal evidence and verifying presence...")

        attendance_evidence_list = self.fusion.fuse_session_attendance(
            video_results=video_results,
            audio_results=audio_results,
            student_roster=roster,
        )

        # 4. Save Attendance Records (Duplicate-Safe)
        for ev in attendance_evidence_list:
            self.db.record_attendance(
                session_id=session_id,
                student_id=ev.student_id,
                status=ev.status,
                confidence_score=ev.confidence_score,
                observation_duration_sec=ev.observation_duration_sec,
                frames_detected=ev.frames_detected,
                marking_source="AI_AUTOMATED",
                explanation=ev.explanation,
            )

        # 5. Multimodal Engagement Timeline
        v_bins = video_results.get("temporal_bins", [])
        a_bins = audio_results.get("temporal_bins", []) if audio_results else None
        fused_timeline = self.fusion.fuse_temporal_engagement(
            video_bins=v_bins,
            audio_bins=a_bins,
            expected_class_size=max(1, len(roster)),
        )

        if fused_timeline:
            self.db.save_engagement_logs_batch(session_id, fused_timeline)

        advisory_tips = self.fusion.generate_pedagogical_advisory(fused_timeline)

        if progress_callback:
            progress_callback(1.0, "Attendance and multimodal analysis complete!")

        return {
            "session_id": session_id,
            "attendance_evidence": attendance_evidence_list,
            "audio_metrics": audio_results,
            "temporal_timeline": fused_timeline,
            "pedagogical_advisory": advisory_tips,
            "preview_frames": video_results.get("preview_frames", []),
        }

    def manual_override_student(
        self,
        session_id: str,
        student_id: str,
        new_status: str,
        note: str = "Manual instructor review",
    ) -> bool:
        """Apply manual teacher correction to a student's session attendance."""
        return self.db.manual_override_attendance(
            session_id=session_id,
            student_id=student_id,
            new_status=new_status,
            note=note,
        )

    def get_session_attendance(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve complete session attendance register."""
        return self.db.get_session_attendance(session_id)

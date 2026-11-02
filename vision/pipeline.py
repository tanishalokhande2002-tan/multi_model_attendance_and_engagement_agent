"""
Vision Processing Pipeline.
Combines face detection, persistent centroid tracking, face recognition,
and temporal presence evaluation for video files or live camera inputs.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import cv2
import numpy as np

from .face_detector import FaceDetector
from .tracker import CentroidTracker, TrackedFace
from .recognizer import FaceRecognizer
from .attendance import AttendanceManager, AttendanceRecord

logger = logging.getLogger(__name__)


class VisionPipeline:
    """
    High-level orchestrator for video frame processing and presence evidence extraction.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        tracker: Optional[CentroidTracker] = None,
        recognizer: Optional[FaceRecognizer] = None,
        attendance_mgr: Optional[AttendanceManager] = None,
        detection_interval: int = 2,
    ):
        self.detector = detector or FaceDetector()
        self.tracker = tracker or CentroidTracker()
        self.recognizer = recognizer or FaceRecognizer()
        self.attendance_mgr = attendance_mgr or AttendanceManager()
        self.detection_interval = max(1, detection_interval)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        student_roster: Optional[List[Dict[str, Any]]] = None,
        is_detection_frame: bool = True,
    ) -> Tuple[np.ndarray, List[TrackedFace]]:
        """
        Process a single image or video frame.
        Annotates detected faces, updates tracks, and performs recognition.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        if is_detection_frame:
            # Run face detection
            boxes = self.detector.detect_faces(frame)
            active_tracks = self.tracker.update(boxes, timestamp_sec)
        else:
            # Update tracker without new detections
            active_tracks = self.tracker.update([], timestamp_sec)

        # Process each active track
        for track in active_tracks:
            x, y, bw, bh = track.bbox

            # Crop face for identification if not yet identified or periodically
            if student_roster and (track.matched_student_id is None or track.frame_count % 15 == 0):
                # Ensure valid crop boundaries
                cx1 = max(0, x)
                cy1 = max(0, y)
                cx2 = min(w, x + bw)
                cy2 = min(h, y + bh)

                if (cx2 - cx1) >= 20 and (cy2 - cy1) >= 20:
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    emb = self.recognizer.extract_embedding(face_crop)
                    if emb:
                        track.face_encoding = emb
                        matched_id, matched_name, conf = self.recognizer.match_face_to_roster(
                            emb, student_roster, strict_consent=True
                        )
                        if matched_id:
                            track.matched_student_id = matched_id
                            track.matched_student_name = matched_name
                            track.match_confidence = conf

            # Annotate bounding box
            color = (46, 204, 113) if track.matched_student_id else (52, 152, 219)  # Green if matched, blue if track
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            # Draw label banner
            label = track.matched_student_name or f"Track #{track.track_id}"
            conf_txt = f"{track.match_confidence * 100:.0f}%" if track.matched_student_id else f"{track.total_visible_sec:.1f}s"
            display_text = f"{label} ({conf_txt})"

            (tw, th), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(
                annotated,
                display_text,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated, active_tracks

    def process_video_file(
        self,
        video_path: str | Path,
        student_roster: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_duration_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process an entire video recording and generate presence and engagement statistics.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        if max_duration_sec and max_duration_sec < duration_sec:
            duration_sec = max_duration_sec
            total_frames = int(duration_sec * fps)

        temporal_bins: List[Dict[str, Any]] = []
        preview_frames: List[np.ndarray] = []
        frame_idx = 0

        # Reset tracker for new session
        self.tracker = CentroidTracker()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_idx / fps
                if max_duration_sec and timestamp_sec > max_duration_sec:
                    break

                is_det = (frame_idx % self.detection_interval == 0)
                annotated, active_tracks = self.process_frame(
                    frame=frame,
                    timestamp_sec=timestamp_sec,
                    student_roster=student_roster,
                    is_detection_frame=is_det,
                )

                # Collect temporal engagement stats every 1 second
                if int(timestamp_sec) > len(temporal_bins):
                    avg_movement = float(np.mean([t.movement_energy for t in active_tracks])) if active_tracks else 0.0
                    temporal_bins.append({
                        "timestamp_sec": float(len(temporal_bins)),
                        "faces_detected": len(active_tracks),
                        "movement_energy": round(avg_movement, 2),
                    })

                # Keep up to 6 preview frames across the video
                sample_interval = max(1, total_frames // 6)
                if frame_idx % sample_interval == 0 and len(preview_frames) < 6:
                    preview_frames.append(annotated)

                frame_idx += 1
                if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
                    prog = min(1.0, frame_idx / total_frames)
                    progress_callback(prog, f"Processing video frame {frame_idx}/{total_frames} ({prog * 100:.0f}%)")

        finally:
            cap.release()

        # Final evaluation of all tracked faces
        final_tracks = list(self.tracker.tracks.values())
        attendance_eval = self.attendance_mgr.evaluate_all(final_tracks)

        return {
            "video_path": str(video_path),
            "fps": round(fps, 2),
            "total_frames_processed": frame_idx,
            "duration_sec": round(duration_sec, 2),
            "tracks": final_tracks,
            "attendance_records": attendance_eval,
            "temporal_bins": temporal_bins,
            "preview_frames": preview_frames,
        }

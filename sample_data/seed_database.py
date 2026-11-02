"""
Seed script to populate SQLite with fictional students, classes, attendance,
and engagement logs for immediate out-of-the-box demonstration.
"""

import json
import logging
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

JSON_PATH = Path(__file__).resolve().parent / "fictional_students.json"


def generate_synthetic_embedding(seed_text: str, dim: int = 128) -> List[float]:
    """
    Generate a deterministic, normalized 128-d pseudo-embedding from a seed string.
    Ensures identical embeddings for the same student ID across runs.
    """
    # Use sha256 hashes to generate pseudo-random normal numbers
    hash_digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    np.random.seed(int.from_bytes(hash_digest[:4], "big"))
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return [round(float(x), 6) for x in vec]


def seed_all(db: Optional[DatabaseManager] = None) -> DatabaseManager:
    """Populate full sample dataset into database."""
    if db is None:
        db = DatabaseManager()

    print("[1/4] Seeding fictional students...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        students = json.load(f)

    for s in students:
        existing = db.get_student(s["student_id"])
        if not existing:
            # Generate deterministic synthetic embedding if consented
            encoding = generate_synthetic_embedding(s["student_id"]) if s.get("consent_enrolled") else None
            db.add_student(
                student_id=s["student_id"],
                name=s["name"],
                email=s.get("email", ""),
                grade=s.get("grade", ""),
                section=s.get("section", ""),
                consent_enrolled=s.get("consent_enrolled", 0),
                guardian_consent=s.get("guardian_consent", 0),
                face_encoding=encoding,
            )

    print("[2/4] Seeding historical classroom sessions...")
    sessions_data = [
        {
            "id": "SES-101",
            "name": "Computer Science 101: Data Structures",
            "instructor": "Dr. Alan Turing",
            "grade": "Grade 10",
            "sec": "A",
            "days_ago": 4,
            "duration": 45.0,
            "speech_ratio": 0.58,
            "intensity": 0.72,
        },
        {
            "id": "SES-102",
            "name": "Physics 201: Newtonian Mechanics",
            "instructor": "Dr. Marie Curie",
            "grade": "Grade 10",
            "sec": "A",
            "days_ago": 3,
            "duration": 50.0,
            "speech_ratio": 0.44,
            "intensity": 0.65,
        },
        {
            "id": "SES-103",
            "name": "Literature 105: Shakespearean Drama",
            "instructor": "Prof. Maya Angelou",
            "grade": "Grade 10",
            "sec": "B",
            "days_ago": 2,
            "duration": 40.0,
            "speech_ratio": 0.70,
            "intensity": 0.82,
        },
        {
            "id": "SES-104",
            "name": "Mathematics 301: Multivariable Calculus",
            "instructor": "Prof. Carl Gauss",
            "grade": "Grade 10",
            "sec": "A",
            "days_ago": 1,
            "duration": 55.0,
            "speech_ratio": 0.32,
            "intensity": 0.48,
        },
        {
            "id": "SES-105",
            "name": "Biology 202: Genetics & Heredity",
            "instructor": "Dr. Gregor Mendel",
            "grade": "Grade 10",
            "sec": "B",
            "days_ago": 0,
            "duration": 45.0,
            "speech_ratio": 0.52,
            "intensity": 0.68,
        },
    ]

    now = datetime.now()
    for s_info in sessions_data:
        sess_date = (now - timedelta(days=s_info["days_ago"])).strftime("%Y-%m-%d")
        existing_sess = db.get_session(s_info["id"])
        if not existing_sess:
            db.create_session(
                session_id=s_info["id"],
                course_name=s_info["name"],
                instructor_name=s_info["instructor"],
                grade=s_info["grade"],
                section=s_info["sec"],
                session_date=sess_date,
                start_time="09:00 AM",
                end_time="09:50 AM",
                video_source="classroom_cam_1.mp4",
                audio_source="mic_array_1.wav",
                duration_sec=s_info["duration"] * 60.0,
            )

            # Audio metrics
            db.save_audio_metrics(
                session_id=s_info["id"],
                speech_ratio=s_info["speech_ratio"],
                silence_ratio=round(1.0 - s_info["speech_ratio"], 2),
                avg_rms_energy=0.035,
                spectral_centroid_mean=1850.0,
                discussion_intensity_score=s_info["intensity"],
                turn_taking_events=int(s_info["speech_ratio"] * 24),
                audio_duration_sec=s_info["duration"] * 60.0,
            )

    print("[3/4] Seeding attendance records...")
    # Map attendance patterns
    status_matrix = {
        "STU-1001": ["Present", "Present", "Present", "Present", "Late"],
        "STU-1002": ["Present", "Present", "Late", "Present", "Present"],
        "STU-1003": ["Present", "Absent", "Present", "Present", "Present"],
        "STU-1004": ["Present", "Present", "Present", "Present", "Present"],
        "STU-1005": ["Late", "Present", "Present", "Excused", "Present"],
        "STU-1006": ["Present", "Present", "Present", "Present", "Present"],
        "STU-1007": ["Present", "Late", "Present", "Present", "Present"],
        "STU-1008": ["Present", "Present", "Absent", "Present", "Late"],
        "STU-1009": ["Absent", "Absent", "Absent", "Absent", "Absent"],  # Non-consented visitor
        "STU-1010": ["Present", "Present", "Present", "Present", "Present"],
    }

    session_ids = [s["id"] for s in sessions_data]
    for s_idx, sess_id in enumerate(session_ids):
        for stu_id, statuses in status_matrix.items():
            status = statuses[s_idx % len(statuses)]
            conf = 0.88 if status == "Present" else (0.82 if status == "Late" else 0.0)
            dur = 42.5 if status == "Present" else (28.0 if status == "Late" else 0.0)
            frames = int(dur * 2)

            expl = f"Autonomous presence recorded with confidence {conf * 100:.0f}%" if status != "Absent" else "Unobserved"
            db.record_attendance(
                session_id=sess_id,
                student_id=stu_id,
                status=status,
                confidence_score=conf,
                observation_duration_sec=dur,
                frames_detected=frames,
                marking_source="AI_AUTOMATED",
                explanation=expl,
            )

    print("[4/4] Seeding engagement timeline logs...")
    # Generate 30-minute simulated engagement points (1 point per 30 sec = 60 points)
    for sess_id in session_ids:
        logs = []
        for step in range(30):
            t_sec = step * 30.0
            faces = int(np.clip(np.random.normal(7.5, 1.2), 4, 9))
            mov = round(float(np.random.uniform(5.0, 22.0)), 2)
            speech = 1 if np.random.rand() > 0.4 else 0
            score = round(float(0.55 + 0.35 * np.sin(step / 4.0) ** 2 + 0.1 * np.random.rand()), 3)
            state = "Interactive Discussion" if speech == 1 else "Attentive Lecture"

            logs.append({
                "timestamp_sec": t_sec,
                "faces_detected": faces,
                "movement_energy": mov,
                "speech_active": speech,
                "aggregate_engagement_score": min(1.0, score),
                "pedagogical_state": state,
            })
        db.save_engagement_logs_batch(sess_id, logs)

    print("[OK] Database populated with full fictional sample roster and 5 historical sessions.")
    return db


if __name__ == "__main__":
    seed_all()

"""
Database Manager for Multimodal Attendance & Engagement Agent.
Provides thread-safe SQLite operations, CRUD routines, and privacy compliance utilities.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "attendance_agent.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseManager:
    """
    Manages SQLite connections, schema initialization, and transactional queries.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create a connection with row factory enabled and foreign keys enforced."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize tables and indexes from schema.sql."""
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                # Seed default settings if empty
                conn.execute("""
                    INSERT OR IGNORE INTO system_settings (key, value, description)
                    VALUES 
                    ('min_presence_duration_sec', '5.0', 'Minimum observation seconds for automated presence candidate'),
                    ('face_match_confidence_threshold', '0.65', 'Minimum cosine similarity threshold for face match'),
                    ('data_retention_days', '90', 'Biometric & video observation retention window in days'),
                    ('enforce_consent_strict', '1', 'If 1, students without consent are never processed for facial identification');
                """)
                conn.commit()
            logger.info("Database initialized successfully at: %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            raise

    # =========================================================================
    # Student Operations
    # =========================================================================

    def add_student(
        self,
        student_id: str,
        name: str,
        email: str = "",
        grade: str = "",
        section: str = "",
        consent_enrolled: int = 0,
        guardian_consent: int = 0,
        face_encoding: Optional[List[float]] = None,
        photo_path: str = "",
    ) -> bool:
        """Insert a new student record with optional face encoding."""
        now = datetime.now(timezone.utc).isoformat()
        encoding_json = json.dumps(face_encoding) if face_encoding else None
        consent_date = now if consent_enrolled else None

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO students (
                    student_id, name, email, grade, section, consent_enrolled,
                    consent_date, guardian_consent, face_encoding, photo_path,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    name,
                    email,
                    grade,
                    section,
                    int(consent_enrolled),
                    consent_date,
                    int(guardian_consent),
                    encoding_json,
                    photo_path,
                    now,
                    now,
                ),
            )
            conn.commit()

        self.log_audit_event("CREATE", "STUDENT", student_id, f"Added student {name} (Consent: {bool(consent_enrolled)})")
        return True

    def update_student(
        self,
        student_id: str,
        name: str,
        email: str,
        grade: str,
        section: str,
        consent_enrolled: int,
        guardian_consent: int = 0,
    ) -> bool:
        """Update student details and consent status."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE students
                SET name = ?, email = ?, grade = ?, section = ?,
                    consent_enrolled = ?, guardian_consent = ?, updated_at = ?
                WHERE student_id = ?
                """,
                (name, email, grade, section, int(consent_enrolled), int(guardian_consent), now, student_id),
            )
            conn.commit()

        self.log_audit_event("UPDATE", "STUDENT", student_id, f"Updated profile & consent for {name}")
        return True

    def update_student_face_encoding(self, student_id: str, face_encoding: List[float], photo_path: str = "") -> bool:
        """Store or update face encoding for an authorized student."""
        now = datetime.now(timezone.utc).isoformat()
        encoding_json = json.dumps(face_encoding)
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE students
                SET face_encoding = ?, photo_path = ?, updated_at = ?
                WHERE student_id = ?
                """,
                (encoding_json, photo_path, now, student_id),
            )
            conn.commit()

        self.log_audit_event("ENROLL_BIOMETRIC", "STUDENT", student_id, "Updated face encoding representation")
        return True

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a student by ID."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("face_encoding"):
                try:
                    res["face_encoding"] = json.loads(res["face_encoding"])
                except Exception:
                    res["face_encoding"] = None
            return res

    def get_all_students(self, grade: Optional[str] = None, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all registered students with optional grade/section filter."""
        query = "SELECT * FROM students WHERE 1=1"
        params: List[Any] = []
        if grade:
            query += " AND grade = ?"
            params.append(grade)
        if section:
            query += " AND section = ?"
            params.append(section)
        query += " ORDER BY name ASC"

        with self.get_connection() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("face_encoding"):
                    try:
                        item["face_encoding"] = json.loads(item["face_encoding"])
                    except Exception:
                        item["face_encoding"] = None
                results.append(item)
            return results

    def delete_student(self, student_id: str) -> bool:
        """Delete student record and all cascading attendance records."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
            conn.commit()
        self.log_audit_event("DELETE", "STUDENT", student_id, "Deleted student profile and associated data")
        return True

    # =========================================================================
    # Session Operations
    # =========================================================================

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
        duration_sec: float = 0.0,
        status: str = "Completed",
    ) -> bool:
        """Create a new classroom session."""
        now = datetime.now(timezone.utc).isoformat()
        date_str = session_date or datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, course_name, instructor_name, grade, section,
                    session_date, start_time, end_time, video_source, audio_source,
                    duration_sec, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    course_name,
                    instructor_name,
                    grade,
                    section,
                    date_str,
                    start_time,
                    end_time,
                    video_source,
                    audio_source,
                    duration_sec,
                    status,
                    now,
                ),
            )
            conn.commit()
        self.log_audit_event("CREATE", "SESSION", session_id, f"Created session '{course_name}' on {date_str}")
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session record by ID."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent classroom sessions."""
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM sessions ORDER BY session_date DESC, created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        """Delete session and associated metrics/attendance."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        self.log_audit_event("DELETE", "SESSION", session_id, "Deleted session and associated logs")
        return True

    # =========================================================================
    # Attendance Operations (Duplicate-Safe)
    # =========================================================================

    def record_attendance(
        self,
        session_id: str,
        student_id: str,
        status: str,
        confidence_score: float = 0.0,
        observation_duration_sec: float = 0.0,
        frames_detected: int = 0,
        marking_source: str = "AI_AUTOMATED",
        explanation: str = "",
        manual_override_note: str = "",
    ) -> bool:
        """
        Record or update attendance for a student in a session.
        Uses INSERT OR REPLACE to prevent duplicate records for the same (session_id, student_id).
        """
        now = datetime.now(timezone.utc).isoformat()
        attendance_id = f"{session_id}_{student_id}"

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO attendance (
                    attendance_id, session_id, student_id, status, confidence_score,
                    observation_duration_sec, frames_detected, marking_source,
                    explanation, manual_override_note, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, student_id) DO UPDATE SET
                    status = excluded.status,
                    confidence_score = excluded.confidence_score,
                    observation_duration_sec = excluded.observation_duration_sec,
                    frames_detected = excluded.frames_detected,
                    marking_source = excluded.marking_source,
                    explanation = excluded.explanation,
                    manual_override_note = excluded.manual_override_note,
                    timestamp = excluded.timestamp
                """,
                (
                    attendance_id,
                    session_id,
                    student_id,
                    status,
                    float(confidence_score),
                    float(observation_duration_sec),
                    int(frames_detected),
                    marking_source,
                    explanation,
                    manual_override_note,
                    now,
                ),
            )
            conn.commit()
        return True

    def manual_override_attendance(
        self,
        session_id: str,
        student_id: str,
        new_status: str,
        note: str = "Manual teacher correction",
    ) -> bool:
        """Explicitly override a student's attendance record."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE attendance
                SET status = ?, marking_source = 'TEACHER_MANUAL',
                    manual_override_note = ?, timestamp = ?
                WHERE session_id = ? AND student_id = ?
                """,
                (new_status, note, now, session_id, student_id),
            )
            conn.commit()

        self.log_audit_event(
            "OVERRIDE_ATTENDANCE",
            "ATTENDANCE",
            f"{session_id}_{student_id}",
            f"Changed status to '{new_status}' with note: {note}",
        )
        return True

    def get_session_attendance(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all attendance entries for a session joined with student metadata."""
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT a.*, s.name, s.grade, s.section, s.email, s.consent_enrolled
                FROM attendance a
                JOIN students s ON a.student_id = s.student_id
                WHERE a.session_id = ?
                ORDER BY s.name ASC
                """,
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_student_attendance_history(self, student_id: str) -> List[Dict[str, Any]]:
        """Retrieve complete session attendance history for an individual student."""
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT a.*, s.course_name, s.session_date, s.start_time
                FROM attendance a
                JOIN sessions s ON a.session_id = s.session_id
                WHERE a.student_id = ?
                ORDER BY s.session_date DESC
                """,
                (student_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Audio Metrics Operations
    # =========================================================================

    def save_audio_metrics(
        self,
        session_id: str,
        speech_ratio: float,
        silence_ratio: float,
        avg_rms_energy: float,
        spectral_centroid_mean: float,
        discussion_intensity_score: float,
        turn_taking_events: int = 0,
        audio_duration_sec: float = 0.0,
    ) -> bool:
        """Save or update classroom audio analysis metrics."""
        now = datetime.now(timezone.utc).isoformat()
        metric_id = f"audio_{session_id}"

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audio_metrics (
                    metric_id, session_id, speech_ratio, silence_ratio, avg_rms_energy,
                    spectral_centroid_mean, discussion_intensity_score, turn_taking_events,
                    audio_duration_sec, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    speech_ratio = excluded.speech_ratio,
                    silence_ratio = excluded.silence_ratio,
                    avg_rms_energy = excluded.avg_rms_energy,
                    spectral_centroid_mean = excluded.spectral_centroid_mean,
                    discussion_intensity_score = excluded.discussion_intensity_score,
                    turn_taking_events = excluded.turn_taking_events,
                    audio_duration_sec = excluded.audio_duration_sec,
                    analyzed_at = excluded.analyzed_at
                """,
                (
                    metric_id,
                    session_id,
                    float(speech_ratio),
                    float(silence_ratio),
                    float(avg_rms_energy),
                    float(spectral_centroid_mean),
                    float(discussion_intensity_score),
                    int(turn_taking_events),
                    float(audio_duration_sec),
                    now,
                ),
            )
            conn.commit()
        return True

    def get_audio_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve acoustic metrics for a given session."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM audio_metrics WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Engagement Timeline Operations
    # =========================================================================

    def save_engagement_logs_batch(self, session_id: str, logs: List[Dict[str, Any]]) -> bool:
        """Batch insert temporal engagement bins."""
        if not logs:
            return True

        records = []
        for i, log in enumerate(logs):
            log_id = f"{session_id}_t{int(log['timestamp_sec'])}_{i}"
            records.append((
                log_id,
                session_id,
                float(log["timestamp_sec"]),
                int(log.get("faces_detected", 0)),
                float(log.get("movement_energy", 0.0)),
                int(log.get("speech_active", 0)),
                float(log.get("aggregate_engagement_score", 0.0)),
                str(log.get("pedagogical_state", "Attentive")),
            ))

        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO engagement_logs (
                    log_id, session_id, timestamp_sec, faces_detected,
                    movement_energy, speech_active, aggregate_engagement_score,
                    pedagogical_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()
        return True

    def get_session_engagement_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve temporal engagement timeline for a session."""
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM engagement_logs WHERE session_id = ? ORDER BY timestamp_sec ASC",
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Privacy, Audit & Retention Routines
    # =========================================================================

    def log_audit_event(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        details: str,
        performed_by: str = "Teacher",
    ) -> None:
        """Append an entry to the privacy audit trail."""
        now = datetime.now(timezone.utc).isoformat()
        audit_id = f"aud_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO privacy_audit_logs (
                        audit_id, action, entity_type, entity_id, details, performed_by, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (audit_id, action, entity_type, entity_id, details, performed_by, now),
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to write audit log: %s", e)

    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent privacy audit entries."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM privacy_audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def purge_biometrics_for_student(self, student_id: str) -> bool:
        """Purge biometric face encoding and photo for a student when consent is revoked."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE students
                SET face_encoding = NULL, photo_path = NULL, consent_enrolled = 0, updated_at = ?
                WHERE student_id = ?
                """,
                (now, student_id),
            )
            conn.commit()
        self.log_audit_event("PURGE_BIOMETRICS", "STUDENT", student_id, "Biometric face encoding purged on consent revocation")
        return True

    def purge_expired_records(self, retention_days: int = 90) -> int:
        """Purge sessions and engagement logs older than retention threshold."""
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM sessions
                WHERE julianday('now') - julianday(session_date) > ?
                """,
                (retention_days,),
            )
            deleted_count = cur.rowcount
            conn.commit()

        self.log_audit_event("DATA_RETENTION_PURGE", "SESSIONS", "SYSTEM", f"Purged {deleted_count} sessions older than {retention_days} days")
        return deleted_count

    # =========================================================================
    # Aggregates & Dashboard Stats
    # =========================================================================

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Compute high-level summary KPIs for the dashboard."""
        with self.get_connection() as conn:
            total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            consented_students = conn.execute("SELECT COUNT(*) FROM students WHERE consent_enrolled = 1").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

            att_row = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_records,
                    SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_count,
                    SUM(CASE WHEN status = 'Late' THEN 1 ELSE 0 END) as late_count
                FROM attendance
                """
            ).fetchone()

            total_records = att_row["total_records"] or 0
            present_count = att_row["present_count"] or 0
            late_count = att_row["late_count"] or 0

            attendance_pct = ((present_count + late_count) / total_records * 100.0) if total_records > 0 else 0.0

            avg_eng_row = conn.execute(
                "SELECT AVG(aggregate_engagement_score) as avg_score FROM engagement_logs"
            ).fetchone()
            avg_engagement = round(float(avg_eng_row["avg_score"] or 0.72) * 100.0, 1)

            return {
                "total_students": total_students,
                "consented_students": consented_students,
                "total_sessions": total_sessions,
                "overall_attendance_pct": round(attendance_pct, 1),
                "avg_engagement_pct": avg_engagement,
            }

-- Multimodal Attendance & Engagement Agent - SQLite Database Schema

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    grade TEXT,
    section TEXT,
    consent_enrolled INTEGER DEFAULT 0, -- 1 = Consented & Authorized, 0 = Not Consented
    consent_date TEXT,
    guardian_consent INTEGER DEFAULT 0,
    face_encoding TEXT,                 -- JSON serialized float vector
    photo_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    instructor_name TEXT,
    grade TEXT,
    section TEXT,
    session_date TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    video_source TEXT,
    audio_source TEXT,
    duration_sec REAL DEFAULT 0.0,
    status TEXT DEFAULT 'Completed',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    attendance_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Present', 'Absent', 'Late', 'Excused')),
    confidence_score REAL DEFAULT 0.0,
    observation_duration_sec REAL DEFAULT 0.0,
    frames_detected INTEGER DEFAULT 0,
    marking_source TEXT DEFAULT 'AI_AUTOMATED' CHECK(marking_source IN ('AI_AUTOMATED', 'TEACHER_MANUAL', 'HYBRID_VERIFIED')),
    explanation TEXT,
    manual_override_note TEXT,
    timestamp TEXT NOT NULL,
    UNIQUE(session_id, student_id),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audio_metrics (
    metric_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    speech_ratio REAL DEFAULT 0.0,
    silence_ratio REAL DEFAULT 0.0,
    avg_rms_energy REAL DEFAULT 0.0,
    spectral_centroid_mean REAL DEFAULT 0.0,
    discussion_intensity_score REAL DEFAULT 0.0,
    turn_taking_events INTEGER DEFAULT 0,
    audio_duration_sec REAL DEFAULT 0.0,
    analyzed_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS engagement_logs (
    log_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp_sec REAL NOT NULL,
    faces_detected INTEGER DEFAULT 0,
    movement_energy REAL DEFAULT 0.0,
    speech_active INTEGER DEFAULT 0,
    aggregate_engagement_score REAL DEFAULT 0.0,
    pedagogical_state TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS privacy_audit_logs (
    audit_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details TEXT,
    performed_by TEXT DEFAULT 'Teacher',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_engagement_session ON engagement_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_students_grade_sec ON students(grade, section);

"""
Unit tests for DatabaseManager and SQLite schema constraints.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing an isolated temporary database for testing."""
    db_file = tmp_path / "test_attendance.db"
    return DatabaseManager(db_path=db_file)


def test_student_crud_and_consent(temp_db):
    """Test student creation, retrieval, and consent management."""
    # 1. Add student with consent
    added = temp_db.add_student(
        student_id="TEST-001",
        name="Test Student",
        email="test@example.com",
        grade="Grade 10",
        section="A",
        consent_enrolled=1,
        face_encoding=[0.1] * 128,
    )
    assert added is True

    # 2. Retrieve student
    s = temp_db.get_student("TEST-001")
    assert s is not None
    assert s["name"] == "Test Student"
    assert s["consent_enrolled"] == 1
    assert len(s["face_encoding"]) == 128

    # 3. Purge biometric upon consent revocation
    temp_db.purge_biometrics_for_student("TEST-001")
    updated_s = temp_db.get_student("TEST-001")
    assert updated_s["consent_enrolled"] == 0
    assert updated_s["face_encoding"] is None


def test_attendance_anti_duplicate_guarantee(temp_db):
    """Ensure duplicate attendance records cannot be created for the same session and student."""
    temp_db.add_student(student_id="TEST-002", name="Jane Doe", consent_enrolled=1)
    temp_db.create_session(session_id="SESS-001", course_name="Testing 101")

    # Record attendance initial
    temp_db.record_attendance(
        session_id="SESS-001",
        student_id="TEST-002",
        status="Present",
        confidence_score=0.85,
        observation_duration_sec=12.0,
    )

    # Re-recording with new status should update without adding a duplicate row
    temp_db.record_attendance(
        session_id="SESS-001",
        student_id="TEST-002",
        status="Late",
        confidence_score=0.90,
        observation_duration_sec=15.0,
    )

    records = temp_db.get_session_attendance("SESS-001")
    assert len(records) == 1
    assert records[0]["status"] == "Late"
    assert records[0]["confidence_score"] == 0.90


def test_manual_override_and_audit_trail(temp_db):
    """Test teacher manual override and audit log persistence."""
    temp_db.add_student(student_id="TEST-003", name="John Smith", consent_enrolled=1)
    temp_db.create_session(session_id="SESS-002", course_name="Ethics in AI")

    temp_db.record_attendance(
        session_id="SESS-002",
        student_id="TEST-003",
        status="Absent",
    )

    # Teacher overrides to Present
    temp_db.manual_override_attendance(
        session_id="SESS-002",
        student_id="TEST-003",
        new_status="Present",
        note="Verified in class via sign-in sheet",
    )

    record = temp_db.get_session_attendance("SESS-002")[0]
    assert record["status"] == "Present"
    assert record["marking_source"] == "TEACHER_MANUAL"
    assert "sign-in sheet" in record["manual_override_note"]

    # Verify audit trail contains the action
    audit_logs = temp_db.get_audit_logs()
    override_entries = [log for log in audit_logs if log["action"] == "OVERRIDE_ATTENDANCE"]
    assert len(override_entries) > 0

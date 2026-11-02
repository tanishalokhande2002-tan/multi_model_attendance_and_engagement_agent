"""
Privacy, Consent, and Data Retention Service.
Enforces strict consent checks, immediate biometric purging on revocation,
configurable data-retention schedules, and transparent audit logging.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class PrivacyService:
    """
    Ensures ethical AI compliance with student privacy regulations (e.g. FERPA, GDPR).
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def check_biometric_eligibility(self, student_id: str) -> bool:
        """
        Verify if student has explicit consent on file before allowing biometric matching.
        """
        student = self.db.get_student(student_id)
        if not student:
            return False
        return bool(student.get("consent_enrolled", 0))

    def revoke_consent_and_purge_biometrics(self, student_id: str) -> bool:
        """
        Revoke consent and permanently purge face encodings and biometric artifacts.
        """
        return self.db.purge_biometrics_for_student(student_id)

    def execute_retention_purge(self, retention_days: int = 90) -> int:
        """
        Purge session records and temporal observation bins older than the retention threshold.
        """
        return self.db.purge_expired_records(retention_days=retention_days)

    def get_audit_trail_df(self, limit: int = 200) -> pd.DataFrame:
        """
        Retrieve formatted privacy and audit log history.
        """
        logs = self.db.get_audit_logs(limit=limit)
        if not logs:
            return pd.DataFrame(columns=["Timestamp", "Action", "Entity Type", "Entity ID", "Details", "Performed By"])

        records = []
        for l in logs:
            records.append({
                "Timestamp": l["timestamp"],
                "Action": l["action"],
                "Entity Type": l["entity_type"],
                "Entity ID": l["entity_id"],
                "Details": l.get("details", ""),
                "Performed By": l.get("performed_by", "Teacher"),
            })
        return pd.DataFrame(records)

    @staticmethod
    def anonymize_attendance_export(df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a de-identified version of attendance records for ethical pedagogical research.
        Replaces student names and IDs with cryptographic pseudo-tokens.
        """
        if df.empty:
            return df

        anon_df = df.copy()
        if "Name" in anon_df.columns:
            anon_df["Name"] = anon_df["Name"].apply(
                lambda n: f"Participant_{hashlib.sha256(str(n).encode()).hexdigest()[:6]}"
            )
        if "Student ID" in anon_df.columns:
            anon_df["Student ID"] = anon_df["Student ID"].apply(
                lambda sid: f"ID_{hashlib.sha256(str(sid).encode()).hexdigest()[:8]}"
            )
        return anon_df

"""
Reporting and CSV Export Service.
Generates student-wise attendance aggregates, session-level registers,
and exportable CSV dataframes.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class ReportingService:
    """
    Computes attendance rates, aggregations, and tabular reports for export.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_student_attendance_summary(
        self,
        grade: Optional[str] = None,
        section: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate aggregate student attendance statistics across all completed sessions.
        """
        students = self.db.get_all_students(grade=grade, section=section)
        if not students:
            return pd.DataFrame(columns=[
                "Student ID", "Name", "Grade", "Section", "Consent",
                "Total Classes", "Present", "Late", "Absent", "Excused", "Attendance Rate %"
            ])

        records = []
        for s in students:
            history = self.db.get_student_attendance_history(s["student_id"])
            total = len(history)
            p_cnt = sum(1 for r in history if r["status"] == "Present")
            l_cnt = sum(1 for r in history if r["status"] == "Late")
            a_cnt = sum(1 for r in history if r["status"] == "Absent")
            e_cnt = sum(1 for r in history if r["status"] == "Excused")

            rate = round(((p_cnt + l_cnt) / total * 100.0), 1) if total > 0 else 0.0

            records.append({
                "Student ID": s["student_id"],
                "Name": s["name"],
                "Grade": s.get("grade", "-"),
                "Section": s.get("section", "-"),
                "Consent": "Authorized" if s.get("consent_enrolled") else "Withheld",
                "Total Classes": total,
                "Present": p_cnt,
                "Late": l_cnt,
                "Absent": a_cnt,
                "Excused": e_cnt,
                "Attendance Rate %": rate,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(by=["Attendance Rate %", "Name"], ascending=[False, True])
        return df

    def get_session_attendance_report(self, session_id: str) -> pd.DataFrame:
        """
        Generate tabular attendance roster for a specific session with confidence and explanations.
        """
        entries = self.db.get_session_attendance(session_id)
        if not entries:
            return pd.DataFrame(columns=[
                "Student ID", "Name", "Status", "Confidence %",
                "Duration (s)", "Marking Source", "Explanation", "Notes"
            ])

        records = []
        for r in entries:
            conf_pct = round(r["confidence_score"] * 100.0, 1) if r.get("confidence_score") else 0.0
            records.append({
                "Student ID": r["student_id"],
                "Name": r.get("name", "Unknown"),
                "Status": r["status"],
                "Confidence %": conf_pct,
                "Duration (s)": round(r.get("observation_duration_sec", 0.0), 1),
                "Marking Source": r.get("marking_source", "AI_AUTOMATED"),
                "Explanation": r.get("explanation", ""),
                "Notes": r.get("manual_override_note", ""),
            })

        return pd.DataFrame(records)

    def export_student_summary_csv(self, grade: Optional[str] = None, section: Optional[str] = None) -> str:
        """Export student summary table as CSV string."""
        df = self.get_student_attendance_summary(grade, section)
        return df.to_csv(index=False)

    def export_session_report_csv(self, session_id: str) -> str:
        """Export session roster table as CSV string."""
        df = self.get_session_attendance_report(session_id)
        return df.to_csv(index=False)

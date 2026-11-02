"""
Student Management Service.
Handles student registration, profile updates, consent tracking,
and biometric enrollment with validation.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from database.db_manager import DatabaseManager
from vision.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class StudentService:
    """
    Business logic layer for student records and biometric consent lifecycle.
    """

    def __init__(self, db: DatabaseManager, recognizer: Optional[FaceRecognizer] = None):
        self.db = db
        self.recognizer = recognizer or FaceRecognizer()

    def register_student(
        self,
        student_id: str,
        name: str,
        email: str = "",
        grade: str = "",
        section: str = "",
        consent_enrolled: bool = False,
        guardian_consent: bool = False,
        photo_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        Register a new student with validation and optional photo encoding.
        """
        student_id = student_id.strip()
        name = name.strip()

        if not student_id or not name:
            return False, "Student ID and Full Name are mandatory."

        # Check existing
        existing = self.db.get_student(student_id)
        if existing:
            return False, f"A student with ID '{student_id}' already exists."

        face_encoding = None
        photo_path = ""

        # Process biometric photo only if consent is given
        if photo_bgr is not None and consent_enrolled:
            emb = self.recognizer.extract_embedding(photo_bgr)
            if emb:
                face_encoding = emb
            else:
                return False, "Could not detect or encode a valid face from the provided reference photo."

        try:
            self.db.add_student(
                student_id=student_id,
                name=name,
                email=email.strip(),
                grade=grade.strip(),
                section=section.strip(),
                consent_enrolled=1 if consent_enrolled else 0,
                guardian_consent=1 if guardian_consent else 0,
                face_encoding=face_encoding,
                photo_path=photo_path,
            )
            return True, f"Student '{name}' ({student_id}) successfully registered."
        except Exception as e:
            logger.error("Failed to register student: %s", e)
            return False, f"Database error: {str(e)}"

    def update_student_consent(
        self,
        student_id: str,
        consent_enrolled: bool,
        guardian_consent: bool = False,
    ) -> Tuple[bool, str]:
        """
        Update consent status. If revoked, immediately purges biometric encoding.
        """
        student = self.db.get_student(student_id)
        if not student:
            return False, f"Student '{student_id}' not found."

        if not consent_enrolled:
            # Purge biometric data
            self.db.purge_biometrics_for_student(student_id)
            return True, f"Consent revoked. Biometric representations for '{student['name']}' have been permanently deleted."
        else:
            self.db.update_student(
                student_id=student_id,
                name=student["name"],
                email=student.get("email", ""),
                grade=student.get("grade", ""),
                section=student.get("section", ""),
                consent_enrolled=1,
                guardian_consent=1 if guardian_consent else 0,
            )
            return True, f"Consent updated to Authorized for '{student['name']}'."

    def enroll_face_from_image(self, student_id: str, photo_bgr: np.ndarray) -> Tuple[bool, str]:
        """
        Extract and associate biometric face embedding with an existing consented student.
        """
        student = self.db.get_student(student_id)
        if not student:
            return False, f"Student '{student_id}' not found."

        if not student.get("consent_enrolled", 0):
            return False, "Cannot enroll biometric representation without verified student/guardian consent."

        emb = self.recognizer.extract_embedding(photo_bgr)
        if not emb:
            return False, "Unable to detect a clear frontal face in the provided image. Please check illumination and pose."

        self.db.update_student_face_encoding(student_id, emb)
        return True, f"Biometric face encoding successfully enrolled for '{student['name']}'."

    def get_all_students(self, grade: Optional[str] = None, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve student roster."""
        return self.db.get_all_students(grade, section)

    def delete_student(self, student_id: str) -> Tuple[bool, str]:
        """Delete student and cascade delete attendance records."""
        student = self.db.get_student(student_id)
        if not student:
            return False, "Student not found."

        self.db.delete_student(student_id)
        return True, f"Student '{student['name']}' ({student_id}) deleted."

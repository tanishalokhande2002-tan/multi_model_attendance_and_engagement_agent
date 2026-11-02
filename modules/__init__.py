"""
Modules package for Multimodal Attendance & Engagement Agent.
"""

from .audio_module import AudioProcessor
from .multimodal_fusion import MultimodalFusionEngine, AttendanceEvidence
from .student_service import StudentService
from .attendance_service import AttendanceService
from .reporting_service import ReportingService
from .privacy_service import PrivacyService

__all__ = [
    "AudioProcessor",
    "MultimodalFusionEngine",
    "AttendanceEvidence",
    "StudentService",
    "AttendanceService",
    "ReportingService",
    "PrivacyService",
]

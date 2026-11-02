"""
Multimodal Attendance & Engagement Agent - Central Configuration
Stores all configurable thresholds, weights, and parameters for vision, audio, fusion, and analytics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class VisionConfig:
    """Configuration for video processing, face detection, and tracking."""
    # Detection settings
    detector_backend: str = "haar"  # 'haar' or 'dnn'
    min_face_size: tuple[int, int] = (30, 30)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    detection_interval: int = 2  # Run full detection every N frames for efficiency
    preview_annotated: bool = False

    # Tracking settings (Centroid + IoU tracking)
    max_disappeared_frames: int = 30  # Frames to retain lost track before deregistrating
    max_centroid_distance: float = 75.0  # Max pixel distance for centroid matching
    iou_match_threshold: float = 0.25  # Minimum IoU for association

    # Attendance qualification settings
    min_presence_duration_sec: float = 5.0  # Sustained presence needed for attendance candidate
    allowed_absence_gap_sec: float = 2.0   # Max allowable gap in seconds within sustained presence


@dataclass
class AudioConfig:
    """Configuration for audio loading, feature extraction, and activity detection."""
    target_sr: int = 16000  # Standard sample rate for speech/audio analysis
    frame_length: int = 2048
    hop_length: int = 512
    energy_percentile_cutoff: float = 35.0  # Dynamic noise floor estimation percentile
    silence_rms_threshold: float = 0.012    # Fallback absolute RMS floor for silence
    zcr_speech_threshold: float = 0.04      # Indicative zero-crossing rate threshold for speech band
    spectral_rolloff_percent: float = 0.85


@dataclass
class FusionConfig:
    """Configuration for multimodal time-synchronization and engagement scoring."""
    time_bin_sec: float = 1.0  # Aggregation window duration in seconds

    # Engagement scoring weights (Normalized sum = 1.0)
    presence_weight: float = 0.35
    movement_weight: float = 0.20
    audio_activity_weight: float = 0.30
    audio_variation_weight: float = 0.15

    # Thresholds for state indicators
    min_active_participants: int = 1
    movement_norm_scale: float = 50.0  # Normalizer for bounding box motion (pixels/sec)


@dataclass
class SuggestionConfig:
    """Configuration for rule-based pedagogical advisory suggestions."""
    quiet_window_sec: float = 15.0      # Duration of low audio to trigger question prompt
    active_window_sec: float = 20.0     # Duration of high audio to suggest pacing check
    high_variation_threshold: float = 0.28  # Engagement score volatility threshold
    low_data_quality_missing_ratio: float = 0.40  # Flag when >40% bins lack one modality


@dataclass
class AppConfig:
    """Master application configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")
    
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    suggestion: SuggestionConfig = field(default_factory=SuggestionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation for logging and export."""
        return {
            "vision": vars(self.vision),
            "audio": vars(self.audio),
            "fusion": vars(self.fusion),
            "suggestion": vars(self.suggestion),
            "paths": {
                "project_root": str(self.project_root),
                "data_dir": str(self.data_dir),
                "output_dir": str(self.output_dir),
            }
        }


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()

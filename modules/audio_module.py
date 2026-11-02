"""
Acoustic Processing and Classroom Speech Activity Analysis Module.
Analyzes speech dynamics, silence ratios, and aggregate discussion intensity
without identifying individual speakers or inferring personal attendance.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Classroom-level acoustic analyzer using SoundFile, Librosa, and NumPy.
    Provides Voice Activity Detection (VAD), silence ratios, and interaction metrics.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        frame_length: int = 2048,
        hop_length: int = 512,
        silence_rms_floor: float = 0.012,
        energy_percentile_cutoff: float = 35.0,
    ):
        self.target_sr = target_sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.silence_rms_floor = silence_rms_floor
        self.energy_percentile_cutoff = energy_percentile_cutoff

    def load_audio(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono at target sample rate.
        Supports WAV, FLAC, OGG, MP3.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Try soundfile first (fast and robust)
            data, sr = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                # Convert multi-channel to mono
                data = np.mean(data, axis=1)

            # Resample if sample rate doesn't match and librosa is available
            if sr != self.target_sr:
                try:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr
                except Exception:
                    # Simple linear decimation/interpolation fallback
                    step = sr / self.target_sr
                    indices = np.arange(0, len(data), step).astype(int)
                    indices = indices[indices < len(data)]
                    data = data[indices]
                    sr = self.target_sr

            return data.astype(np.float32), sr

        except Exception as e:
            logger.warning("Soundfile load failed (%s), attempting librosa...", e)
            try:
                import librosa
                data, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
                return data.astype(np.float32), sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file: {e2}")

    @staticmethod
    def compute_frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Calculate Root-Mean-Square (RMS) energy for sliding frames."""
        num_frames = 1 + (len(audio) - frame_length) // hop_length
        if num_frames <= 0:
            return np.array([float(np.sqrt(np.mean(audio ** 2)))], dtype=np.float32)

        # Vectorized frame slicing
        shape = (num_frames, frame_length)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        return rms.astype(np.float32)

    @staticmethod
    def compute_frame_zcr(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Compute Zero-Crossing Rate (ZCR) for sliding frames."""
        num_frames = 1 + (len(audio) - frame_length) // hop_length
        if num_frames <= 0:
            return np.array([0.0], dtype=np.float32)

        shape = (num_frames, frame_length)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        zero_crossings = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        return zero_crossings.astype(np.float32)

    def analyze_audio(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Analyze audio recording and extract classroom participation indicators.
        """
        audio, sr = self.load_audio(audio_path)
        duration_sec = len(audio) / sr if sr > 0 else 0.0

        if duration_sec < 0.2:
            return {
                "duration_sec": 0.0,
                "speech_ratio": 0.0,
                "silence_ratio": 1.0,
                "avg_rms_energy": 0.0,
                "spectral_centroid_mean": 0.0,
                "discussion_intensity_score": 0.0,
                "turn_taking_events": 0,
                "temporal_bins": [],
            }

        # Frame energy and ZCR
        rms = self.compute_frame_rms(audio, self.frame_length, self.hop_length)
        zcr = self.compute_frame_zcr(audio, self.frame_length, self.hop_length)

        # Dynamic noise floor estimation
        dynamic_floor = float(np.percentile(rms, self.energy_percentile_cutoff))
        adaptive_threshold = max(self.silence_rms_floor, dynamic_floor * 1.4)

        # Voice Activity Detection (VAD) boolean mask per frame
        speech_mask = (rms > adaptive_threshold) & (zcr > 0.02) & (zcr < 0.45)

        # Compute Spectral Centroid if librosa is available, else fallback approximation
        spectral_centroid_mean = 1500.0
        try:
            import librosa
            cent = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length
            )
            spectral_centroid_mean = float(np.mean(cent))
        except Exception:
            # Fallback estimation using ZCR * Nyquist
            spectral_centroid_mean = float(np.mean(zcr) * (sr / 2.0))

        # Classroom aggregate ratios
        speech_frames = int(np.sum(speech_mask))
        total_frames = max(1, len(speech_mask))
        speech_ratio = round(float(speech_frames / total_frames), 3)
        silence_ratio = round(max(0.0, 1.0 - speech_ratio), 3)

        # Estimate turn-taking transitions (transitions from silence > 1.2s to speech)
        fps_audio = sr / self.hop_length
        min_silence_gap_frames = int(1.2 * fps_audio)
        turn_events = 0
        consecutive_silence = 0

        for is_speech in speech_mask:
            if not is_speech:
                consecutive_silence += 1
            else:
                if consecutive_silence >= min_silence_gap_frames:
                    turn_events += 1
                consecutive_silence = 0

        # Discussion Intensity Index (0.0 to 1.0)
        # Balances healthy conversational speech (30-65%), turns, and dynamic energy range
        ideal_speech_factor = 1.0 - abs(speech_ratio - 0.45) / 0.45
        ideal_speech_factor = max(0.0, min(1.0, ideal_speech_factor))
        turn_factor = min(1.0, turn_events / max(1.0, duration_sec / 30.0))
        discussion_intensity = round(0.55 * ideal_speech_factor + 0.45 * turn_factor, 3)

        # Second-by-second temporal bins for multimodal fusion
        frames_per_sec = int(round(fps_audio))
        temporal_bins: List[Dict[str, Any]] = []

        total_secs = int(np.floor(duration_sec))
        for s in range(total_secs):
            start_f = s * frames_per_sec
            end_f = min(len(speech_mask), (s + 1) * frames_per_sec)
            if start_f < len(speech_mask):
                bin_speech_frames = speech_mask[start_f:end_f]
                bin_active = 1 if np.mean(bin_speech_frames) >= 0.35 else 0
                bin_rms = float(np.mean(rms[start_f:end_f])) if start_f < len(rms) else 0.0
                temporal_bins.append({
                    "timestamp_sec": float(s),
                    "speech_active": bin_active,
                    "rms_energy": round(bin_rms, 4),
                })

        return {
            "duration_sec": round(duration_sec, 2),
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
            "avg_rms_energy": round(float(np.mean(rms)), 4),
            "spectral_centroid_mean": round(spectral_centroid_mean, 1),
            "discussion_intensity_score": discussion_intensity,
            "turn_taking_events": turn_events,
            "temporal_bins": temporal_bins,
            "methodology_note": (
                "Acoustic indicators reflect aggregate classroom-level participation. "
                "Individual students are never identified or marked absent/present from audio alone."
            ),
        }

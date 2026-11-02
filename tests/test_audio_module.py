"""
Unit tests for AudioProcessor, Voice Activity Detection (VAD), and acoustic feature extraction.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.audio_module import AudioProcessor


def test_frame_rms_and_zcr_calculations():
    """Verify sliding frame RMS energy and Zero-Crossing Rate mathematical formulas."""
    processor = AudioProcessor(target_sr=16000, frame_length=1024, hop_length=512)

    # 1. Pure sine wave: f = 1000Hz, amplitude = 0.5
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    rms = processor.compute_frame_rms(sine_wave, 1024, 512)
    # Theoretical RMS for sine wave with amplitude A is A / sqrt(2) ~ 0.5 / 1.414 = 0.3535
    assert len(rms) > 0
    assert pytest.approx(float(np.mean(rms)), rel=0.05) == 0.3535

    zcr = processor.compute_frame_zcr(sine_wave, 1024, 512)
    assert len(zcr) > 0
    assert np.all(zcr > 0.0)


def test_audio_analysis_on_synthetic_speech():
    """Verify audio analysis pipeline on synthesized multi-segment signal."""
    processor = AudioProcessor(target_sr=16000)

    # Create 4 seconds: 2s active speech burst, 2s near silence
    sr = 16000
    t = np.linspace(0, 4.0, 4 * sr, endpoint=False)
    signal = np.random.normal(0, 0.002, 4 * sr).astype(np.float32)  # Low background noise

    # Add speech harmonics in first 2 seconds
    speech_mask = t < 2.0
    signal[speech_mask] += (0.15 * np.sin(2 * np.pi * 200 * t[speech_mask])).astype(np.float32)

    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(str(tmp_path), signal, sr)
        results = processor.analyze_audio(tmp_path)

        assert results["duration_sec"] == pytest.approx(4.0, rel=0.05)
        # Speech ratio should be ~ 0.50 (around half of the 4 seconds)
        assert 0.30 <= results["speech_ratio"] <= 0.70
        assert results["silence_ratio"] > 0.0
        assert "methodology_note" in results
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

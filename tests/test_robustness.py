import numpy as np
import pytest

from harmonydagger.robustness import (
    augment_and_check_survival,
    simulate_low_pass_filter,
    simulate_mp3_compression,
    simulate_resampling,
)


class TestRobustness:
    """Tests for the robustness augmentation module."""

    @pytest.fixture
    def sine_wave(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        return audio, sr

    def test_low_pass_filter_shape(self, sine_wave):
        audio, sr = sine_wave
        result = simulate_low_pass_filter(audio, sr, cutoff_hz=8000)
        assert len(result) == len(audio)

    def test_low_pass_removes_high_freq(self, sine_wave):
        """A 440Hz signal should survive a 8kHz low-pass filter."""
        audio, sr = sine_wave
        result = simulate_low_pass_filter(audio, sr, cutoff_hz=8000)
        corr = np.corrcoef(audio, result)[0, 1]
        assert corr > 0.95

    def test_resample_roundtrip_shape(self, sine_wave):
        audio, sr = sine_wave
        result = simulate_resampling(audio, sr, target_sr=16000)
        assert len(result) == len(audio)

    def test_mp3_compression_shape(self, sine_wave):
        audio, sr = sine_wave
        result = simulate_mp3_compression(audio, sr, bitrate=128)
        # MP3 may slightly change length; should be close
        assert abs(len(result) - len(audio)) < sr * 0.1

    def test_augment_and_check_survival(self, sine_wave):
        audio, sr = sine_wave
        perturbation = np.random.default_rng(42).normal(0, 0.01, len(audio))
        report = augment_and_check_survival(audio, perturbation, sr)
        assert "low_pass" in report
        assert "resample" in report
        assert all(isinstance(v, float) for v in report.values())

import numpy as np
import pytest

from harmonydagger.verify import compute_feature_similarity, verify_protection


class TestVerify:
    """Tests for the verification module."""

    @pytest.fixture
    def sine_wave(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        return audio, sr

    def test_identical_audio_high_similarity(self, sine_wave):
        audio, sr = sine_wave
        sim = compute_feature_similarity(audio, audio, sr)
        assert sim > 0.99

    def test_different_audio_lower_similarity(self, sine_wave):
        audio, sr = sine_wave
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, len(audio))
        sim = compute_feature_similarity(audio, audio + noise, sr)
        assert sim < 0.95

    def test_verify_protection_returns_dict(self, sine_wave):
        audio, sr = sine_wave
        protected = audio + np.random.default_rng(42).normal(0, 0.01, len(audio))
        result = verify_protection(audio, protected, sr)
        assert "mfcc_similarity" in result
        assert "spectral_similarity" in result
        assert "protection_score" in result
        assert 0.0 <= result["protection_score"] <= 1.0

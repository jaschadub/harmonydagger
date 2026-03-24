import numpy as np
import pytest

from harmonydagger.benchmark import compute_snr, generate_benchmark_report


class TestBenchmark:
    """Tests for the benchmark module."""

    @pytest.fixture
    def sine_wave(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        return audio, sr

    def test_snr_identical_is_high(self, sine_wave):
        audio, sr = sine_wave
        snr = compute_snr(audio, audio)
        assert snr > 100

    def test_snr_with_noise(self, sine_wave):
        audio, sr = sine_wave
        noisy = audio + np.random.default_rng(42).normal(0, 0.01, len(audio))
        snr = compute_snr(audio, noisy)
        assert 20 < snr < 60

    def test_benchmark_report_keys(self, sine_wave):
        audio, sr = sine_wave
        noisy = audio + np.random.default_rng(42).normal(0, 0.01, len(audio))
        report = generate_benchmark_report(audio, noisy, sr)
        assert "snr_db" in report
        assert "max_perturbation" in report
        assert "rms_perturbation" in report
        assert "perturbation_ratio" in report

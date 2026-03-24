import numpy as np
import pytest

from harmonydagger.phase import generate_phase_perturbation


class TestPhasePerturbation:
    """Tests for the phase perturbation module."""

    @pytest.fixture
    def sine_wave(self):
        """Create a 1-second 440Hz sine wave at 22050 Hz sample rate."""
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        return audio, sr

    @pytest.fixture
    def broadband_signal(self):
        """Create a broadband signal (white noise + tones) for robust testing."""
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        rng = np.random.default_rng(123)
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.2 * np.sin(2 * np.pi * 1000 * t)
            + 0.1 * rng.normal(0, 1, sr)
        ).astype(np.float64)
        return audio, sr

    def test_output_shape_matches_input(self, sine_wave):
        audio, sr = sine_wave
        result = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=42)
        assert len(result) == len(audio)

    def test_perturbation_is_small(self, sine_wave):
        audio, sr = sine_wave
        perturbation = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=42)
        rms = np.sqrt(np.mean(perturbation**2))
        signal_rms = np.sqrt(np.mean(audio**2))
        assert rms < signal_rms * 0.15

    def test_perturbation_not_all_zeros(self, sine_wave):
        audio, sr = sine_wave
        perturbation = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=42)
        assert np.any(perturbation != 0)

    def test_max_phase_shift_respected(self, broadband_signal):
        """Smaller max_phase_shift should produce smaller or equal perturbation."""
        audio, sr = broadband_signal
        perturbation_small = generate_phase_perturbation(
            audio, sr, window_size=1024, hop_size=512, max_phase_shift=0.1, seed=42
        )
        perturbation_large = generate_phase_perturbation(
            audio, sr, window_size=1024, hop_size=512, max_phase_shift=0.5, seed=42
        )
        rms_small = np.sqrt(np.mean(perturbation_small**2))
        rms_large = np.sqrt(np.mean(perturbation_large**2))
        assert rms_small <= rms_large + 1e-6

    def test_reproducible_with_seed(self, sine_wave):
        audio, sr = sine_wave
        p1 = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=99)
        p2 = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, sine_wave):
        audio, sr = sine_wave
        p1 = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=1)
        p2 = generate_phase_perturbation(audio, sr, window_size=1024, hop_size=512, seed=2)
        assert not np.allclose(p1, p2)

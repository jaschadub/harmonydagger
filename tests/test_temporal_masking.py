import numpy as np
import pytest

from harmonydagger.temporal_masking import apply_temporal_masking


class TestTemporalMasking:
    """Tests for the temporal masking module."""

    @pytest.fixture
    def sine_wave(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        return audio, sr

    @pytest.fixture
    def impulse_signal(self):
        """Signal with a loud burst followed by silence - ideal for forward masking."""
        sr = 22050
        audio = np.zeros(sr, dtype=np.float64)
        burst_samples = int(0.1 * sr)
        t = np.linspace(0, 0.1, burst_samples, endpoint=False)
        audio[:burst_samples] = 0.9 * np.sin(2 * np.pi * 1000 * t)
        return audio, sr

    def test_output_shape_matches_input(self, sine_wave):
        audio, sr = sine_wave
        result = apply_temporal_masking(audio, sr, seed=42)
        assert len(result) == len(audio)

    def test_masking_is_small(self, sine_wave):
        audio, sr = sine_wave
        perturbation = apply_temporal_masking(audio, sr, seed=42)
        rms = np.sqrt(np.mean(perturbation**2))
        signal_rms = np.sqrt(np.mean(audio**2))
        assert rms < signal_rms * 0.15

    def test_perturbation_after_loud_burst(self, impulse_signal):
        """After a loud burst, forward masking should allow more perturbation."""
        audio, sr = impulse_signal
        perturbation = apply_temporal_masking(audio, sr, seed=42)
        burst_end = int(0.1 * sr)
        post_burst = perturbation[burst_end:int(0.3 * sr)]
        late_silence = perturbation[int(0.5 * sr):int(0.7 * sr)]
        post_burst_rms = np.sqrt(np.mean(post_burst**2))
        late_rms = np.sqrt(np.mean(late_silence**2))
        assert post_burst_rms >= late_rms

    def test_reproducible_with_seed(self, sine_wave):
        audio, sr = sine_wave
        p1 = apply_temporal_masking(audio, sr, seed=99)
        p2 = apply_temporal_masking(audio, sr, seed=99)
        np.testing.assert_array_equal(p1, p2)

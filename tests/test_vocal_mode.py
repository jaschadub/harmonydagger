import numpy as np
import pytest

from harmonydagger.vocal_mode import apply_vocal_emphasis, compute_vocal_emphasis_curve


class TestVocalMode:
    """Tests for the vocal-specific mode module."""

    @pytest.fixture
    def freqs(self):
        """STFT frequency bins for sr=22050, window=1024."""
        return np.linspace(0, 22050 / 2, 513)

    def test_emphasis_curve_shape(self, freqs):
        curve = compute_vocal_emphasis_curve(freqs)
        assert len(curve) == len(freqs)

    def test_emphasis_higher_in_vocal_range(self, freqs):
        curve = compute_vocal_emphasis_curve(freqs)
        vocal_mask = (freqs >= 300) & (freqs <= 3000)
        non_vocal_mask = (freqs > 3000) & (freqs < 8000)
        avg_vocal = np.mean(curve[vocal_mask])
        avg_non_vocal = np.mean(curve[non_vocal_mask])
        assert avg_vocal > avg_non_vocal

    def test_emphasis_curve_values_bounded(self, freqs):
        curve = compute_vocal_emphasis_curve(freqs)
        assert np.all(curve >= 0.0)
        assert np.all(curve <= 3.0)

    def test_apply_vocal_emphasis_shape(self):
        noise_mag = np.ones((513, 10))
        freqs = np.linspace(0, 11025, 513)
        result = apply_vocal_emphasis(noise_mag, freqs)
        assert result.shape == noise_mag.shape

    def test_apply_vocal_emphasis_boosts_vocal_bins(self):
        noise_mag = np.ones((513, 10))
        freqs = np.linspace(0, 11025, 513)
        result = apply_vocal_emphasis(noise_mag, freqs)
        vocal_mask = (freqs >= 300) & (freqs <= 3000)
        assert np.mean(result[vocal_mask, :]) > 1.0

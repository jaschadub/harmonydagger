"""
Vocal-specific mode for HarmonyDagger.

Optimizes the psychoacoustic model specifically for the human vocal range
(300Hz-3kHz) with emphasis on formant frequencies. Useful for podcasters,
singers, and voice actors protecting against AI voice cloning.
"""
import numpy as np
from numpy.typing import NDArray

from .common import (
    VOCAL_EMPHASIS_FACTOR,
    VOCAL_FORMANT_FREQS,
    VOCAL_FREQ_HIGH_HZ,
    VOCAL_FREQ_LOW_HZ,
)


def compute_vocal_emphasis_curve(
    freqs: NDArray[np.float64],
    emphasis_factor: float = VOCAL_EMPHASIS_FACTOR,
) -> NDArray[np.float64]:
    """
    Compute a frequency-dependent emphasis curve for the vocal range.

    The curve is 1.0 outside the vocal range and boosted within it,
    with extra emphasis near formant frequencies.

    Args:
        freqs: Array of frequency bin centers in Hz.
        emphasis_factor: How much to boost perturbation in vocal range.

    Returns:
        Emphasis multiplier array (same shape as freqs).
    """
    curve = np.ones_like(freqs, dtype=np.float64)

    for i, f in enumerate(freqs):
        if VOCAL_FREQ_LOW_HZ <= f <= VOCAL_FREQ_HIGH_HZ:
            base_boost = emphasis_factor
            formant_boost = 0.0
            for formant_f in VOCAL_FORMANT_FREQS:
                formant_boost += 0.5 * np.exp(-0.5 * ((f - formant_f) / 200.0) ** 2)
            curve[i] = base_boost + formant_boost

    return curve


def apply_vocal_emphasis(
    noise_magnitude: NDArray[np.float64],
    freqs: NDArray[np.float64],
    emphasis_factor: float = VOCAL_EMPHASIS_FACTOR,
) -> NDArray[np.float64]:
    """
    Apply vocal emphasis to a noise magnitude spectrogram.

    Args:
        noise_magnitude: Noise magnitude array (freq_bins x time_frames).
        freqs: Frequency bin centers in Hz.
        emphasis_factor: Boost factor for vocal range.

    Returns:
        Emphasized noise magnitude (same shape).
    """
    curve = compute_vocal_emphasis_curve(freqs, emphasis_factor)
    return noise_magnitude * curve[:, np.newaxis]

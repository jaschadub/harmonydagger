"""
Benchmarking module for HarmonyDagger.

Provides SNR measurement and transparency reporting to help users
understand the impact of protection at different settings.
"""
import logging
from typing import Dict

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_snr(
    original: NDArray[np.float64],
    processed: NDArray[np.float64],
) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    Args:
        original: Original audio signal.
        processed: Processed (protected) audio signal.

    Returns:
        SNR in dB. Higher = less audible noise.
    """
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    noise = processed - original
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)

    if noise_power < 1e-20:
        return 200.0

    return float(10 * np.log10(signal_power / noise_power))


def generate_benchmark_report(
    original: NDArray[np.float64],
    protected: NDArray[np.float64],
    sr: int,
) -> Dict[str, float]:
    """
    Generate a transparency report comparing original and protected audio.

    Args:
        original: Original audio signal.
        protected: Protected audio signal.
        sr: Sample rate.

    Returns:
        Dict with SNR, max perturbation, RMS perturbation, and ratio metrics.
    """
    min_len = min(len(original), len(protected))
    original = original[:min_len]
    protected = protected[:min_len]

    perturbation = protected - original

    snr = compute_snr(original, protected)
    max_pert = float(np.max(np.abs(perturbation)))
    rms_pert = float(np.sqrt(np.mean(perturbation**2)))
    signal_rms = float(np.sqrt(np.mean(original**2)))

    return {
        "snr_db": snr,
        "max_perturbation": max_pert,
        "rms_perturbation": rms_pert,
        "signal_rms": signal_rms,
        "perturbation_ratio": rms_pert / signal_rms if signal_rms > 1e-10 else 0.0,
        "duration_seconds": float(min_len / sr),
    }

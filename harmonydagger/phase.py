"""
Phase perturbation module for HarmonyDagger.

Applies subtle phase shifts to audio that are imperceptible to humans
but disrupt AI model feature extraction. AI models are often more
sensitive to phase inconsistencies than human listeners.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import istft, stft

from .common import PHASE_PERTURBATION_MAX_RADIANS
from .psychoacoustics import hearing_threshold, magnitude_to_db


def generate_phase_perturbation(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int = 1024,
    hop_size: int = 512,
    max_phase_shift: float = PHASE_PERTURBATION_MAX_RADIANS,
    seed: int = None,
) -> NDArray[np.float64]:
    """
    Generate a phase-based perturbation signal.

    Applies frequency-dependent phase shifts that stay within
    psychoacoustic masking thresholds. Phase shifts are larger
    where the signal magnitude is well above the hearing threshold
    (more masking available) and smaller near threshold.

    Args:
        audio: Input audio signal (mono, float64).
        sr: Sample rate in Hz.
        window_size: STFT window size.
        hop_size: STFT hop size.
        max_phase_shift: Maximum phase shift in radians.
        seed: RNG seed for reproducibility. None for random.

    Returns:
        Perturbation signal (same length as input) to be added to audio.
    """
    overlap = window_size - hop_size
    freqs, times, stft_matrix = stft(
        audio, fs=sr, nperseg=window_size, noverlap=overlap
    )
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    rng = np.random.default_rng(seed=seed)
    phase_offsets = np.zeros_like(phase)

    for f_idx in range(len(freqs)):
        freq_hz = freqs[f_idx]
        if freq_hz <= 0:
            continue
        thresh_db = hearing_threshold(freq_hz)

        for t_idx in range(magnitude.shape[1]):
            sig_db = magnitude_to_db(magnitude[f_idx, t_idx])
            margin_db = sig_db - thresh_db

            if margin_db > 0:
                # Scale phase shift by how far above threshold we are
                # 60 dB above threshold -> full max_phase_shift
                scale = min(1.0, margin_db / 60.0)
                phase_offsets[f_idx, t_idx] = rng.uniform(
                    -max_phase_shift * scale, max_phase_shift * scale
                )

    # Perturbation is the difference between shifted and original
    shifted_stft = magnitude * np.exp(1j * (phase + phase_offsets))
    original_stft = magnitude * np.exp(1j * phase)
    diff_stft = shifted_stft - original_stft

    _, perturbation = istft(diff_stft, fs=sr, nperseg=window_size, noverlap=overlap)

    if len(perturbation) > len(audio):
        perturbation = perturbation[: len(audio)]
    elif len(perturbation) < len(audio):
        perturbation = np.pad(perturbation, (0, len(audio) - len(perturbation)))

    return perturbation

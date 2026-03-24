"""
Temporal masking module for HarmonyDagger.

Exploits forward masking: after a loud sound, quieter sounds are
masked for ~200ms. This allows hiding more aggressive perturbations
in the temporal shadow of loud events without degrading perceived quality.
"""
import numpy as np
from numpy.typing import NDArray

from .common import (
    FORWARD_MASKING_DECAY_MS,
    FORWARD_MASKING_DECAY_RATE,
    FORWARD_MASKING_INITIAL_DB,
)


def _compute_envelope(audio: NDArray[np.float64], sr: int, frame_ms: float = 10.0) -> NDArray[np.float64]:
    """Compute RMS envelope with given frame size in milliseconds."""
    frame_samples = max(1, int(sr * frame_ms / 1000.0))
    n_frames = len(audio) // frame_samples
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        envelope[i] = np.sqrt(np.mean(audio[start:end] ** 2))
    return envelope


def _compute_forward_masking_curve(
    envelope: NDArray[np.float64],
    frame_ms: float = 10.0,
) -> NDArray[np.float64]:
    """
    Compute the forward masking curve from an RMS envelope.

    For each frame, the masking level is the maximum of the current
    signal level and the decayed masking from previous loud frames.
    """
    masking = np.zeros_like(envelope)
    decay_frames = int(FORWARD_MASKING_DECAY_MS / frame_ms)

    for i in range(len(envelope)):
        current_level = envelope[i]
        decayed_masking = 0.0
        lookback = min(i, decay_frames)
        for j in range(1, lookback + 1):
            decay = FORWARD_MASKING_INITIAL_DB * np.exp(-FORWARD_MASKING_DECAY_RATE * j * frame_ms)
            past_contribution = envelope[i - j] * (decay / FORWARD_MASKING_INITIAL_DB)
            decayed_masking = max(decayed_masking, past_contribution)

        masking[i] = max(current_level, decayed_masking)

    return masking


def apply_temporal_masking(
    audio: NDArray[np.float64],
    sr: int,
    noise_scale: float = 0.05,
    frame_ms: float = 10.0,
    seed: int = None,
) -> NDArray[np.float64]:
    """
    Generate perturbation noise shaped by temporal forward masking.

    Louder noise is placed in temporal regions where forward masking
    makes it imperceptible (after loud events).

    Args:
        audio: Input audio signal (mono, float64).
        sr: Sample rate in Hz.
        noise_scale: Base scale for the perturbation.
        frame_ms: Frame size in ms for envelope computation.
        seed: RNG seed for reproducibility. None for random.

    Returns:
        Perturbation signal (same length as input).
    """
    frame_samples = max(1, int(sr * frame_ms / 1000.0))
    envelope = _compute_envelope(audio, sr, frame_ms)
    masking_curve = _compute_forward_masking_curve(envelope, frame_ms)

    max_mask = np.max(masking_curve) if np.max(masking_curve) > 0 else 1.0
    masking_normalized = masking_curve / max_mask

    rng = np.random.default_rng(seed=seed)
    perturbation = np.zeros(len(audio), dtype=np.float64)

    for i in range(len(masking_normalized)):
        start = i * frame_samples
        end = min(start + frame_samples, len(audio))
        n_samples = end - start
        noise = rng.normal(0, noise_scale * masking_normalized[i], size=n_samples)
        perturbation[start:end] = noise

    return perturbation

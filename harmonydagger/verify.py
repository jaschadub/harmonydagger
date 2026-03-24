"""
Verification module for HarmonyDagger.

Tests whether protected audio is effectively "unlearnable" by checking
if standard audio feature extractors can still recognize similarity
between original and protected versions.
"""
import logging
from typing import Dict

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_feature_similarity(
    audio_a: NDArray[np.float64],
    audio_b: NDArray[np.float64],
    sr: int,
    n_mfcc: int = 13,
) -> float:
    """
    Compute MFCC-based feature similarity between two audio signals.

    Args:
        audio_a: First audio signal.
        audio_b: Second audio signal.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Cosine similarity (0-1) between mean MFCC vectors.
    """
    mfcc_a = librosa.feature.mfcc(y=audio_a, sr=sr, n_mfcc=n_mfcc)
    mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sr, n_mfcc=n_mfcc)

    feat_a = np.mean(mfcc_a, axis=1)
    feat_b = np.mean(mfcc_b, axis=1)

    dot = np.dot(feat_a, feat_b)
    norm_a = np.linalg.norm(feat_a)
    norm_b = np.linalg.norm(feat_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(dot / (norm_a * norm_b))


def _spectral_similarity(
    audio_a: NDArray[np.float64],
    audio_b: NDArray[np.float64],
    sr: int,
) -> float:
    """Compute spectral centroid similarity."""
    spec_a = librosa.feature.spectral_centroid(y=audio_a, sr=sr)[0]
    spec_b = librosa.feature.spectral_centroid(y=audio_b, sr=sr)[0]

    min_len = min(len(spec_a), len(spec_b))
    spec_a = spec_a[:min_len]
    spec_b = spec_b[:min_len]

    if np.std(spec_a) < 1e-10 or np.std(spec_b) < 1e-10:
        return 1.0 if np.allclose(spec_a, spec_b) else 0.0

    corr = np.corrcoef(spec_a, spec_b)[0, 1]
    return float(max(0.0, corr))


def verify_protection(
    original: NDArray[np.float64],
    protected: NDArray[np.float64],
    sr: int,
) -> Dict[str, float]:
    """
    Verify how effectively audio is protected.

    A low protection_score means features are very similar (weak protection).
    A high score means features are disrupted (effective protection).

    Args:
        original: Original unprotected audio.
        protected: Protected audio.
        sr: Sample rate.

    Returns:
        Dict with mfcc_similarity, spectral_similarity, and protection_score.
    """
    mfcc_sim = compute_feature_similarity(original, protected, sr)
    spec_sim = _spectral_similarity(original, protected, sr)

    avg_similarity = (mfcc_sim + spec_sim) / 2.0
    protection_score = 1.0 - avg_similarity

    return {
        "mfcc_similarity": mfcc_sim,
        "spectral_similarity": spec_sim,
        "protection_score": max(0.0, min(1.0, protection_score)),
    }

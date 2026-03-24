"""
Ensemble attack module for HarmonyDagger.

Instead of a single perturbation pattern, generates an ensemble of
perturbations targeting different AI model architectures:

- **Spectral (STFT-domain)**: Targets GAN-based audio generators that
  operate on STFT magnitude spectrograms.
- **Mel-band**: Targets diffusion-based TTS/voice cloning models that
  use mel-spectrograms as input representation.
- **Embedding-disruption**: Targets encoder/embedding models (speaker
  verification, voice conversion) by perturbing MFCC-sensitive bands.

The combined perturbation is more robust against a wider range of
AI systems than any single strategy alone.
"""
import logging
from typing import Dict, List, Optional

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import istft, stft

from .common import (
    DEFAULT_HOP_SIZE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WINDOW_SIZE,
)
from .psychoacoustics import hearing_threshold, magnitude_to_db

logger = logging.getLogger(__name__)

# Default ensemble strategy weights (must sum to 1.0)
DEFAULT_STRATEGY_WEIGHTS = {
    "spectral": 0.4,
    "mel_band": 0.35,
    "embedding": 0.25,
}


def _generate_spectral_perturbation(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int,
    hop_size: int,
    noise_scale: float,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Spectral perturbation targeting STFT-domain models (GANs).

    Adds broadband noise shaped by spectral energy distribution,
    spreading perturbation across all active frequency bins rather
    than just the dominant frequency's critical band.
    """
    overlap = window_size - hop_size
    freqs, _, stft_matrix = stft(
        audio, fs=sr, nperseg=window_size, noverlap=overlap
    )
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    rng = np.random.default_rng(seed=seed)
    noise_magnitude = np.zeros_like(magnitude)

    for f_idx in range(len(freqs)):
        freq_hz = freqs[f_idx]
        if freq_hz <= 0:
            continue
        thresh_db = hearing_threshold(freq_hz)

        for t_idx in range(magnitude.shape[1]):
            sig_db = magnitude_to_db(magnitude[f_idx, t_idx])
            margin = sig_db - thresh_db
            if margin > 0:
                # Perturbation proportional to signal energy at this bin
                scale = min(1.0, margin / 40.0)
                noise_mag = noise_scale * magnitude[f_idx, t_idx] * scale
                noise_magnitude[f_idx, t_idx] = min(
                    noise_mag, 0.5 * magnitude[f_idx, t_idx]
                )

    # Randomize phase of noise (uncorrelated with signal)
    random_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
    noise_stft = noise_magnitude * np.exp(1j * random_phase)
    _, noise_audio = istft(noise_stft, fs=sr, nperseg=window_size, noverlap=overlap)

    if len(noise_audio) > len(audio):
        noise_audio = noise_audio[: len(audio)]
    elif len(noise_audio) < len(audio):
        noise_audio = np.pad(noise_audio, (0, len(audio) - len(noise_audio)))

    return noise_audio


def _generate_mel_band_perturbation(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int,
    hop_size: int,
    noise_scale: float,
    n_mels: int = 80,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Mel-band perturbation targeting diffusion-based TTS models.

    Diffusion models typically use mel-spectrograms (80 or 128 bands).
    This strategy concentrates perturbation energy at mel-band boundaries
    where quantization artifacts are most disruptive to the denoising process.
    """
    overlap = window_size - hop_size
    freqs, _, stft_matrix = stft(
        audio, fs=sr, nperseg=window_size, noverlap=overlap
    )
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    rng = np.random.default_rng(seed=seed)

    # Build mel filterbank to identify mel-band boundaries
    mel_filters = librosa.filters.mel(sr=sr, n_fft=window_size, n_mels=n_mels)
    # mel_filters shape: (n_mels, n_fft//2+1)

    # Find mel-band boundary frequencies (where filters overlap)
    boundary_weights = np.zeros(magnitude.shape[0])
    for mel_idx in range(n_mels - 1):
        # Overlap region between adjacent mel bands
        overlap_region = np.minimum(mel_filters[mel_idx], mel_filters[mel_idx + 1])
        boundary_weights += overlap_region

    # Normalize
    max_bw = np.max(boundary_weights)
    if max_bw > 0:
        boundary_weights /= max_bw

    # Also boost perturbation at mel-band centers (disrupt mel features)
    center_weights = np.zeros(magnitude.shape[0])
    for mel_idx in range(n_mels):
        center_bin = np.argmax(mel_filters[mel_idx])
        if center_bin < len(center_weights):
            # Gaussian around center
            for f_idx in range(len(center_weights)):
                dist = abs(f_idx - center_bin)
                center_weights[f_idx] += np.exp(-0.5 * (dist / 3.0) ** 2)

    max_cw = np.max(center_weights)
    if max_cw > 0:
        center_weights /= max_cw

    # Combine boundary and center emphasis
    mel_emphasis = 0.6 * boundary_weights + 0.4 * center_weights
    max_me = np.max(mel_emphasis)
    if max_me > 0:
        mel_emphasis /= max_me

    noise_magnitude = np.zeros_like(magnitude)
    for f_idx in range(len(freqs)):
        freq_hz = freqs[f_idx]
        if freq_hz <= 0:
            continue
        thresh_db = hearing_threshold(freq_hz)

        for t_idx in range(magnitude.shape[1]):
            sig_db = magnitude_to_db(magnitude[f_idx, t_idx])
            margin = sig_db - thresh_db
            if margin > 0:
                scale = min(1.0, margin / 40.0)
                emphasis = mel_emphasis[f_idx] if f_idx < len(mel_emphasis) else 0.0
                noise_mag = noise_scale * magnitude[f_idx, t_idx] * scale * (0.3 + 0.7 * emphasis)
                noise_magnitude[f_idx, t_idx] = min(
                    noise_mag, 0.5 * magnitude[f_idx, t_idx]
                )

    random_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
    noise_stft = noise_magnitude * np.exp(1j * random_phase)
    _, noise_audio = istft(noise_stft, fs=sr, nperseg=window_size, noverlap=overlap)

    if len(noise_audio) > len(audio):
        noise_audio = noise_audio[: len(audio)]
    elif len(noise_audio) < len(audio):
        noise_audio = np.pad(noise_audio, (0, len(audio) - len(noise_audio)))

    return noise_audio


def _generate_embedding_perturbation(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int,
    hop_size: int,
    noise_scale: float,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Embedding-disruption perturbation targeting encoder models.

    Speaker verification and voice conversion models rely on MFCC-like
    features. This strategy concentrates noise in frequency bands that
    are most influential to the first 13 MFCC coefficients, particularly
    the lower-order coefficients that capture speaker identity.
    """
    overlap = window_size - hop_size
    freqs, _, stft_matrix = stft(
        audio, fs=sr, nperseg=window_size, noverlap=overlap
    )
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    rng = np.random.default_rng(seed=seed)

    # MFCC-sensitive bands: lower mel bands carry speaker identity
    # Focus perturbation on 100Hz-4kHz range where MFCC coefficients
    # 1-6 (speaker identity) are most sensitive
    mfcc_emphasis = np.zeros(len(freqs))
    for f_idx, freq_hz in enumerate(freqs):
        if 100 <= freq_hz <= 4000:
            # Peak emphasis around 300-2000Hz (formant region)
            center = 800.0
            width = 1500.0
            mfcc_emphasis[f_idx] = np.exp(-0.5 * ((freq_hz - center) / width) ** 2)
        elif 4000 < freq_hz <= 8000:
            # Lighter emphasis on higher frequencies (MFCC detail)
            mfcc_emphasis[f_idx] = 0.3 * np.exp(-0.5 * ((freq_hz - 6000) / 2000) ** 2)

    max_emphasis = np.max(mfcc_emphasis)
    if max_emphasis > 0:
        mfcc_emphasis /= max_emphasis

    noise_magnitude = np.zeros_like(magnitude)
    for f_idx in range(len(freqs)):
        freq_hz = freqs[f_idx]
        if freq_hz <= 0:
            continue
        thresh_db = hearing_threshold(freq_hz)

        for t_idx in range(magnitude.shape[1]):
            sig_db = magnitude_to_db(magnitude[f_idx, t_idx])
            margin = sig_db - thresh_db
            if margin > 0:
                scale = min(1.0, margin / 40.0)
                emphasis = mfcc_emphasis[f_idx]
                noise_mag = noise_scale * magnitude[f_idx, t_idx] * scale * (0.2 + 0.8 * emphasis)
                noise_magnitude[f_idx, t_idx] = min(
                    noise_mag, 0.5 * magnitude[f_idx, t_idx]
                )

    random_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
    noise_stft = noise_magnitude * np.exp(1j * random_phase)
    _, noise_audio = istft(noise_stft, fs=sr, nperseg=window_size, noverlap=overlap)

    if len(noise_audio) > len(audio):
        noise_audio = noise_audio[: len(audio)]
    elif len(noise_audio) < len(audio):
        noise_audio = np.pad(noise_audio, (0, len(audio) - len(noise_audio)))

    return noise_audio


# Registry of available strategies
STRATEGIES = {
    "spectral": _generate_spectral_perturbation,
    "mel_band": _generate_mel_band_perturbation,
    "embedding": _generate_embedding_perturbation,
}


def generate_ensemble_perturbation(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    strategy_weights: Optional[Dict[str, float]] = None,
    strategies: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate an ensemble perturbation combining multiple attack strategies.

    Each strategy targets a different class of AI model architecture.
    The final perturbation is a weighted sum of all strategy outputs,
    providing broader protection than any single approach.

    Args:
        audio: Input audio signal (mono, float64).
        sr: Sample rate in Hz.
        window_size: STFT window size.
        hop_size: STFT hop size.
        noise_scale: Base noise scale (0-1).
        strategy_weights: Dict mapping strategy name to weight (must sum to 1.0).
            Defaults to DEFAULT_STRATEGY_WEIGHTS.
        strategies: List of strategy names to use. If None, uses all strategies
            in strategy_weights.
        seed: Base RNG seed. Each strategy gets seed+offset for independence.

    Returns:
        Combined perturbation signal (same length as input).
    """
    if strategy_weights is None:
        strategy_weights = DEFAULT_STRATEGY_WEIGHTS.copy()

    if strategies is not None:
        # Filter to requested strategies and renormalize weights
        strategy_weights = {k: v for k, v in strategy_weights.items() if k in strategies}
        if not strategy_weights:
            raise ValueError(f"No valid strategies in {strategies}. Available: {list(STRATEGIES.keys())}")

    # Normalize weights to sum to 1.0
    total_weight = sum(strategy_weights.values())
    if total_weight <= 0:
        raise ValueError("Strategy weights must be positive")
    strategy_weights = {k: v / total_weight for k, v in strategy_weights.items()}

    combined = np.zeros(len(audio), dtype=np.float64)

    for i, (name, weight) in enumerate(strategy_weights.items()):
        if name not in STRATEGIES:
            logger.warning(f"Unknown strategy '{name}', skipping")
            continue

        strategy_fn = STRATEGIES[name]
        strategy_seed = (seed + i * 1000) if seed is not None else None

        logger.debug(f"Running ensemble strategy '{name}' (weight={weight:.2f})")
        perturbation = strategy_fn(
            audio, sr, window_size, hop_size, noise_scale, seed=strategy_seed
        )
        combined += weight * perturbation

    return combined


def list_strategies() -> Dict[str, str]:
    """Return available ensemble strategies with descriptions."""
    return {
        "spectral": "Broadband STFT-domain noise targeting GAN-based audio generators",
        "mel_band": "Mel-band boundary/center noise targeting diffusion-based TTS models",
        "embedding": "MFCC-sensitive band noise targeting encoder/speaker-verification models",
    }

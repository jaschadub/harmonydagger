"""
Core audio processing and noise generation functions for HarmonyDagger.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import istft, stft

from .common import (
    ADAPTIVE_SCALE_NORM_MIN,
    ADAPTIVE_SCALE_NORM_RANGE,
    ADAPTIVE_SIGNAL_STRENGTH_DIV,
    DEFAULT_DRY_WET,
    DEFAULT_HOP_SIZE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WINDOW_SIZE,
    MASKING_CURVE_SLOPE,
    NOISE_UPPER_BOUND_FACTOR,
)
from .psychoacoustics import (
    bark_scale,
    critical_band_width,
    db_to_magnitude,
    hearing_threshold,
    magnitude_to_db,
)
from .vocal_mode import apply_vocal_emphasis


def generate_psychoacoustic_noise(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    vocal_mode: bool = False,
) -> NDArray[np.float64]:
    """
    Generate psychoacoustically masked noise for a single audio channel.
    """
    overlap = window_size - hop_size
    freqs, times, stft_matrix = stft(
        audio,
        fs=sr,
        nperseg=window_size,
        noverlap=overlap
    )
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    noise_magnitude = np.zeros_like(magnitude)

    # Performance optimization: Pre-calculate bark scale frequencies once instead of
    # repeatedly calling bark_scale(f) for each frequency in the time loop
    bark_freqs = np.array([bark_scale(f) for f in freqs])

    for t in range(magnitude.shape[1]):
        mag_frame = magnitude[:, t]
        dom_freq_idx = np.argmax(mag_frame)
        dom_freq_hz = freqs[dom_freq_idx]
        # Use pre-calculated bark value instead of recalculating
        dom_freq_bark = bark_freqs[dom_freq_idx]

        cb_width_hz = critical_band_width(dom_freq_hz)
        # Ensure masking_band is at least 1 bin wide
        masking_band_bins = max(1, int(cb_width_hz / (sr / window_size)))

        for offset in range(-masking_band_bins, masking_band_bins + 1):
            idx = dom_freq_idx + offset
            if 0 <= idx < magnitude.shape[0]:
                freq_dist_bark = abs(bark_freqs[idx] - dom_freq_bark)
                masking_attenuation_db = MASKING_CURVE_SLOPE * freq_dist_bark

                current_freq_hz = freqs[idx]
                hearing_thresh_db = hearing_threshold(current_freq_hz)
                hearing_thresh_mag = db_to_magnitude(hearing_thresh_db)

                signal_mag_db = magnitude_to_db(mag_frame[idx])
                signal_mag_linear = db_to_magnitude(signal_mag_db)

                current_noise_scale = noise_scale
                if adaptive_scaling:
                    signal_strength_above_thresh_db = signal_mag_db - hearing_thresh_db
                    if signal_strength_above_thresh_db > 0:
                        adaptive_factor = ADAPTIVE_SCALE_NORM_MIN + \
                                          min(ADAPTIVE_SCALE_NORM_RANGE, signal_strength_above_thresh_db / ADAPTIVE_SIGNAL_STRENGTH_DIV)
                        current_noise_scale = noise_scale * adaptive_factor

                noise_level_mag = current_noise_scale * signal_mag_linear * (1.0 - masking_attenuation_db / 20.0)

                # Clip noise to be between hearing threshold and a factor of original signal magnitude
                noise_magnitude[idx, t] = np.clip(
                    noise_level_mag,
                    hearing_thresh_mag,
                    NOISE_UPPER_BOUND_FACTOR * signal_mag_linear
                )

    # Apply vocal emphasis if enabled
    if vocal_mode:
        noise_magnitude = apply_vocal_emphasis(noise_magnitude, freqs)

    noise_stft = noise_magnitude * np.exp(1j * phase)
    _, noise_audio = istft(
        noise_stft,
        fs=sr,
        nperseg=window_size,
        noverlap=overlap
    )

    if len(noise_audio) > len(audio):
        noise_audio = noise_audio[:len(audio)]
    elif len(noise_audio) < len(audio):
        noise_audio = np.pad(noise_audio, (0, len(audio) - len(noise_audio)))

    return noise_audio


def generate_protected_audio(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    dry_wet: float = DEFAULT_DRY_WET,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
) -> NDArray[np.float64]:
    """
    Generate fully protected audio combining all perturbation techniques.

    This is the main orchestrator that composes magnitude noise, phase
    perturbation, temporal masking, and ensemble attacks into a single
    protection signal.

    Args:
        audio: Input audio (mono channel).
        sr: Sample rate.
        window_size: STFT window size.
        hop_size: STFT hop size.
        noise_scale: Base noise scale (0-1).
        adaptive_scaling: Use adaptive noise scaling.
        dry_wet: Mix ratio (0.0 = original, 1.0 = fully protected).
        vocal_mode: Optimize for vocal frequency range.
        use_phase_perturbation: Add phase-based perturbation.
        use_temporal_masking: Add temporal forward masking noise.
        use_ensemble: Use ensemble of perturbations targeting different
            AI architectures (spectral, mel-band, embedding).

    Returns:
        Protected audio signal.
    """
    from .phase import generate_phase_perturbation
    from .temporal_masking import apply_temporal_masking

    if use_ensemble:
        # Ensemble mode: replace base noise with multi-strategy ensemble
        from .ensemble import generate_ensemble_perturbation
        noise = generate_ensemble_perturbation(
            audio, sr, window_size, hop_size, noise_scale
        )
    else:
        # Standard single-strategy psychoacoustic noise
        noise = generate_psychoacoustic_noise(
            audio, sr, window_size, hop_size, noise_scale, adaptive_scaling, vocal_mode
        )

    # Layer phase perturbation
    if use_phase_perturbation:
        phase_noise = generate_phase_perturbation(audio, sr, window_size, hop_size)
        noise = noise + phase_noise

    # Layer temporal masking noise
    if use_temporal_masking:
        temporal_noise = apply_temporal_masking(audio, sr, noise_scale=noise_scale * 0.5)
        noise = noise + temporal_noise

    # Apply dry/wet mix
    protected = audio + dry_wet * noise
    return np.clip(protected, -1.0, 1.0)


def apply_noise_multichannel(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int,
    hop_size: int,
    noise_scale: float,
    adaptive_scaling: bool = True,
    dry_wet: float = DEFAULT_DRY_WET,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
) -> NDArray[np.float64]:
    """
    Process multi-channel audio by applying psychoacoustic noise to each channel.
    """
    if audio.ndim == 1:  # Mono
        return generate_protected_audio(
            audio, sr, window_size, hop_size, noise_scale, adaptive_scaling,
            dry_wet, vocal_mode, use_phase_perturbation, use_temporal_masking,
            use_ensemble
        )
    else:  # Multi-channel
        noisy_channels = []
        for ch_idx in range(audio.shape[0]):
            channel_audio = audio[ch_idx]
            noisy_channel = generate_protected_audio(
                channel_audio, sr, window_size, hop_size, noise_scale, adaptive_scaling,
                dry_wet, vocal_mode, use_phase_perturbation, use_temporal_masking,
                use_ensemble
            )
            noisy_channels.append(noisy_channel)
        return np.vstack(noisy_channels)


def apply_noise_to_audio(audio: NDArray[np.float64], noise: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply generated noise to audio signal and prevent clipping.
    """
    perturbed_audio = audio + noise
    return np.clip(perturbed_audio, -1.0, 1.0)  # Assuming audio is normalized to [-1, 1]

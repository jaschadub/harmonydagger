"""
GPU-accelerated audio processing backend for HarmonyDagger.

Uses PyTorch to vectorize the STFT-based perturbation generation,
replacing the nested Python loops in the CPU implementation with
batched tensor operations. Falls back to CPU NumPy when PyTorch
is unavailable or no GPU is detected.

Requires: pip install harmonydagger[gpu]
"""
import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .common import (
    ADAPTIVE_SCALE_NORM_MIN,
    ADAPTIVE_SCALE_NORM_RANGE,
    ADAPTIVE_SIGNAL_STRENGTH_DIV,
    BARK_SCALE_C1,
    BARK_SCALE_C2,
    BARK_SCALE_C3,
    BARK_SCALE_F_DIV,
    CBW_C1,
    CBW_C2,
    CBW_C3,
    CBW_F_POW,
    DB_LOG_EPSILON,
    DEFAULT_HOP_SIZE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WINDOW_SIZE,
    HEARING_THRESH_C1,
    HEARING_THRESH_C2,
    HEARING_THRESH_EXP_C1,
    HEARING_THRESH_F_OFFSET,
    HEARING_THRESH_F_POW,
    HZ_TO_KHZ,
    MASKING_CURVE_SLOPE,
    NOISE_UPPER_BOUND_FACTOR,
    REFERENCE_PRESSURE,
)

logger = logging.getLogger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def is_gpu_available() -> bool:
    """Check if PyTorch and a CUDA/MPS GPU are available."""
    if not _TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def get_device() -> Optional["torch.device"]:
    """Get the best available device (CUDA > MPS > CPU)."""
    if not _TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hearing_threshold_tensor(freqs_hz: "torch.Tensor") -> "torch.Tensor":
    """Vectorized hearing threshold (ISO 226:2003 simplified) on tensors."""
    f_khz = torch.clamp(freqs_hz / HZ_TO_KHZ, min=1e-6)
    return (
        HEARING_THRESH_C1 * f_khz.pow(HEARING_THRESH_F_POW)
        - HEARING_THRESH_C2
        * torch.exp(HEARING_THRESH_EXP_C1 * (f_khz - HEARING_THRESH_F_OFFSET).pow(2))
    )


def _bark_scale_tensor(freqs_hz: "torch.Tensor") -> "torch.Tensor":
    """Vectorized Bark scale conversion on tensors."""
    return BARK_SCALE_C1 * torch.arctan(
        BARK_SCALE_C2 * freqs_hz
    ) + BARK_SCALE_C3 * torch.arctan((freqs_hz / BARK_SCALE_F_DIV).pow(2))


def _critical_band_width_tensor(freqs_hz: "torch.Tensor") -> "torch.Tensor":
    """Vectorized critical bandwidth (Zwicker) on tensors."""
    f_khz = freqs_hz / HZ_TO_KHZ
    return CBW_C1 + CBW_C2 * (1 + CBW_C3 * f_khz.pow(2)).pow(CBW_F_POW)


def _magnitude_to_db_tensor(mag: "torch.Tensor") -> "torch.Tensor":
    """Vectorized magnitude to dB SPL on tensors."""
    mag = torch.clamp(mag, min=DB_LOG_EPSILON)
    return 20.0 * torch.log10(mag / REFERENCE_PRESSURE)


def _db_to_magnitude_tensor(db: "torch.Tensor") -> "torch.Tensor":
    """Vectorized dB SPL to magnitude on tensors."""
    db_clipped = torch.clamp(db, max=350.0)
    return 10.0 ** (db_clipped / 20.0) * REFERENCE_PRESSURE


def generate_psychoacoustic_noise_gpu(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    vocal_mode: bool = False,
    device: Optional["torch.device"] = None,
) -> NDArray[np.float64]:
    """
    GPU-accelerated psychoacoustic noise generation.

    Replaces the nested Python loops in the CPU version with vectorized
    PyTorch tensor operations. The entire STFT magnitude processing is
    done on GPU/device in a single pass per time frame.

    Args:
        audio: Input audio signal (mono, float64).
        sr: Sample rate in Hz.
        window_size: STFT window size.
        hop_size: STFT hop size.
        noise_scale: Base noise scale (0-1).
        adaptive_scaling: Use adaptive noise scaling.
        vocal_mode: Optimize for vocal frequency range.
        device: PyTorch device. None = auto-detect.

    Returns:
        Noise signal (same length as input) as numpy array.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU acceleration. Install with: pip install torch")

    if device is None:
        device = get_device()

    logger.debug(f"GPU noise generation on device: {device}")

    # Convert audio to tensor and compute STFT
    audio_t = torch.from_numpy(audio).to(device=device, dtype=torch.float64)
    window = torch.hann_window(window_size, dtype=torch.float64, device=device)

    stft_matrix = torch.stft(
        audio_t,
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        return_complex=True,
    )
    # stft_matrix shape: (n_freq, n_time)

    magnitude = stft_matrix.abs()
    phase = stft_matrix.angle()
    n_freq, n_time = magnitude.shape

    # Frequency bin centers in Hz
    freqs = torch.linspace(0, sr / 2.0, n_freq, device=device, dtype=torch.float64)

    # Pre-compute psychoacoustic curves for all frequency bins (vectorized)
    bark_freqs = _bark_scale_tensor(freqs)  # (n_freq,)
    hearing_thresh_db = _hearing_threshold_tensor(freqs)  # (n_freq,)
    hearing_thresh_mag = _db_to_magnitude_tensor(hearing_thresh_db)  # (n_freq,)
    cb_widths_hz = _critical_band_width_tensor(freqs)  # (n_freq,)
    bin_resolution = sr / window_size
    masking_band_bins = torch.clamp((cb_widths_hz / bin_resolution).to(torch.int64), min=1)  # (n_freq,)

    # Pre-compute signal dB and linear magnitudes for all bins/frames
    signal_mag_db = _magnitude_to_db_tensor(magnitude)  # (n_freq, n_time)
    signal_mag_linear = _db_to_magnitude_tensor(signal_mag_db)  # (n_freq, n_time)

    noise_magnitude = torch.zeros_like(magnitude)

    # Process each time frame (the outer loop stays — it's fast with vectorized inner ops)
    for t in range(n_time):
        mag_frame = magnitude[:, t]  # (n_freq,)

        # Find dominant frequency
        dom_idx = torch.argmax(mag_frame).item()
        dom_bark = bark_freqs[dom_idx]
        band_size = masking_band_bins[dom_idx].item()

        # Compute index range for masking band
        idx_start = max(0, dom_idx - band_size)
        idx_end = min(n_freq, dom_idx + band_size + 1)

        # Slice out the masking band (all vectorized)
        band_bark = bark_freqs[idx_start:idx_end]
        band_hearing_db = hearing_thresh_db[idx_start:idx_end]
        band_hearing_mag = hearing_thresh_mag[idx_start:idx_end]
        band_sig_db = signal_mag_db[idx_start:idx_end, t]
        band_sig_linear = signal_mag_linear[idx_start:idx_end, t]

        # Bark distance from dominant
        freq_dist_bark = (band_bark - dom_bark).abs()
        masking_attenuation_db = MASKING_CURVE_SLOPE * freq_dist_bark

        # Compute noise scale (with optional adaptive scaling)
        if adaptive_scaling:
            strength_above = band_sig_db - band_hearing_db
            adaptive_factor = ADAPTIVE_SCALE_NORM_MIN + torch.clamp(
                strength_above / ADAPTIVE_SIGNAL_STRENGTH_DIV,
                min=0.0,
                max=ADAPTIVE_SCALE_NORM_RANGE,
            )
            # Only apply where above threshold
            current_scale = torch.where(
                strength_above > 0,
                noise_scale * adaptive_factor,
                torch.tensor(noise_scale, device=device, dtype=torch.float64),
            )
        else:
            current_scale = torch.full_like(band_sig_linear, noise_scale)

        # Compute noise level
        noise_level = current_scale * band_sig_linear * (1.0 - masking_attenuation_db / 20.0)

        # Clip between hearing threshold and upper bound
        noise_clipped = torch.clamp(
            noise_level,
            min=band_hearing_mag,
            max=NOISE_UPPER_BOUND_FACTOR * band_sig_linear,
        )

        noise_magnitude[idx_start:idx_end, t] = noise_clipped

    # Apply vocal emphasis if enabled
    if vocal_mode:
        from .vocal_mode import compute_vocal_emphasis_curve

        emphasis = compute_vocal_emphasis_curve(freqs.cpu().numpy())
        emphasis_t = torch.from_numpy(emphasis).to(device=device, dtype=torch.float64)
        noise_magnitude = noise_magnitude * emphasis_t.unsqueeze(1)

    # iSTFT to get noise audio
    noise_stft = noise_magnitude * torch.exp(1j * phase)
    noise_audio = torch.istft(
        noise_stft,
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        length=len(audio),
    )

    return noise_audio.cpu().numpy()


def generate_protected_audio_gpu(
    audio: NDArray[np.float64],
    sr: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    dry_wet: float = 1.0,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
    device: Optional["torch.device"] = None,
) -> NDArray[np.float64]:
    """
    GPU-accelerated full protection pipeline.

    Uses GPU for the core psychoacoustic noise generation (the bottleneck),
    then layers CPU-based phase perturbation and temporal masking on top.

    Args:
        audio: Input audio (mono, float64).
        sr: Sample rate.
        window_size: STFT window size.
        hop_size: STFT hop size.
        noise_scale: Base noise scale (0-1).
        adaptive_scaling: Use adaptive noise scaling.
        dry_wet: Mix ratio (0.0 = original, 1.0 = fully protected).
        vocal_mode: Optimize for vocal frequency range.
        use_phase_perturbation: Add phase-based perturbation.
        use_temporal_masking: Add temporal forward masking noise.
        use_ensemble: Use ensemble attacks (runs on CPU, ensemble has its own loops).
        device: PyTorch device. None = auto-detect.

    Returns:
        Protected audio signal as numpy array.
    """
    if use_ensemble:
        # Ensemble strategies have their own STFT loops; keep on CPU for now
        from .ensemble import generate_ensemble_perturbation

        noise = generate_ensemble_perturbation(
            audio, sr, window_size, hop_size, noise_scale
        )
    else:
        # GPU-accelerated core noise generation
        noise = generate_psychoacoustic_noise_gpu(
            audio, sr, window_size, hop_size, noise_scale,
            adaptive_scaling, vocal_mode, device
        )

    # Phase perturbation (CPU — it's not the bottleneck)
    if use_phase_perturbation:
        from .phase import generate_phase_perturbation

        phase_noise = generate_phase_perturbation(audio, sr, window_size, hop_size)
        noise = noise + phase_noise

    # Temporal masking (CPU — fast already)
    if use_temporal_masking:
        from .temporal_masking import apply_temporal_masking

        temporal_noise = apply_temporal_masking(audio, sr, noise_scale=noise_scale * 0.5)
        noise = noise + temporal_noise

    protected = audio + dry_wet * noise
    return np.clip(protected, -1.0, 1.0)

"""
Robustness simulation module for HarmonyDagger.

Simulates common post-processing transforms (MP3 compression, low-pass
filtering, resampling) to test whether perturbations survive purification
attempts.
"""
import logging
import tempfile
from math import gcd
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfilt

logger = logging.getLogger(__name__)


def simulate_low_pass_filter(
    audio: NDArray[np.float64],
    sr: int,
    cutoff_hz: int = 8000,
    order: int = 5,
) -> NDArray[np.float64]:
    """
    Apply a Butterworth low-pass filter to simulate bandwidth reduction.

    Args:
        audio: Input audio signal.
        sr: Sample rate in Hz.
        cutoff_hz: Filter cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered audio signal (same length as input).
    """
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        return audio.copy()
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(order, normalized_cutoff, btype="low", output="sos")
    return sosfilt(sos, audio)


def simulate_resampling(
    audio: NDArray[np.float64],
    sr: int,
    target_sr: int = 16000,
) -> NDArray[np.float64]:
    """
    Simulate resampling by downsampling then upsampling back to original rate.

    Args:
        audio: Input audio signal.
        sr: Original sample rate.
        target_sr: Intermediate sample rate to downsample to.

    Returns:
        Audio resampled back to original rate (same length as input).
    """
    if target_sr >= sr:
        return audio.copy()

    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g

    downsampled = resample_poly(audio, up, down)
    upsampled = resample_poly(downsampled, down, up)

    if len(upsampled) > len(audio):
        upsampled = upsampled[: len(audio)]
    elif len(upsampled) < len(audio):
        upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)))

    return upsampled


def simulate_mp3_compression(
    audio: NDArray[np.float64],
    sr: int,
    bitrate: int = 128,
) -> NDArray[np.float64]:
    """
    Simulate MP3 compression by encoding and decoding through pydub/ffmpeg.

    Falls back to low-pass filter approximation if ffmpeg is unavailable.

    Args:
        audio: Input audio signal (mono, float64, normalized to [-1, 1]).
        sr: Sample rate in Hz.
        bitrate: MP3 bitrate in kbps.

    Returns:
        Audio after MP3 round-trip (may differ slightly in length).
    """
    try:
        import os

        import librosa
        import soundfile as sf
        from pydub import AudioSegment
        from pydub.utils import which

        if which("ffmpeg") is None:
            logger.warning("ffmpeg not found, approximating MP3 with low-pass filter")
            cutoff = min(bitrate * 50, sr // 2 - 1)
            return simulate_low_pass_filter(audio, sr, cutoff_hz=cutoff)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
            wav_path = wav_f.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
            mp3_path = mp3_f.name

        sf.write(wav_path, audio, sr, format="WAV")
        seg = AudioSegment.from_wav(wav_path)
        seg.export(mp3_path, format="mp3", bitrate=f"{bitrate}k")

        decoded, _ = librosa.load(mp3_path, sr=sr, mono=True)

        os.unlink(wav_path)
        os.unlink(mp3_path)

        return decoded

    except Exception as e:
        logger.warning(f"MP3 simulation failed ({e}), using low-pass approximation")
        cutoff = min(bitrate * 50, sr // 2 - 1)
        return simulate_low_pass_filter(audio, sr, cutoff_hz=cutoff)


def augment_and_check_survival(
    original: NDArray[np.float64],
    perturbation: NDArray[np.float64],
    sr: int,
) -> Dict[str, float]:
    """
    Test how well a perturbation survives common post-processing transforms.

    Args:
        original: Clean original audio.
        perturbation: The perturbation signal (noise only, not mixed).
        sr: Sample rate.

    Returns:
        Dict mapping transform name to perturbation survival ratio (0-1).
        A ratio of 1.0 means the perturbation fully survives.
    """
    protected = original + perturbation
    original_perturbation_power = np.mean(perturbation**2)

    if original_perturbation_power < 1e-20:
        return {"low_pass": 0.0, "resample": 0.0, "mp3_128k": 0.0}

    results = {}

    # Test low-pass filter
    filtered = simulate_low_pass_filter(protected, sr, cutoff_hz=8000)
    min_len = min(len(filtered), len(original))
    residual = filtered[:min_len] - original[:min_len]
    surviving_power = np.mean(residual**2)
    results["low_pass"] = float(min(1.0, surviving_power / original_perturbation_power))

    # Test resampling
    resampled = simulate_resampling(protected, sr, target_sr=16000)
    min_len = min(len(resampled), len(original))
    residual = resampled[:min_len] - original[:min_len]
    surviving_power = np.mean(residual**2)
    results["resample"] = float(min(1.0, surviving_power / original_perturbation_power))

    # Test MP3 compression
    mp3_result = simulate_mp3_compression(protected, sr, bitrate=128)
    min_len = min(len(mp3_result), len(original))
    residual = mp3_result[:min_len] - original[:min_len]
    surviving_power = np.mean(residual**2)
    results["mp3_128k"] = float(min(1.0, surviving_power / original_perturbation_power))

    return results

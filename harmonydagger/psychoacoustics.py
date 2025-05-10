"""
Psychoacoustic modeling functions for HarmonyDagger.
"""
import numpy as np
from numpy.typing import NDArray

from .common import (
    BARK_SCALE_C1,
    BARK_SCALE_C2,
    BARK_SCALE_C3,
    BARK_SCALE_F_DIV,
    CBW_C1,
    CBW_C2,
    CBW_C3,
    CBW_F_POW,
    DB_LOG_EPSILON,
    HEARING_THRESH_C1,
    HEARING_THRESH_C2,
    HEARING_THRESH_EXP_C1,
    HEARING_THRESH_F_OFFSET,
    HEARING_THRESH_F_POW,
    HZ_TO_KHZ,
    REFERENCE_PRESSURE,
)


def hearing_threshold(frequency_hz: float) -> float:
    """
    Calculate the absolute hearing threshold in dB SPL at a given frequency.

    This implements a simplified model of human hearing threshold
    based on ISO 226:2003 equal-loudness contours.

    Args:
        frequency_hz: Frequency in Hz

    Returns:
        Hearing threshold in dB SPL
    """
    # Avoid divide by zero in power calculation with small epsilon
    f_khz = max(frequency_hz / HZ_TO_KHZ, 1e-6)
    threshold_db = (
        HEARING_THRESH_C1 * (f_khz**HEARING_THRESH_F_POW) -
        HEARING_THRESH_C2 * np.exp(HEARING_THRESH_EXP_C1 * ((f_khz - HEARING_THRESH_F_OFFSET)**2))
    )
    return threshold_db


def bark_scale(frequency_hz: float) -> float:
    """
    Convert frequency in Hz to Bark scale.

    The Bark scale is a psychoacoustic scale that matches the critical bands
    of human hearing, which is important for masking effects.

    Args:
        frequency_hz: Frequency in Hz

    Returns:
        Frequency in Bark scale
    """
    return (
        BARK_SCALE_C1 * np.arctan(BARK_SCALE_C2 * frequency_hz) +
        BARK_SCALE_C3 * np.arctan((frequency_hz / BARK_SCALE_F_DIV)**2)
    )


def critical_band_width(center_frequency_hz: float) -> float:
    """
    Calculate the width of the critical band at a given center frequency.

    Args:
        center_frequency_hz: Center frequency in Hz

    Returns:
        Critical bandwidth in Hz
    """
    # Simplified critical bandwidth equation based on Zwicker's model
    f_khz = center_frequency_hz / HZ_TO_KHZ
    return CBW_C1 + CBW_C2 * (1 + CBW_C3 * (f_khz**2))**CBW_F_POW


def magnitude_to_db(magnitude: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert linear magnitude to dB SPL.

    Args:
        magnitude: Linear magnitude values

    Returns:
        Magnitude in dB SPL
    """
    # Avoid log(0) by setting a minimum value
    magnitude = np.maximum(magnitude, DB_LOG_EPSILON)
    db = 20 * np.log10(magnitude / REFERENCE_PRESSURE) # 20 is standard for dB SPL from pressure
    return db


def db_to_magnitude(db: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert dB SPL back to linear magnitude.

    Args:
        db: Values in dB SPL

    Returns:
        Linear magnitude values
    """
    # Protect against overflow by clipping extremely high dB values
    # 350 dB corresponds to 10^17.5, close to float64 max (~10^308)
    db_clipped = np.minimum(db, 350.0)
    return 10 ** (db_clipped / 20.0) * REFERENCE_PRESSURE # 20.0 is standard for dB SPL to pressure

"""
Audio analysis functions for HarmonyDagger.
"""
import numpy as np
from numpy.typing import NDArray


def is_zero_crossing(x: NDArray[np.float64], threshold: float = 0.0, zero_pos: bool = True) -> bool:
    """
    Determine if an audio segment contains a zero crossing.
    
    Args:
        x: Audio segment (usually 2 samples) as numpy array
        threshold: Threshold to consider value as zero
        zero_pos: Whether to use numpy's signbit (True) or sign (False) function
            
    Returns:
        True if a zero crossing is detected, False otherwise
    """
    # Handle special cases to avoid signbit/sign issues
    if len(x) < 2:
        return False
    
    # Since we're looking at transitions, we need at least two samples
    x0 = x[0]
    x1 = x[-1]
    
    # Apply threshold to avoid noise-related crossings
    x0 = 0 if -threshold <= x0 <= threshold else x0
    x1 = 0 if -threshold <= x1 <= threshold else x1
    
    # If either sample is zero, they must have different signs to be a crossing
    if x0 == 0 or x1 == 0:
        return (x0 < 0 and x1 > 0) or (x0 > 0 and x1 < 0)
        
    # Otherwise check if the sign changes
    return (x0 < 0 and x1 > 0) or (x0 > 0 and x1 < 0)


def count_zero_crossings(audio: NDArray[np.float64], threshold: float = 0.0) -> int:
    """
    Count the number of zero crossings in an audio signal.
    
    Zero crossings are points where the signal changes from positive to negative or vice versa.
    
    Args:
        audio: Audio signal as numpy array
        threshold: Threshold to consider value as zero
        
    Returns:
        Number of zero crossings
    """
    # Apply threshold to avoid counting crossings due to low-level noise
    audio_thresholded = np.copy(audio)
    audio_thresholded[np.abs(audio) < threshold] = 0
    
    # Count sign changes between consecutive samples
    crossings = 0
    for i in range(1, len(audio_thresholded)):
        if (audio_thresholded[i-1] > 0 and audio_thresholded[i] < 0) or \
           (audio_thresholded[i-1] < 0 and audio_thresholded[i] > 0):
            crossings += 1
    
    return crossings


def get_audio_statistics(audio: NDArray[np.float64]) -> dict:
    """
    Calculate basic statistics of an audio signal.
    
    Args:
        audio: Audio signal as numpy array
        
    Returns:
        Dictionary of statistics
    """
    return {
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
        "mean": float(np.mean(audio)),
        "std": float(np.std(audio)),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "zero_crossings": count_zero_crossings(audio),
        "duration_samples": len(audio)
    }

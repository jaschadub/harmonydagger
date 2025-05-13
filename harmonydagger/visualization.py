"""
Visualization functions for HarmonyDagger.
"""
import logging
import os
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .common import (
    DEFAULT_FIGSIZE_DIFFERENCE,
    DEFAULT_FIGSIZE_SPECTROGRAM,
    DEFAULT_VIS_DPI,
    DEFAULT_VIS_NFFT,
    DEFAULT_VIS_NOVERLAP,
    DIFFERENCE_SUFFIX,
    SPECTROGRAM_SUFFIX,
)


def visualize_spectrograms(
    original: NDArray[np.float64],
    perturbed: NDArray[np.float64],
    sr: int,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the spectrograms of original and perturbed audio.
    """
    plt.figure(figsize=DEFAULT_FIGSIZE_SPECTROGRAM)

    plt.subplot(2, 1, 1)
    plt.specgram(original, Fs=sr, NFFT=DEFAULT_VIS_NFFT, noverlap=DEFAULT_VIS_NOVERLAP, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Audio")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.subplot(2, 1, 2)
    plt.specgram(perturbed, Fs=sr, NFFT=DEFAULT_VIS_NFFT, noverlap=DEFAULT_VIS_NOVERLAP, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Perturbed Audio (with HarmonyDagger protection)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_VIS_DPI, bbox_inches='tight')
        logging.info(f"Spectrogram saved to: {output_path}")
    else:
        plt.show()


def visualize_difference(
    original: NDArray[np.float64],
    perturbed: NDArray[np.float64],
    sr: int,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the difference between original and perturbed audio.
    """
    difference = perturbed - original

    plt.figure(figsize=DEFAULT_FIGSIZE_DIFFERENCE)

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(original)) / sr, original)
    plt.title("Original Audio Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(difference)) / sr, difference)
    plt.title("Difference (Added Noise)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.specgram(difference, Fs=sr, NFFT=DEFAULT_VIS_NFFT, noverlap=DEFAULT_VIS_NOVERLAP, cmap='hot')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Added Noise")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_VIS_DPI, bbox_inches='tight')
        logging.info(f"Difference visualization saved to: {output_path}")
    else:
        plt.show()


def create_audio_comparison(
    original_path: str,
    perturbed_path: str,
    output_dir: Optional[str] = None,
    visualize_spectrogram: bool = True,
    visualize_diff: bool = False
) -> None:
    """
    Create comparison visualizations between original and perturbed audio files.
    
    Args:
        original_path: Path to the original audio file
        perturbed_path: Path to the perturbed audio file
        output_dir: Directory to save visualizations (defaults to original file directory)
        visualize_spectrogram: Whether to generate a spectrogram comparison
        visualize_diff: Whether to generate a difference visualization
    """
    # Set default output directory if not provided
    if output_dir is None or output_dir == '':
        output_dir = os.path.dirname(original_path)
        if not output_dir:  # If still empty, use current directory
            output_dir = '.'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the directory being used
    logging.debug(f"Visualization output directory: {output_dir}")
    
    # Load audio files
    try:
        # Use the same sample rate for both files (using the original's sample rate)
        y_orig, sr = librosa.load(original_path, sr=None)
        y_pert, _ = librosa.load(perturbed_path, sr=sr)  # Use the same sample rate
        
        # Ensure both have the same length
        min_length = min(len(y_orig), len(y_pert))
        y_orig = y_orig[:min_length]
        y_pert = y_pert[:min_length]
        
        # Prepare file names for visualizations
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # Generate visualizations
        if visualize_spectrogram:
            spec_path = os.path.join(output_dir, f"{base_name}{SPECTROGRAM_SUFFIX}")
            visualize_spectrograms(y_orig, y_pert, sr, spec_path)
        
        if visualize_diff:
            diff_path = os.path.join(output_dir, f"{base_name}{DIFFERENCE_SUFFIX}")
            visualize_difference(y_orig, y_pert, sr, diff_path)
            
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}")

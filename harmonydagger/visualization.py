"""
Visualization functions for HarmonyDagger.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import logging

from .common import (
    DEFAULT_FIGSIZE_SPECTROGRAM,
    DEFAULT_FIGSIZE_DIFFERENCE,
    DEFAULT_VIS_NFFT,
    DEFAULT_VIS_NOVERLAP,
    DEFAULT_VIS_DPI
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

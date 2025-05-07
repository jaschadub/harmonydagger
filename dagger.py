#!/usr/bin/env python3
"""
HarmonyDagger: Making Music Unlearnable for Generative AI

This tool implements the HarmonyCloak approach, introducing imperceptible
psychoacoustic noise into audio files to protect them from being learned
by generative AI models, while preserving perceptual quality for human listeners.

Based on research paper: "HarmonyCloak: Making Music Unlearnable for Generative AI"
by Syed Irfan Ali Meerza, Lichao Sun, and Jian Liu
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, List, Union, Optional

# Constants for psychoacoustic modeling
REFERENCE_PRESSURE = 20e-6  # Reference pressure in air (20 μPa)
MASKING_CURVE_SLOPE = 0.8   # Slope for frequency masking (dB/Bark)

# ===== Psychoacoustic Modeling Functions =====

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
    f = frequency_hz / 1000.0  # Convert to kHz
    threshold_db = 3.64 * (f ** -0.8) - 6.5 * np.exp(-0.6 * ((f - 3.3) ** 2))
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
    return 13 * np.arctan(0.00076 * frequency_hz) + 3.5 * np.arctan((frequency_hz / 7500.0) ** 2)


def critical_band_width(center_frequency_hz: float) -> float:
    """
    Calculate the width of the critical band at a given center frequency.
    
    Args:
        center_frequency_hz: Center frequency in Hz
        
    Returns:
        Critical bandwidth in Hz
    """
    # Simplified critical bandwidth equation based on Zwicker's model
    return 25 + 75 * (1 + 1.4 * (center_frequency_hz / 1000.0) ** 2) ** 0.69


# ===== Audio Processing Utility Functions =====

def magnitude_to_db(magnitude: np.ndarray) -> np.ndarray:
    """
    Convert linear magnitude to dB SPL.
    
    Args:
        magnitude: Linear magnitude values
        
    Returns:
        Magnitude in dB SPL
    """
    # Avoid log(0) by setting a minimum value
    magnitude = np.maximum(magnitude, 1e-10)
    db = 20 * np.log10(magnitude / REFERENCE_PRESSURE)
    return db


def db_to_magnitude(db: np.ndarray) -> np.ndarray:
    """
    Convert dB SPL back to linear magnitude.
    
    Args:
        db: Values in dB SPL
        
    Returns:
        Linear magnitude values
    """
    return 10 ** (db / 20.0) * REFERENCE_PRESSURE


# ===== Noise Generation Functions =====

def generate_psychoacoustic_noise(
    audio: np.ndarray, 
    sr: int, 
    window_size: int = 1024, 
    hop_size: int = 512, 
    noise_scale: float = 0.01,
    adaptive_scaling: bool = True
) -> np.ndarray:
    """
    Generate psychoacoustically masked noise for a single audio channel.
    
    This function analyzes the audio using STFT and generates noise
    specifically designed to be masked by the dominant frequencies,
    making it imperceptible to humans but disruptive to AI models.
    
    Args:
        audio: Audio data as numpy array (single channel)
        sr: Sample rate in Hz
        window_size: STFT window size in samples
        hop_size: STFT hop size in samples
        noise_scale: Scaling factor for noise (0-1)
        adaptive_scaling: Whether to adjust noise level adaptively
        
    Returns:
        Generated noise as numpy array
    """
    # Calculate overlap for STFT
    overlap = window_size - hop_size
    
    # Perform Short-Time Fourier Transform (STFT)
    freqs, times, stft_matrix = stft(
        audio, 
        fs=sr, 
        nperseg=window_size, 
        noverlap=overlap
    )
    
    # Extract magnitude and phase components
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    # Initialize noise matrix
    noise = np.zeros_like(magnitude)
    
    # Calculate critical band widths for each frequency
    bark_freqs = np.array([bark_scale(f) for f in freqs])
    
    # Process each time frame
    for t in range(magnitude.shape[1]):
        # Extract magnitude frame at time t
        mag_frame = magnitude[:, t]
        
        # Find dominant frequency
        dom_freq_idx = np.argmax(mag_frame)
        dom_freq_hz = freqs[dom_freq_idx]
        dom_freq_bark = bark_scale(dom_freq_hz)
        
        # Determine masking band size (in frequency bins)
        # This is adaptive based on the critical band width at the dominant frequency
        cb_width = critical_band_width(dom_freq_hz)
        masking_band = max(1, int(cb_width / (sr / window_size)))
        
        # Apply frequency masking within the critical band
        for offset in range(-masking_band, masking_band + 1):
            idx = dom_freq_idx + offset
            
            # Ensure index is within valid range
            if 0 <= idx < magnitude.shape[0]:
                # Calculate frequency distance in Bark scale
                freq_dist_bark = abs(bark_freqs[idx] - dom_freq_bark)
                
                # Calculate masking threshold based on distance (simplified masking curve)
                masking_attenuation = MASKING_CURVE_SLOPE * freq_dist_bark
                
                # Get hearing threshold at this frequency
                hearing_db = hearing_threshold(freqs[idx])
                hearing_mag = db_to_magnitude(hearing_db)
                
                # Get magnitude at this frequency
                mag_db = magnitude_to_db(mag_frame[idx])
                mag_val = db_to_magnitude(mag_db)
                
                # Apply adaptive scaling if enabled
                effective_scale = noise_scale
                if adaptive_scaling:
                    # Scale noise based on signal strength (stronger signal -> more noise)
                    signal_strength = mag_db - hearing_db
                    if signal_strength > 0:
                        # Normalize to a reasonable range (0.5-1.5 * noise_scale)
                        adaptive_factor = 0.5 + min(1.0, signal_strength / 60)
                        effective_scale = noise_scale * adaptive_factor
                
                # Calculate noise level: between hearing threshold and masking threshold
                noise_mag = np.clip(
                    effective_scale * mag_val * (1.0 - masking_attenuation/20), 
                    hearing_mag,  # Lower bound: hearing threshold
                    0.8 * mag_val  # Upper bound: slightly below original magnitude
                )
                
                # Add noise to the frequency bin
                noise[idx, t] = noise_mag
    
    # Convert noise back to time domain using inverse STFT
    noise_stft = noise * np.exp(1j * phase)
    _, noise_audio = istft(
        noise_stft, 
        fs=sr, 
        nperseg=window_size, 
        noverlap=overlap
    )
    
    # Ensure noise length matches original audio
    if len(noise_audio) > len(audio):
        noise_audio = noise_audio[:len(audio)]
    elif len(noise_audio) < len(audio):
        # Pad with zeros if needed
        noise_audio = np.pad(noise_audio, (0, len(audio) - len(noise_audio)))
    
    return noise_audio


def apply_noise_multichannel(
    audio: np.ndarray, 
    sr: int, 
    window_size: int, 
    hop_size: int, 
    noise_scale: float,
    adaptive_scaling: bool = True
) -> np.ndarray:
    """
    Process multi-channel audio by applying psychoacoustic noise to each channel.
    
    Args:
        audio: Audio data as numpy array (can be mono or stereo)
        sr: Sample rate in Hz
        window_size: STFT window size in samples
        hop_size: STFT hop size in samples
        noise_scale: Scaling factor for noise (0-1)
        adaptive_scaling: Whether to adjust noise level adaptively
        
    Returns:
        Perturbed audio with noise added
    """
    # Process mono audio
    if audio.ndim == 1:
        noise = generate_psychoacoustic_noise(
            audio, 
            sr, 
            window_size, 
            hop_size, 
            noise_scale,
            adaptive_scaling
        )
        return apply_noise_to_audio(audio, noise)
    
    # Process multi-channel audio
    else:
        noisy_channels = []
        for ch in range(audio.shape[0]):
            noise = generate_psychoacoustic_noise(
                audio[ch], 
                sr, 
                window_size, 
                hop_size, 
                noise_scale,
                adaptive_scaling
            )
            noisy = apply_noise_to_audio(audio[ch], noise)
            noisy_channels.append(noisy)
        return np.vstack(noisy_channels)


def apply_noise_to_audio(audio: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Apply generated noise to audio signal and prevent clipping.
    
    Args:
        audio: Original audio signal
        noise: Noise signal to add
        
    Returns:
        Perturbed audio with noise added
    """
    perturbed_audio = audio + noise
    return np.clip(perturbed_audio, -1.0, 1.0)


# ===== Visualization Functions =====

def visualize_spectrograms(
    original: np.ndarray, 
    perturbed: np.ndarray, 
    sr: int,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the spectrograms of original and perturbed audio.
    
    Args:
        original: Original audio data
        perturbed: Perturbed audio data
        sr: Sample rate in Hz
        output_path: If provided, save plot to this path instead of displaying
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original audio spectrogram
    plt.subplot(2, 1, 1)
    plt.specgram(original, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Audio")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    # Plot perturbed audio spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(perturbed, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Perturbed Audio (with HarmonyDagger protection)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to: {output_path}")
    else:
        plt.show()


def visualize_difference(
    original: np.ndarray, 
    perturbed: np.ndarray, 
    sr: int,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the difference between original and perturbed audio.
    
    Args:
        original: Original audio data
        perturbed: Perturbed audio data
        sr: Sample rate in Hz
        output_path: If provided, save plot to this path instead of displaying
    """
    # Compute the difference
    difference = perturbed - original
    
    plt.figure(figsize=(12, 10))
    
    # Plot original waveform
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(original)) / sr, original)
    plt.title("Original Audio Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    # Plot difference waveform
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(difference)) / sr, difference)
    plt.title("Difference (Added Noise)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    # Plot spectrogram of the difference
    plt.subplot(3, 1, 3)
    plt.specgram(difference, Fs=sr, NFFT=1024, noverlap=512, cmap='hot')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Added Noise")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Difference visualization saved to: {output_path}")
    else:
        plt.show()


# ===== File Processing Functions =====

def process_audio_file(
    input_file: str, 
    output_file: str, 
    window_size: int = 1024, 
    hop_size: int = 512, 
    noise_scale: float = 0.01,
    force_mono: bool = False,
    adaptive_scaling: bool = True,
    visualize: bool = False,
    visualize_diff: bool = False,
    visualization_path: Optional[str] = None
) -> Tuple[float, float]:
    """
    Process a single audio file and add psychoacoustic noise.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to save output audio file
        window_size: STFT window size in samples
        hop_size: STFT hop size in samples
        noise_scale: Scaling factor for noise (0-1)
        force_mono: Whether to convert stereo to mono
        adaptive_scaling: Whether to use adaptive noise scaling
        visualize: Whether to visualize spectrograms
        visualize_diff: Whether to visualize the difference
        visualization_path: Directory to save visualizations
        
    Returns:
        Tuple of (processing_time, file_size_ratio)
    """
    start_time = time.time()
    
    try:
        # Load audio file
        print(f"Loading input file: {input_file}")
        audio, sr = sf.read(input_file, always_2d=True)
        audio = audio.T  # (channels, samples)
        
        # Store original file size
        original_size = os.path.getsize(input_file)
        
        # Convert to mono if requested
        if force_mono and audio.shape[0] > 1:
            print("Converting to mono...")
            audio = np.mean(audio, axis=0, keepdims=True)
        
        # Process audio
        print(f"Generating psychoacoustic noise with scale: {noise_scale}")
        additional_info = "with adaptive scaling" if adaptive_scaling else ""
        print(f"Processing audio {additional_info}...")
        
        perturbed = apply_noise_multichannel(
            audio, sr, window_size, hop_size, noise_scale, adaptive_scaling
        )
        perturbed = perturbed.T  # (samples, channels)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save output file
        print(f"Saving perturbed audio to: {output_file}")
        sf.write(output_file, perturbed, sr)
        
        # Calculate processing time and file size ratio
        processing_time = time.time() - start_time
        new_size = os.path.getsize(output_file)
        file_size_ratio = new_size / original_size
        
        # Create visualizations
        if visualize or visualize_diff:
            # Create visualization directory if it doesn't exist and path is provided
            if visualization_path:
                os.makedirs(visualization_path, exist_ok=True)
            
            if visualize:
                # Determine visualization output path
                spec_output = None
                if visualization_path:
                    base_name = os.path.splitext(os.path.basename(output_file))[0]
                    spec_output = os.path.join(visualization_path, f"{base_name}_spectrogram.png")
                
                visualize_spectrograms(audio[0], perturbed.T[0], sr, spec_output)
            
            if visualize_diff:
                # Determine difference visualization output path
                diff_output = None
                if visualization_path:
                    base_name = os.path.splitext(os.path.basename(output_file))[0]
                    diff_output = os.path.join(visualization_path, f"{base_name}_difference.png")
                
                visualize_difference(audio[0], perturbed.T[0], sr, diff_output)
        
        print(f"Process completed successfully in {processing_time:.2f} seconds.")
        return processing_time, file_size_ratio
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return 0, 0


def batch_process(
    input_dir: str, 
    output_dir: str,
    window_size: int = 1024, 
    hop_size: int = 512, 
    noise_scale: float = 0.01,
    force_mono: bool = False,
    adaptive_scaling: bool = True,
    file_extension: str = ".wav",
    visualize: bool = False,
    visualize_diff: bool = False
) -> None:
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed files
        window_size: STFT window size in samples
        hop_size: STFT hop size in samples
        noise_scale: Scaling factor for noise (0-1)
        force_mono: Whether to convert stereo to mono
        adaptive_scaling: Whether to use adaptive noise scaling
        file_extension: File extension to process (e.g., ".wav", ".mp3")
        visualize: Whether to visualize spectrograms
        visualize_diff: Whether to visualize the difference
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    if visualize or visualize_diff:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Find all audio files with the specified extension
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(file_extension)]
    
    if not files:
        print(f"No {file_extension} files found in {input_dir}")
        return
    
    print(f"Found {len(files)} {file_extension} files to process")
    
    # Process each file
    total_time = 0
    file_size_ratios = []
    
    for i, file_name in enumerate(files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"\nProcessing file {i+1}/{len(files)}: {file_name}")
        proc_time, size_ratio = process_audio_file(
            input_path, 
            output_path, 
            window_size, 
            hop_size, 
            noise_scale,
            force_mono,
            adaptive_scaling,
            visualize,
            visualize_diff,
            vis_dir
        )
        
        total_time += proc_time
        if size_ratio > 0:
            file_size_ratios.append(size_ratio)
    
    # Print summary
    if file_size_ratios:
        avg_ratio = sum(file_size_ratios) / len(file_size_ratios)
        print("\nBatch Processing Summary:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average file size ratio: {avg_ratio:.2f}")
        print(f"Processed files saved to: {output_dir}")
        if visualize or visualize_diff:
            print(f"Visualizations saved to: {vis_dir}")


# ===== Main Execution =====

def main(args):
    """
    Main execution function that processes command line arguments.
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Single file processing
    if not args.batch_mode:
        process_audio_file(
            args.input_file,
            args.output_file,
            args.window_size,
            args.hop_size,
            args.noise_scale,
            args.force_mono,
            args.adaptive_scaling,
            args.visualize,
            args.visualize_diff
        )
    
    # Batch processing mode
    else:
        if not args.output_dir:
            print("Error: --output_dir is required for batch processing")
            return
        
        batch_process(
            args.input_dir,
            args.output_dir,
            args.window_size,
            args.hop_size,
            args.noise_scale,
            args.force_mono,
            args.adaptive_scaling,
            args.file_extension,
            args.visualize,
            args.visualize_diff
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HarmonyDagger - Make Music Unlearnable for Generative AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    input_group = parser.add_argument_group("Input/Output")
    input_group.add_argument("--batch_mode", action="store_true", 
                            help="Process all audio files in a directory")
    input_group.add_argument("--input_dir", type=str,
                            help="Directory containing input audio files (for batch mode)")
    input_group.add_argument("--output_dir", type=str,
                            help="Directory to save processed files (for batch mode)")
    input_group.add_argument("--file_extension", type=str, default=".wav",
                            help="File extension to process in batch mode")
    
    # Make input_file and output_file optional for batch mode
    input_group.add_argument("input_file", type=str, nargs="?",
                            help="Path to input audio file")
    input_group.add_argument("output_file", type=str, nargs="?",
                            help="Path to save perturbed audio file")
    
    # Processing parameters
    param_group = parser.add_argument_group("Processing Parameters")
    param_group.add_argument("--window_size", type=int, default=1024,
                            help="STFT window size")
    param_group.add_argument("--hop_size", type=int, default=512,
                            help="STFT hop size")
    param_group.add_argument("--noise_scale", type=float, default=0.01,
                            help="Noise scale factor (0-1)")
    param_group.add_argument("--adaptive_scaling", action="store_true",
                            help="Use adaptive noise scaling based on signal strength")
    param_group.add_argument("--force_mono", action="store_true",
                            help="Force stereo input to mono before processing")
    
    # Visualization options
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument("--visualize", action="store_true",
                          help="Show spectrogram comparison of original and perturbed audio")
    vis_group.add_argument("--visualize_diff", action="store_true",
                          help="Visualize the difference between original and perturbed audio")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_mode:
        if not args.input_dir:
            parser.error("--input_dir is required for batch mode")
    else:
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required for single file mode")
    
    main(args)

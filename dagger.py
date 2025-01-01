import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft

def generate_psychoacoustic_noise(audio, sr, window_size=1024, hop_size=512, noise_scale=0.01):
    """
    Generate imperceptible noise aligned with dominant frequencies using psychoacoustic masking.
    """
    # Short-Time Fourier Transform (STFT)
    freqs, times, stft_matrix = stft(audio, sr, nperseg=window_size, noverlap=hop_size)
    
    # Magnitude and phase
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    # Identify dominant frequencies
    dominant_frequencies = np.argmax(magnitude, axis=0)
    
    # Generate noise aligned with dominant frequencies
    noise = np.zeros_like(magnitude)
    for t, dom_freq in enumerate(dominant_frequencies):
        noise[dom_freq, t] = noise_scale * magnitude[dom_freq, t]  # Scale the noise
    
    # Convert noise back to the time domain
    noise_stft = noise * np.exp(1j * phase)  # Combine noise with original phase
    _, noise_audio = istft(noise_stft, sr, nperseg=window_size, noverlap=hop_size)
    
    return noise_audio

def apply_noise_to_audio(audio, noise):
    """
    Apply the generated noise to the original audio.
    """
    return audio + noise

def main(args):
    # Load the WAV file
    print(f"Loading input file: {args.input_file}")
    audio, sr = librosa.load(args.input_file, sr=None, mono=True)
    
    # Generate psychoacoustic noise
    print(f"Generating noise with scale: {args.noise_scale}")
    noise = generate_psychoacoustic_noise(
        audio, sr, window_size=args.window_size, hop_size=args.hop_size, noise_scale=args.noise_scale
    )
    
    # Apply noise to the original audio
    perturbed_audio = apply_noise_to_audio(audio, noise)
    
    # Normalize the audio to prevent clipping
    perturbed_audio = np.clip(perturbed_audio, -1.0, 1.0)
    
    # Save the perturbed audio
    print(f"Saving perturbed audio to: {args.output_file}")
    sf.write(args.output_file, perturbed_audio, sr)
    print("Process completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HARMONYCLOAK Proof of Concept - Render music unlearnable.")
    parser.add_argument("input_file", type=str, help="Path to the input WAV file.")
    parser.add_argument("output_file", type=str, help="Path to save the perturbed WAV file.")
    parser.add_argument("--window_size", type=int, default=1024, help="Window size for STFT. Default is 1024.")
    parser.add_argument("--hop_size", type=int, default=512, help="Hop size for STFT. Default is 512.")
    parser.add_argument("--noise_scale", type=float, default=0.01, help="Scale of the generated noise. Default is 0.01.")
    
    args = parser.parse_args()
    main(args)

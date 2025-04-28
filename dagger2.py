import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft

# Psychoacoustic hearing threshold function (in dB SPL)
def hearing_threshold(frequency_hz):
    f = frequency_hz / 1000.0
    threshold_db = 3.64 * (f ** -0.8) - 6.5 * np.exp(-0.6 * ((f - 3.3) ** 2))
    return threshold_db

# Convert magnitude to dB SPL assuming 20uPa reference
# (20uPa is the standard reference pressure in air)
def magnitude_to_db(magnitude):
    # Avoid log(0)
    magnitude = np.maximum(magnitude, 1e-10)
    db = 20 * np.log10(magnitude / (20e-6))
    return db

# Convert dB SPL back to linear magnitude
def db_to_magnitude(db):
    return 10 ** (db / 20.0) * 20e-6

def generate_psychoacoustic_noise(audio, sr, window_size=1024, hop_size=512, noise_scale=0.01):
    """
    Generate psychoacoustically masked noise within each window.
    """
    # STFT
    freqs, times, stft_matrix = stft(audio, fs=sr, nperseg=window_size, noverlap=window_size - hop_size)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    noise = np.zeros_like(magnitude)

    for t in range(magnitude.shape[1]):
        # Extract spectrum at time t
        mag_frame = magnitude[:, t]
        
        # Find dominant frequency
        dom_freq_idx = np.argmax(mag_frame)
        dom_freq_hz = freqs[dom_freq_idx]
        
        # Compute hearing threshold at dominant frequency
        hearing_db = hearing_threshold(dom_freq_hz)
        hearing_mag = db_to_magnitude(hearing_db)

        # Compute magnitude at dominant frequency
        mag_db = magnitude_to_db(mag_frame[dom_freq_idx])
        mag_val = db_to_magnitude(mag_db)

        # Set noise between threshold and dominant magnitude
        noise_mag = np.clip(noise_scale * mag_val, hearing_mag, mag_val)

        noise[dom_freq_idx, t] = noise_mag

    # Recombine magnitude and phase
    noise_stft = noise * np.exp(1j * phase)

    # Inverse STFT to time domain
    _, noise_audio = istft(noise_stft, fs=sr, nperseg=window_size, noverlap=window_size - hop_size)

    # Clip length to match original
    noise_audio = noise_audio[:len(audio)]

    return noise_audio


def apply_noise_to_audio(audio, noise):
    """
    Apply generated noise to audio, ensuring no clipping.
    """
    perturbed_audio = audio + noise
    perturbed_audio = np.clip(perturbed_audio, -1.0, 1.0)
    return perturbed_audio


def main(args):
    print(f"Loading input file: {args.input_file}")
    audio, sr = librosa.load(args.input_file, sr=None, mono=True)

    print(f"Generating psychoacoustic noise with scale: {args.noise_scale}")
    noise = generate_psychoacoustic_noise(
        audio,
        sr,
        window_size=args.window_size,
        hop_size=args.hop_size,
        noise_scale=args.noise_scale
    )

    perturbed_audio = apply_noise_to_audio(audio, noise)

    print(f"Saving perturbed audio to: {args.output_file}")
    sf.write(args.output_file, perturbed_audio, sr)
    print("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved HARMONYCLOAK - Psychoacoustic Noise Insertion.")
    parser.add_argument("input_file", type=str, help="Path to input WAV file.")
    parser.add_argument("output_file", type=str, help="Path to output WAV file.")
    parser.add_argument("--window_size", type=int, default=1024, help="STFT window size (default: 1024).")
    parser.add_argument("--hop_size", type=int, default=512, help="STFT hop size (default: 512).")
    parser.add_argument("--noise_scale", type=float, default=0.01, help="Noise scale factor (default: 0.01).")

    args = parser.parse_args()
    main(args)

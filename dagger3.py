import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

# Psychoacoustic hearing threshold function (in dB SPL)
def hearing_threshold(frequency_hz):
    f = frequency_hz / 1000.0
    threshold_db = 3.64 * (f ** -0.8) - 6.5 * np.exp(-0.6 * ((f - 3.3) ** 2))
    return threshold_db

# Convert magnitude to dB SPL assuming 20uPa reference
def magnitude_to_db(magnitude):
    magnitude = np.maximum(magnitude, 1e-10)
    db = 20 * np.log10(magnitude / (20e-6))
    return db

# Convert dB SPL back to linear magnitude
def db_to_magnitude(db):
    return 10 ** (db / 20.0) * 20e-6

# Generate psychoacoustic noise for a single channel
def generate_psychoacoustic_noise(audio, sr, window_size=1024, hop_size=512, noise_scale=0.01):
    freqs, times, stft_matrix = stft(audio, fs=sr, nperseg=window_size, noverlap=window_size - hop_size)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    noise = np.zeros_like(magnitude)

    for t in range(magnitude.shape[1]):
        mag_frame = magnitude[:, t]
        dom_freq_idx = np.argmax(mag_frame)
        dom_freq_hz = freqs[dom_freq_idx]

        masking_band = 5
        for offset in range(-masking_band, masking_band + 1):
            idx = dom_freq_idx + offset
            if 0 <= idx < magnitude.shape[0]:
                hearing_db = hearing_threshold(freqs[idx])
                hearing_mag = db_to_magnitude(hearing_db)
                mag_db = magnitude_to_db(mag_frame[idx])
                mag_val = db_to_magnitude(mag_db)
                noise_mag = np.clip(noise_scale * mag_val, hearing_mag, mag_val)
                noise[idx, t] = noise_mag

    noise_stft = noise * np.exp(1j * phase)
    _, noise_audio = istft(noise_stft, fs=sr, nperseg=window_size, noverlap=window_size - hop_size)
    noise_audio = noise_audio[:len(audio)]

    return noise_audio

# Process mono or stereo audio
def apply_noise_multichannel(audio, sr, window_size, hop_size, noise_scale):
    if audio.ndim == 1:
        noise = generate_psychoacoustic_noise(audio, sr, window_size, hop_size, noise_scale)
        return apply_noise_to_audio(audio, noise)
    else:
        noisy_channels = []
        for ch in range(audio.shape[0]):
            noise = generate_psychoacoustic_noise(audio[ch], sr, window_size, hop_size, noise_scale)
            noisy = apply_noise_to_audio(audio[ch], noise)
            noisy_channels.append(noisy)
        return np.vstack(noisy_channels)

# Add noise and prevent clipping
def apply_noise_to_audio(audio, noise):
    perturbed_audio = audio + noise
    return np.clip(perturbed_audio, -1.0, 1.0)

# Visualize spectrogram difference
def visualize_spectrograms(original, perturbed, sr):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.specgram(original, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Original Audio")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.subplot(1, 2, 2)
    plt.specgram(perturbed, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Perturbed Audio")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

# Main execution
def main(args):
    print(f"Loading input file: {args.input_file}")
    audio, sr = sf.read(args.input_file, always_2d=True)
    audio = audio.T  # (channels, samples)

    if args.force_mono:
        print("Forcing mono...")
        audio = np.mean(audio, axis=0, keepdims=True)

    print(f"Generating psychoacoustic noise with scale: {args.noise_scale}")
    perturbed = apply_noise_multichannel(audio, sr, args.window_size, args.hop_size, args.noise_scale)
    perturbed = perturbed.T  # (samples, channels)

    print(f"Saving perturbed audio to: {args.output_file}")
    sf.write(args.output_file, perturbed, sr)

    if args.visualize:
        visualize_spectrograms(audio[0], perturbed.T[0], sr)

    print("Process completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved HARMONYCLOAK - Psychoacoustic Noise Insertion.")
    parser.add_argument("input_file", type=str, help="Path to input WAV file.")
    parser.add_argument("output_file", type=str, help="Path to output WAV file.")
    parser.add_argument("--window_size", type=int, default=1024, help="STFT window size (default: 1024).")
    parser.add_argument("--hop_size", type=int, default=512, help="STFT hop size (default: 512).")
    parser.add_argument("--noise_scale", type=float, default=0.01, help="Noise scale factor (default: 0.01).")
    parser.add_argument("--force_mono", action="store_true", help="Force stereo input to mono before processing.")
    parser.add_argument("--visualize", action="store_true", help="Show spectrogram comparison of original and perturbed audio.")

    args = parser.parse_args()
    main(args)

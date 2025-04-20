import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft

def generate_psychoacoustic_noise(audio, sr, window_size=1024, hop_size=512, noise_scale=0.01):
    freqs, times, stft_matrix = stft(audio, sr, nperseg=window_size, noverlap=hop_size)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    dominant_frequencies = np.argmax(magnitude, axis=0)
    noise = np.zeros_like(magnitude)
    for t, dom_freq in enumerate(dominant_frequencies):
        noise[dom_freq, t] = noise_scale * magnitude[dom_freq, t]
    noise_stft = noise * np.exp(1j * phase)
    _, noise_audio = istft(noise_stft, sr, nperseg=window_size, noverlap=hop_size)
    return noise_audio

def apply_noise_to_audio(audio, noise):
    min_len = min(len(audio), len(noise))
    return audio[:min_len] + noise[:min_len]

def main(args):
    print(f"Loading input file: {args.input_file}")
    audio, sr = librosa.load(args.input_file, sr=None, mono=False, duration=None)
    print(f"Original audio duration: {audio.shape[1]/sr:.2f} seconds")

    left_channel = audio[0]
    right_channel = audio[1]

    print(f"Generating noise for left channel with scale: {args.noise_scale}")
    left_noise = generate_psychoacoustic_noise(
        left_channel, sr, window_size=args.window_size, hop_size=args.hop_size, noise_scale=args.noise_scale
    )
    
    print(f"Generating noise for right channel with scale: {args.noise_scale}")
    right_noise = generate_psychoacoustic_noise(
        right_channel, sr, window_size=args.window_size, hop_size=args.hop_size, noise_scale=args.noise_scale
    )

    perturbed_left = apply_noise_to_audio(left_channel, left_noise)
    perturbed_right = apply_noise_to_audio(right_channel, right_noise)

    perturbed_left = np.clip(perturbed_left, -1.0, 1.0)
    perturbed_right = np.clip(perturbed_right, -1.0, 1.0)

    perturbed_audio = np.array([perturbed_left, perturbed_right])

    print(f"Saving perturbed audio to: {args.output_file}")
    sf.write(args.output_file, perturbed_audio.T, sr)

    print(f"Done. Output duration: {len(perturbed_left)/sr:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HARMONYCLOAK Proof of Concept - Render music unlearnable.")
    parser.add_argument("input_file", type=str, help="Path to the input WAV file.")
    parser.add_argument("output_file", type=str, help="Path to save the perturbed WAV file.")
    parser.add_argument("--window_size", type=int, default=1024, help="Window size for STFT. Default is 1024.")
    parser.add_argument("--hop_size", type=int, default=512, help="Hop size for STFT. Default is 512.")
    parser.add_argument("--noise_scale", type=float, default=0.01, help="Scale of the generated noise. Default is 0.01.")
    
    args = parser.parse_args()
    main(args)

print(" ███▄    █  ▒█████   ██▓  ██████ ▓█████     ██▓ ███▄    █  ▄▄▄██▀▀▀")
print(" ██ ▀█   █ ▒██▒  ██▒▓██▒▒██    ▒ ▓█   ▀    ▓██▒ ██ ▀█   █    ▒██   ")
print("▓██  ▀█ ██▒▒██░  ██▒▒██▒░ ▓██▄   ▒███      ▒██▒▓██  ▀█ ██▒   ░██   ")
print("▓██▒  ▐▌██▒▒██   ██░░██░  ▒   ██▒▒▓█  ▄    ░██░▓██▒  ▐▌██▒▓██▄██▓  ")
print("▒██░   ▓██░░ ████▓▒░░██░▒██████▒▒░▒████▒   ░██░▒██░   ▓██░ ▓███▒   ")
print("░ ▒░   ▒ ▒ ░ ▒░▒░▒░ ░▓  ▒ ▒▓▒ ▒ ░░░ ▒░ ░   ░▓  ░ ▒░   ▒ ▒  ▒▓▒▒░   ")
print("░ ░░   ░ ▒░  ░ ▒ ▒░  ▒ ░░ ░▒  ░ ░ ░ ░  ░    ▒ ░░ ░░   ░ ▒░ ▒ ░▒░   ")
print("   ░   ░ ░ ░ ░ ░ ▒   ▒ ░░  ░  ░     ░       ▒ ░   ░   ░ ░  ░ ░ ░   ")
print("         ░     ░ ░   ░        ░     ░  ░    ░           ░  ░   ░  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
print(" ┣▇▇▇═─  ┣▇▇▇═─  ┣▇▇▇═─  ┣▇▇▇═─  ┣▇▇▇═─ ")
print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
print("   ██╗ ██████╗  ██████╗ ██╗ ██╗")
print("  ███║██╔═████╗██╔═████╗╚═╝██╔")
print("  ╚██║██║██╔██║██║██╔██║  ██╔╝ ")
print("   ██║████╔╝██║████╔╝██║ ██╔╝  ")
print("   ██║╚██████╔╝╚██████╔╝██╔╝██╗ ")
print("   ╚═╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═╝")
print(" ")
print(" S U C C E S S ")


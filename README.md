# HarmonyDagger

Proof-of-Concept (PoC) implementation inspired by the [HarmonyCloak](https://mosis.eecs.utk.edu/harmonycloak.html) research paper.

`dagger3.py` — Make Music Unlearnable for Generative AI.

**HarmonyDagger** introduces imperceptible psychoacoustic noise into audio files to make them unlearnable by generative AI models, preserving perceptual quality for human listeners.

---

## ✨ Features

- **Psychoacoustic Noise Generation**  
  Aligned with dominant musical frequencies and masked using advanced hearing models.

- **Multi-Channel Audio Support**  
  Works with both mono and stereo WAV files.

- **Optional Mono Conversion**  
  Force stereo input to mono with `--force_mono`.

- **Spectrogram Visualization**  
  View a side-by-side comparison of original and perturbed audio with `--visualize`.

- **STFT-Based Processing**  
  Uses Short-Time Fourier Transform for precise time-frequency domain noise injection.

- **Fully Configurable CLI**  
  Adjust window size, hop size, and noise scale on the fly.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/harmonydagger.git
cd harmonydagger
```

Install dependencies:

```bash
pip install numpy scipy librosa soundfile matplotlib
```

---

## 🚀 Usage

Run the updated script:

```bash
python dagger3.py <input_file> <output_file> [OPTIONS]
```

### Required Arguments
- `input_file`: Path to input WAV file.
- `output_file`: Path to save the perturbed WAV file.

### Optional Arguments
- `--window_size`: STFT window size (default: 1024).
- `--hop_size`: STFT hop size (default: 512).
- `--noise_scale`: Relative strength of the noise (default: 0.01).
- `--force_mono`: Convert stereo audio to mono before processing.
- `--visualize`: Show spectrogram comparison after processing.

### Example

```bash
python dagger3.py input.wav output_perturbed.wav --window_size 2048 --hop_size 1024 --noise_scale 0.02 --visualize
```

---

## 🛠 How It Works

1. **Frequency Analysis**  
   Identifies dominant frequencies in each window using STFT.

2. **Noise Generation**  
   Injects imperceptible noise into masking regions based on psychoacoustic principles.

3. **Noise Injection**  
   Applies noise to audio in mono or stereo channels, preserving perceptual quality.

4. **Spectrogram Visualization (optional)**  
   View the spectral differences introduced by the defense.

---

## 📚 Dependencies

- `numpy`  
- `scipy`  
- `librosa`  
- `soundfile`  
- `matplotlib`

Install them all:

```bash
pip install numpy scipy librosa soundfile matplotlib
```

---

## 📌 Notes

- Supports mono and stereo WAV files.
- `--force_mono` is helpful for consistent behavior across stereo input.
- Spectrogram output is useful for visually inspecting defense effectiveness.

---

## 🔮 Future Work

- Batch processing for folders.
- Export perturbation metrics.
- Integrate masking curves based on critical bands or Bark scale.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repository, improve the code, and submit a pull request.


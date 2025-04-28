# HarmonyDagger

Proof-of-Concept (PoC) implementation inspired by the [HarmonyCloak](https://mosis.eecs.utk.edu/harmonycloak.html) research paper.

`dagger2.py` — Make Music Unlearnable for Generative AI.

HarmonyDagger introduces imperceptible psychoacoustic noise into audio files to protect them from being learned by generative AI models.

---

## ✨ Features

- **Psychoacoustic Noise Generation**: Adds noise aligned with dominant musical frequencies, hidden below human hearing thresholds.
- **STFT-Based Processing**: Analyzes audio using Short-Time Fourier Transform (STFT) for accurate frequency-domain manipulation.
- **Configurable Command-Line Interface**: Easily adjust processing parameters with CLI options.
- **WAV File Support**: Efficiently processes mono WAV audio files.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/harmonydagger.git
cd harmonydagger
```

Install the required Python packages:

```bash
pip install numpy scipy librosa soundfile
```

---

## 🚀 Usage

Basic command:

```bash
python dagger.py <input_file> <output_file> [OPTIONS]
```

### Required Arguments
- `input_file`: Path to the input WAV file.
- `output_file`: Path where the perturbed WAV file will be saved.

### Optional Arguments
- `--window_size`: Window size for STFT (default: 1024).
- `--hop_size`: Hop size for STFT (default: 512).
- `--noise_scale`: Scale of generated noise relative to dominant frequency magnitude (default: 0.01).

### Example

```bash
python dagger.py input.wav output_perturbed.wav --window_size 2048 --hop_size 1024 --noise_scale 0.02
```

---

## 🛠 How It Works

1. **Frequency Analysis**:  
   Performs STFT to detect dominant frequencies over time.

2. **Noise Generation**:  
   Creates imperceptible noise masked by dominant audio content based on psychoacoustic principles.

3. **Noise Injection**:  
   Adds generated noise into the audio without compromising perceptual quality.

4. **Output**:  
   Saves the perturbed, protected audio to the specified path.

---

## 📚 Dependencies

- `numpy`
- `scipy`
- `librosa`
- `soundfile`

Install with:

```bash
pip install numpy scipy librosa soundfile
```

---

## 📌 Notes

- **Input Requirements**:  
  Input audio must be a mono WAV file. (Stereo can be downmixed using `librosa`.)

- **Perceptual Quality**:  
  The perturbed output sounds nearly identical to the original but is resistant to learning by generative AI models.

- **Proof of Concept**:  
  This is an experimental prototype. Further refinements are needed for production or large-scale deployment.

---

## 🔮 Future Work

- Support for multi-channel (stereo) audio.
- Integration with advanced psychoacoustic masking curves.
- Evaluation against real-world generative AI models like MusicLM, MuseGAN, etc.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## 🤝 Contributing

Contributions are welcome!  
Please fork the repository, make your changes, and submit a pull request.


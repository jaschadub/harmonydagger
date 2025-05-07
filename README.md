# HarmonyDagger

Implementation of the [HarmonyCloak](https://mosis.eecs.utk.edu/harmonycloak.html) technique, based on the research paper "HarmonyCloak: Making Music Unlearnable for Generative AI" by Syed Irfan Ali Meerza, Lichao Sun, and Jian Liu.

`dagger.py` — Make Music Unlearnable for Generative AI.

**HarmonyDagger** introduces imperceptible psychoacoustic noise into audio files to make them unlearnable by generative AI models, while preserving perceptual quality for human listeners.

---

## ✨ Features

- **Advanced Psychoacoustic Masking**  
  Uses Bark scale and critical band analysis to optimize noise placement within human hearing thresholds.

- **Adaptive Noise Scaling**  
  Dynamically adjusts noise levels based on signal strength for optimal protection.

- **Multi-Channel Audio Support**  
  Works with both mono and stereo WAV files.

- **Batch Processing**  
  Process entire directories of audio files with a single command.

- **Detailed Visualizations**  
  View spectrograms and waveform differences between original and protected audio.

- **STFT-Based Processing**  
  Uses Short-Time Fourier Transform for precise time-frequency domain noise injection.

- **Comprehensive Error Handling**  
  Robust file processing with detailed error reporting.

- **Configurable CLI**  
  Well-organized command-line interface with sensible defaults.

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

### Single File Processing

Process a single audio file:

```bash
python dagger.py input.wav output.wav [OPTIONS]
```

### Batch Processing

Process all audio files in a directory:

```bash
python dagger.py --batch_mode --input_dir /path/to/input --output_dir /path/to/output [OPTIONS]
```

### Required Arguments (Single File Mode)

- `input_file`: Path to input WAV file.
- `output_file`: Path to save the perturbed WAV file.

### Batch Processing Arguments

- `--batch_mode`: Enable batch processing mode
- `--input_dir`: Directory containing input audio files
- `--output_dir`: Directory to save processed files
- `--file_extension`: File extension to process (default: ".wav")

### Processing Options

- `--window_size`: STFT window size (default: 1024)
- `--hop_size`: STFT hop size (default: 512)
- `--noise_scale`: Relative strength of the noise (default: 0.01)
- `--adaptive_scaling`: Use adaptive noise scaling based on signal strength
- `--force_mono`: Convert stereo audio to mono before processing

### Visualization Options

- `--visualize`: Show spectrogram comparison of original and perturbed audio
- `--visualize_diff`: Visualize the difference between original and perturbed audio

### Example Commands

```bash
# Basic protection with default settings
python dagger.py input.wav output.wav

# Apply stronger protection with adaptive scaling
python dagger.py input.wav output.wav --noise_scale 0.02 --adaptive_scaling

# Process with visualization
python dagger.py input.wav output.wav --visualize --visualize_diff

# Batch process all WAV files with custom settings
python dagger.py --batch_mode --input_dir ./music --output_dir ./protected --noise_scale 0.015 --adaptive_scaling
```

---

## 🛠 How It Works

1. **Psychoacoustic Analysis**  
   Audio is analyzed using STFT and converted to psychoacoustic Bark scale.

2. **Critical Band Calculation**  
   Critical bands are identified based on dominant frequencies in each time window.

3. **Masking Threshold Determination**  
   For each frequency bin, masking thresholds are calculated based on hearing model.

4. **Adaptive Noise Generation**  
   Noise is precisely generated to stay between the hearing threshold and masking threshold.

5. **Multi-channel Processing**  
   Each audio channel is processed independently to maintain stereo image.

6. **Visualization (optional)**  
   Spectrograms and difference analysis show the impact of protection.

---

## 📚 Dependencies

- `numpy`: Numerical processing
- `scipy`: Signal processing functions
- `librosa`: Audio analysis
- `soundfile`: Audio file I/O
- `matplotlib`: Visualization
- `typing`: Type annotations

Install them all:

```bash
pip install numpy scipy librosa soundfile matplotlib
```

---

## 📌 Technical Details

- **Bark Scale**: Used for psychoacoustic frequency mapping
- **Critical Bands**: Calculated based on Zwicker's model
- **Hearing Threshold**: Based on ISO 226:2003 equal-loudness contours
- **Adaptive Scaling**: Adjusts noise based on signal-to-threshold ratio

---

## 🔮 Future Work

- Integration with common DAWs as plugins
- Support for additional audio formats
- Real-time processing capability
- User interface for non-technical users
- Evaluation against state-of-the-art AI music models

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repository, improve the code, and submit a pull request.

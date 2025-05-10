# HarmonyDagger

Implementation of the [HarmonyCloak](https://mosis.eecs.utk.edu/harmonycloak.html) technique, based on the research paper "HarmonyCloak: Making Music Unlearnable for Generative AI" by Syed Irfan Ali Meerza, Lichao Sun, and Jian Liu.

`dagger.py` — Make Music Unlearnable for Generative AI.

**HarmonyDagger** introduces imperceptible psychoacoustic noise into audio files to make them unlearnable by generative AI models, while preserving perceptual quality for human listeners.

<figure>
  <img src="https://i.imgur.com/BIkRLMU.png" alt="How ChatGPT depicts HarmonyDagger" width="500"/>
  <figcaption><em>How ChatGPT depicts HarmonyDagger</em></figcaption>
</figure>


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

### Method 1: Install locally (until published to PyPI)

```bash
# Clone the repository
git clone https://github.com/yourusername/harmonydagger.git
cd harmonydagger

# Install the package (v0.2.0)
pip install -e .
```

### Method 2: Install from PyPI (coming soon)

In the future, the package will be available via:

```bash
pip install harmonydagger
```

### Method 3: Manual Installation

If you don't want to install as a package:

```bash
git clone https://github.com/yourusername/harmonydagger.git
cd harmonydagger
pip install -r requirements.txt  # If a requirements.txt file exists
```

---

## 🚀 Usage

### From Command Line

After installation, you can use HarmonyDagger directly from the command line:

```bash
harmonydagger input.wav output.wav [OPTIONS]
```

### As a Python Module

```python
from harmonydagger.file_operations import process_audio_file, batch_process

# Process single file
process_audio_file('input.wav', 'output.wav', noise_scale=0.015)

# Process directory
batch_process('input_dir', 'output_dir', noise_scale=0.015, parallel=True)
```

### Single File Processing

Process a single audio file:

```bash
harmonydagger input.wav output.wav [OPTIONS]
```

### Batch Processing

Process all audio files in a directory:

```bash
harmonydagger --input_dir /path/to/input --output_dir /path/to/output [OPTIONS]
```

### Parallel Batch Processing

Process files in parallel for significant speed improvements:

```bash
harmonydagger --input_dir /path/to/input --output_dir /path/to/output --parallel [OPTIONS]
```

### Required Arguments (Single File Mode)

- `input_file`: Path to input WAV file.
- `output_file`: Path to save the perturbed WAV file.

### Batch Processing Arguments

- `--input_dir`: Directory containing input audio files
- `--output_dir`: Directory to save processed files
- `--ext`: File extension to process (default: ".wav")
- `--parallel`: Enable parallel processing for batch operations
- `--workers`: Number of worker processes for parallel processing (default: CPU count)

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
harmonydagger input.wav output.wav

# Apply stronger protection with adaptive scaling
harmonydagger input.wav output.wav --noise_scale 0.02 --adaptive_scaling

# Process with visualization
harmonydagger input.wav output.wav --visualize --visualize_diff

# Batch process all WAV files with custom settings
harmonydagger --input_dir ./music --output_dir ./protected --noise_scale 0.015 --adaptive_scaling

# Parallel batch processing for faster results
harmonydagger --input_dir ./music --output_dir ./protected --parallel --workers 4
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

All dependencies are automatically managed when installing via pip:

- `numpy`: Numerical processing
- `scipy`: Signal processing functions
- `soundfile`: Audio file I/O
- `matplotlib`: Visualization

Optional development dependencies:
- `pytest`: Testing
- `black`: Code formatting
- `isort`: Import sorting

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
- GPU acceleration for even faster parallel processing

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repository, improve the code, and submit a pull request.

## 🛠 Performance Optimization

HarmonyDagger includes several performance optimizations:

- **Parallel Processing**: Process multiple files simultaneously using multiprocessing.
- **Efficient Psychoacoustic Calculations**: Pre-calculated frequency transformations.
- **Adaptive Resource Usage**: Automatically scales to available CPU cores.

To get the best performance, use the `--parallel` flag for batch processing:

```bash
harmonydagger --input_dir ./large_collection --output_dir ./protected --parallel
```

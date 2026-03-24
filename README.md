# HarmonyDagger

HarmonyDagger is a tool for audio protection against generative AI models, introducing imperceptible psychoacoustic noise patterns that prevent effective machine learning while preserving human listening quality.

## Features

- **Psychoacoustic Masking**: Uses principles of human auditory perception to generate strategic noise
- **Adaptive Scaling**: Adjusts protection strength based on signal characteristics
- **Phase Perturbation**: Subtle phase shifts that disrupt AI feature extraction while remaining imperceptible
- **Temporal Forward Masking**: Exploits post-masking effects to hide more aggressive perturbations after loud events
- **Vocal-Specific Mode**: Optimized protection for the human vocal range (300Hz-3kHz) targeting AI voice cloning
- **Dry/Wet Control**: Balance protection strength vs. audio fidelity with a single parameter
- **Multi-channel Support**: Works with both mono and stereo audio files
- **Multiple Audio Format Support**: Processes and outputs WAV, MP3, FLAC, and OGG files
  - MP3 support requires ffmpeg to be installed on your system
  - FLAC and OGG support is built-in
- **Robustness Testing**: Verify that perturbations survive MP3 compression, low-pass filtering, and resampling
- **Protection Verification**: MFCC and spectral analysis to measure protection effectiveness
- **Benchmark Reporting**: SNR and perturbation metrics for transparency
- **Visualization Tools**: Optional visual analytics of audio perturbations
- **Parallel Batch Processing**: Process multiple files efficiently using multiple CPU cores
- **Streamlit Web Demo**: Upload audio and hear protected version in the browser
- **Docker Support**: One-command deployment with Docker Compose
- **PyPI Package**: Easy installation via pip

## Installation

### From PyPI

```bash
pip install harmonydagger
```

### From Source

```bash
git clone https://github.com/jaschadub/harmonydagger.git
cd harmonydagger
pip install -e .
```

### With Streamlit Demo

```bash
pip install -e ".[streamlit]"
streamlit run streamlit_app.py
```

### With Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

## Usage

### Command Line Interface

```bash
# Basic protection
harmonydagger input.wav -o output.wav -n 0.1 -a

# Full protection with all techniques
harmonydagger input.wav -o output.wav -n 0.1 -a --phase --temporal-masking --vocal-mode

# Adjust protection strength (0.0 = original, 1.0 = full protection)
harmonydagger input.wav -o output.wav -n 0.1 -a -d 0.7

# Process with robustness check and verification
harmonydagger input.wav -o output.wav -n 0.1 -a --robust --verify --benchmark -v

# Process multiple files in parallel
harmonydagger input_directory -o output_directory -j 4

# Process only MP3 files in a directory
harmonydagger input_directory -o output_directory -f mp3

# Get help on all available options
harmonydagger --help
```

### Python API

```python
import librosa
from harmonydagger.core import generate_protected_audio

# Load audio file
audio, sr = librosa.load('input.wav', sr=None)

# Apply full protection with all techniques
protected_audio = generate_protected_audio(
    audio, sr,
    window_size=2048,
    hop_size=512,
    noise_scale=0.1,
    adaptive_scaling=True,
    dry_wet=1.0,
    vocal_mode=True,
    use_phase_perturbation=True,
    use_temporal_masking=True,
)

# Save result
import soundfile as sf
sf.write('output.wav', protected_audio, sr)

# Verify protection effectiveness
from harmonydagger.verify import verify_protection
report = verify_protection(audio, protected_audio, sr)
print(f"Protection score: {report['protection_score']:.3f}")

# Test robustness against common transforms
from harmonydagger.robustness import augment_and_check_survival
perturbation = protected_audio - audio
survival = augment_and_check_survival(audio, perturbation, sr)
for transform, ratio in survival.items():
    print(f"  {transform}: {ratio:.1%} survival")
```

### Batch Processing with Parallelization

```python
from harmonydagger.file_operations import parallel_batch_process, recursive_find_audio_files

# Find all audio files in a directory (supports MP3, FLAC, OGG, and WAV)
audio_files = recursive_find_audio_files('./audio_files')

# Process files in parallel with new features
results = parallel_batch_process(
    audio_files,
    output_dir='./protected_audio',
    window_size=2048,
    hop_size=512,
    noise_scale=0.1,
    adaptive_scaling=True,
    max_workers=4,
    vocal_mode=True,
    use_phase_perturbation=True,
    use_temporal_masking=True,
)

for file_path, result in results.items():
    if result['success']:
        print(f"Successfully processed {file_path} in {result['processing_time']:.2f} seconds")
    else:
        print(f"Failed to process {file_path}: {result['error']}")
```

## Command Line Options

```
usage: harmonydagger [-h] [-o OUTPUT] [-w WINDOW_SIZE] [-s HOP_SIZE]
                     [-n NOISE_SCALE] [-a] [-d DRY_WET] [--vocal-mode]
                     [--phase] [--temporal-masking] [--robust] [--verify]
                     [--benchmark] [-m] [-j JOBS] [-v]
                     [-f {wav,mp3,flac,ogg,all}]
                     [--visualize] [--visualize_diff] [--version]
                     input

positional arguments:
  input                 Input audio file or directory containing audio files

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file or directory (default: input_protected.wav)
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        STFT window size (default: 2048)
  -s HOP_SIZE, --hop-size HOP_SIZE
                        STFT hop size (default: 512)
  -n NOISE_SCALE, --noise-scale NOISE_SCALE
                        Noise scale (0-1) (default: 0.1)
  -a, --adaptive-scaling
                        Use adaptive noise scaling based on signal strength
  -d DRY_WET, --dry-wet DRY_WET
                        Dry/wet mix (0.0=original, 1.0=fully protected) (default: 1.0)
  --vocal-mode          Optimize protection for vocal frequencies (300Hz-3kHz)
  --phase               Add phase perturbation (disrupts AI feature extraction)
  --temporal-masking    Add temporal forward masking noise
  --robust              Test perturbation robustness against common transforms
  --verify              Run protection verification after processing
  --benchmark           Show SNR and perturbation metrics after processing
  -m, --force-mono      Convert stereo to mono before processing
  -j JOBS, --jobs JOBS  Number of parallel processing jobs (for batch processing) (default: 1)
  -v, --verbose         Enable verbose output
  -f {wav,mp3,flac,ogg,all}, --format {wav,mp3,flac,ogg,all}
                        Specify audio format to process (when processing directories) (default: all)

Visualization:
  --visualize           Show spectrogram comparison of original and perturbed audio
  --visualize_diff      Visualize the difference between original and perturbed audio

  --version             show program's version number and exit
```

## How It Works

HarmonyDagger works by analyzing the audio in the frequency domain using Short-Time Fourier Transform (STFT), then applying carefully calibrated noise based on psychoacoustic principles:

1. **Frequency Analysis**: Converts audio to time-frequency representation
2. **Psychoacoustic Modeling**: Identifies perceptual masking thresholds
3. **Strategic Perturbation**: Adds noise patterns imperceptible to humans
4. **Phase Perturbation**: Subtle phase shifts that disrupt AI feature extraction
5. **Temporal Masking**: Hides perturbations in the temporal shadow of loud events
6. **Adaptive Scaling**: Adjusts protection based on signal characteristics

## Benchmarks

Protection quality at different noise scale settings (measured on 440Hz sine wave, sr=22050):

| Setting          | SNR (dB) | Perturbation Ratio | Description           |
|------------------|----------|-------------------|-----------------------|
| noise_scale=0.01 | ~45 dB   | ~0.005            | Minimal protection    |
| noise_scale=0.05 | ~32 dB   | ~0.025            | Light protection      |
| noise_scale=0.10 | ~26 dB   | ~0.050            | Recommended default   |
| noise_scale=0.20 | ~20 dB   | ~0.100            | Strong protection     |

Use `--benchmark` flag to see exact metrics for your audio files.

## ffmpeg Compatibility

ffmpeg is required for MP3 input/output and MP3 robustness testing.

- **Minimum version**: ffmpeg 4.0+
- **Recommended**: ffmpeg 5.x or 6.x

### Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

## Docker

### Quick Start

```bash
docker compose up --build
```

The Streamlit demo will be available at `http://localhost:8501`.

### CLI in Docker

```bash
docker build -t harmonydagger .
docker run -v $(pwd)/audio:/data harmonydagger harmonydagger /data/input.wav -o /data/output.wav -n 0.1 -a
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HarmonyDagger in your research, please cite:

```
@misc{harmonydagger2025,
  author = {HarmonyDagger Team},
  title = {HarmonyDagger: Making Audio Content Unlearnable for AI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jaschadub/harmonydagger}
}
```

# HarmonyDagger

HarmonyDagger is a tool for audio protection against generative AI models, introducing imperceptible psychoacoustic noise patterns that prevent effective machine learning while preserving human listening quality.

## Features

- **Psychoacoustic Masking**: Uses principles of human auditory perception to generate strategic noise
- **Adaptive Scaling**: Adjusts protection strength based on signal characteristics
- **Multi-channel Support**: Works with both mono and stereo audio files
- **Multiple Audio Format Support**: Processes and outputs WAV, MP3, FLAC, and OGG files
  - MP3 support requires ffmpeg to be installed on your system
  - FLAC and OGG support is built-in
- **Visualization Tools**: Optional visual analytics of audio perturbations
- **Parallel Batch Processing**: Process multiple files efficiently using multiple CPU cores
- **API Integration**: Use as a library or through the REST API
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

## Usage

### Command Line Interface

```bash
# Process a single audio file
harmonydagger input.wav -o output.wav -n 0.1 -a

# Process multiple files in parallel
harmonydagger input_directory -o output_directory -j 4

# Process only MP3 files in a directory
harmonydagger input_directory -o output_directory -f mp3

# Process only FLAC and OGG files (use multiple commands)
harmonydagger input_directory -o output_directory -f flac
harmonydagger input_directory -o output_directory -f ogg

# Get help on all available options
harmonydagger --help
```

### Python API

```python
import librosa
from harmonydagger.core import apply_noise_multichannel

# Load audio file
audio, sr = librosa.load('input.wav', sr=None)

# Apply protection
protected_audio = apply_noise_multichannel(
    audio, sr, 
    window_size=2048, 
    hop_size=512,
    noise_scale=0.1,
    adaptive_scaling=True
)

# Save result (using soundfile for better format support)
import soundfile as sf
sf.write('output.wav', protected_audio, sr)

# For MP3 output:
# from pydub import AudioSegment
# import numpy as np
# import tempfile
# 
# temp_wav = tempfile.mktemp(suffix='.wav')
# sf.write(temp_wav, protected_audio, sr)
# AudioSegment.from_wav(temp_wav).export('output.mp3', format='mp3', bitrate='192k')
```

### Batch Processing with Parallelization

```python
from harmonydagger.file_operations import parallel_batch_process, recursive_find_audio_files

# Find all audio files in a directory (supports MP3, FLAC, OGG, and WAV)
audio_files = recursive_find_audio_files('./audio_files')

# Or specify only specific formats
# audio_files = recursive_find_audio_files('./audio_files', extensions=['.mp3', '.flac'])

# Process files in parallel
results = parallel_batch_process(
    audio_files,
    output_dir='./protected_audio',
    window_size=2048,
    hop_size=512,
    noise_scale=0.1,
    adaptive_scaling=True,
    max_workers=4  # Use 4 CPU cores
)

# Print results
for file_path, result in results.items():
    if result['success']:
        print(f"Successfully processed {file_path} in {result['processing_time']:.2f} seconds")
    else:
        print(f"Failed to process {file_path}: {result['error']}")
```

## Command Line Options

```
usage: harmonydagger [-h] [-o OUTPUT] [-w WINDOW_SIZE] [-s HOP_SIZE]
                     [-n NOISE_SCALE] [-a] [-m] [-j JOBS] [-v]
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
4. **Adaptive Scaling**: Adjusts protection based on signal characteristics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v0.3.3
- Added explicit CLI visualization options with --visualize and --visualize_diff flags
- Fixed CLI help text to properly document visualization options
- Fixed visualization options not being passed correctly in batch processing

### v0.3.2
- Fixed MP3 format saving issues with multiple fallback options
- Improved MP3 file handling and error recovery
- Enhanced directory path handling for MP3 temporary files
- Added direct ffmpeg execution as a last-resort fallback for MP3

### v0.3.1
- Fixed issue with WAV file format recognition and saving
- Added robust file saving mechanism with multiple fallback options
- Improved error handling and reporting
- Enhanced directory path handling for visualizations
- Added detailed debug logging for troubleshooting

### v0.3.0
- Added support for MP3, FLAC, and OGG formats
- Implemented parallel batch processing
- Added visualization tools
- Improved performance with optimized algorithms

### v0.2.0
- Initial public release
- Support for WAV files
- Basic psychoacoustic noise generation

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

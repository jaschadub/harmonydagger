# HarmonyDagger
POC code for HarmonyCloak paper 
https://mosis.eecs.utk.edu/harmonycloak.html

`dagger.py` - Make Music Unlearnable for Generative AI

dagger.py is a proof-of-concept script that demonstrates how to render audio files unlearnable for generative AI models by introducing imperceptible noise.


---

Features

Psychoacoustic Noise Generation: Introduces imperceptible noise aligned with dominant frequencies to protect audio from generative AI learning.

STFT-Based Processing: Uses Short-Time Fourier Transform for frequency analysis and noise insertion.

Command-Line Interface: Fully configurable via CLI options for flexibility.

Supports WAV Files: Handles mono WAV files efficiently.



---

Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/harmonydagger.git
cd harmonydagger
```

2. Install the required Python packages:

`pip install numpy scipy librosa soundfile`




---

Usage

Run the script using the command line:

`python dagger.py <input_file> <output_file> [OPTIONS]`

Required Arguments:

input_file: Path to the input WAV file.

output_file: Path to save the perturbed WAV file.


Optional Arguments:

--window_size: Window size for STFT (default: 1024).

--hop_size: Hop size for STFT overlap (default: 512).

--noise_scale: Scale of the generated noise (default: 0.01).


Example:

`python dagger.py input.wav output_perturbed.wav --window_size 2048 --hop_size 1024 --noise_scale 0.02`


---

How It Works

1. Frequency Analysis:

The script analyzes the input audio file using STFT to identify dominant frequencies.



2. Noise Generation:

Imperceptible noise is generated based on psychoacoustic masking and aligned with the dominant frequencies.



3. Noise Injection:

The noise is added to the original audio while preserving perceptual quality.

4. Output:

The perturbed audio file is saved to the specified location.

---

### Dependencies

numpy
scipy
librosa
soundfile


Install them using:

`pip install numpy scipy librosa soundfile`


---

Notes

Input Audio: Ensure the input audio is in mono WAV format. Stereo files can be converted using tools like librosa.

Output Audio: The perturbed audio retains perceptual quality and is safe for distribution.

Effectiveness: This script is a proof of concept and is intended for experimentation. Further enhancements are required for real-world robustness.



---

Future Work

Support for multi-channel (stereo) WAV files.

Integration with more advanced psychoacoustic models.

Evaluation against specific generative AI models.



---

License

This project is licensed under the MIT License. See the LICENSE file for details.


---

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.






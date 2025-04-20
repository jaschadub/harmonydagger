# HarmonyDagger
POC code for HarmonyCloak paper 
https://mosis.eecs.utk.edu/harmonycloak.html

`dagger.py` - Make Music Unlearnable for Generative AI

dagger.py is a proof-of-concept script that demonstrates how to render audio files unlearnable for generative AI models by introducing imperceptible noise.


---

### Features

Psychoacoustic Noise Generation: Introduces imperceptible noise aligned with dominant frequencies to protect audio from generative AI learning.

STFT-Based Processing: Uses Short-Time Fourier Transform for frequency analysis and noise insertion.

Command-Line Interface: Fully configurable via CLI options for flexibility.

Supports WAV Files: Handles mono WAV files efficiently.



---

### Installation

1. Clone the repository:

```
git clone https://github.com/mjsoa666/harmonydagger.git
cd harmonydagger
```



2. Install the required Python packages:

`pip install numpy scipy librosa soundfile`




---

### ALTERNATIVE INSTALLATION (mac)

create new empty project file
download dagger.py into project file manually
(run the following using command line, replace ''/Users/yourusername/projectfile/'' with path to your project file u just created)
python3 -m venv /Users/yourusername/projectfile/
source /Users/yourusername/projectfile/bin/activate
python3 -m pip install numpy scipy librosa soundfile
????
profit

what this does: install and run everything inside a virtual environment using python3


### Usage

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

Alternate example using python3 virtual environment:
`python3 /Users/mjosa666/virten/dagger.py /Users/mjosa666/Downloads/somethingdarkpsy160-300.wav /Users/mjosa666/daggerout.wav`
(if [OPTIONS] not specified they default to 1024 512 0.01)


---

### How It Works

0.1 Pulls input wav, splits stereo L and R into two mono tracks, does the following stuff to each channel individually before merging them back into a stereo wav output. 


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

### Notes

I̶n̶p̶u̶t̶ ̶A̶u̶d̶i̶o̶:̶ ̶E̶n̶s̶u̶r̶e̶ ̶t̶h̶e̶ ̶i̶n̶p̶u̶t̶ ̶a̶u̶d̶i̶o̶ ̶i̶s̶ ̶i̶n̶ ̶m̶o̶n̶o̶ ̶W̶A̶V̶ ̶f̶o̶r̶m̶a̶t̶.̶ ̶S̶t̶e̶r̶e̶o̶ ̶f̶i̶l̶e̶s̶ ̶c̶a̶n̶ ̶b̶e̶ ̶c̶o̶n̶v̶e̶r̶t̶e̶d̶ ̶u̶s̶i̶n̶g̶ ̶t̶o̶o̶l̶s̶ ̶l̶i̶k̶e̶ ̶l̶i̶b̶r̶o̶s̶a̶.̶ ̶

O̶u̶t̶p̶u̶t̶ ̶A̶u̶d̶i̶o̶:̶ ̶T̶h̶e̶ ̶p̶e̶r̶t̶u̶r̶b̶e̶d̶ ̶a̶u̶d̶i̶o̶ ̶r̶e̶t̶a̶i̶n̶s̶ ̶p̶e̶r̶c̶e̶p̶t̶u̶a̶l̶ ̶q̶u̶a̶l̶i̶t̶y̶ ̶a̶n̶d̶ ̶i̶s̶ ̶s̶a̶f̶e̶ ̶f̶o̶r̶ ̶d̶i̶s̶t̶r̶i̶b̶u̶t̶i̶o̶n̶.̶

So this was easily fixable by splitting stereo into two mono signals and then merging back after noise injection which i implemented to this fork.

Effectiveness: This script is a proof of concept and is intended for experimentation. Further enhancements are required for real-world robustness.



---

### Future Work

S̶u̶p̶p̶o̶r̶t̶ ̶f̶o̶r̶ ̶m̶u̶l̶t̶i̶-̶c̶h̶a̶n̶n̶e̶l̶ ̶(̶s̶t̶e̶r̶e̶o̶)̶ ̶W̶A̶V̶ ̶f̶i̶l̶e̶s̶.̶ 
(oh didn't realize this was planned as a future update hope i could contribute thanks for building the actual whole thing <3 great idea thx)

Integration with more advanced psychoacoustic models.

Evaluation against specific generative AI models.



---

### License

This project is licensed under the MIT License. See the LICENSE file for details.


---

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.






# HarmonyDagger Feature Roadmap

This document outlines the feature roadmap for extending HarmonyDagger's psychoacoustic protection capabilities toward a more complete implementation inspired by the HarmonyCloak research paper. Each feature is scoped as a new Git branch with specific implementation goals and technical details.

## Executive Summary

HarmonyDagger currently implements basic psychoacoustic noise injection for audio files using STFT, targeting human hearing threshold masking. This roadmap expands the implementation with advanced features that don't require GPU-intensive training or inference, making the tool accessible to average musicians on consumer-grade laptops.

## Current Features

* Adds psychoacoustic noise to WAV files using STFT
* CLI interface for processing files
* Targets human hearing threshold masking
* Basic adaptive scaling based on signal strength
* Multi-channel (stereo) processing support
* Batch processing capability for directories
* Basic visualization of spectrograms and differences

## Feature Branches

### 1. `branch:windowed-frequency-alignment`

**Purpose:** Implement a windowed processing strategy to dynamically align injected noise with dominant frequencies in short time frames.

**Technical Details:**
- Refine STFT windowing to use shorter time frames (10ms) for more precise frequency analysis
- Implement peak detection to identify multiple dominant frequencies in each frame
- Create a moving average filter to smooth frequency transitions between adjacent frames
- Apply weighted distribution of noise based on detected peaks
- Ensure smooth transitions between frames to prevent audible artifacts

**Key Files to Modify:**
- `dagger.py`: Update `generate_psychoacoustic_noise()` function 
- Add new helper functions for peak detection and moving average calculation

**Example Implementation:**
```python
def detect_multiple_peaks(magnitude_frame, n_peaks=3):
    """Detect multiple dominant frequencies in an STFT frame."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(magnitude_frame, height=0.1*np.max(magnitude_frame), distance=3)
    # Sort by magnitude and take top n_peaks
    peak_values = [(i, magnitude_frame[i]) for i in peaks]
    peak_values.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in peak_values[:n_peaks]]

def calculate_moving_average(frames, window_size=3):
    """Apply moving average to frequency peaks across time frames."""
    import numpy as np
    # Using sliding window approach
    result = np.zeros_like(frames)
    for i in range(len(frames)):
        start = max(0, i - window_size // 2)
        end = min(len(frames), i + window_size // 2 + 1)
        result[i] = np.mean(frames[start:end], axis=0)
    return result
```

**Benefits:**
- More accurate targeting of noise to mask specific frequencies
- Better preservation of audio quality with dynamic alignment
- Reduced perceptibility of protection noise

---

### 2. `branch:multi-track-support`

**Purpose:** Allow processing of multichannel audio files as independent tracks with channel-specific noise generation.

**Technical Details:**
- Enhance channel-specific processing for true independent track handling
- Implement channel-specific masking threshold calculations
- Add cross-channel analysis to preserve stereo imaging and phase relationships
- Support more than 2 channels for surround sound formats (e.g., 5.1)
- Add channel-specific noise scaling parameters based on spectral content

**Key Files to Modify:**
- `dagger.py`: Enhance `apply_noise_multichannel()` function
- Add new functions for channel correlation analysis

**Example Implementation:**
```python
def analyze_channel_correlation(audio_channels):
    """Analyze correlation between audio channels to preserve stereo imaging."""
    import numpy as np
    n_channels = len(audio_channels)
    correlation_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # Calculate normalized correlation coefficient
                corr = np.corrcoef(audio_channels[i], audio_channels[j])[0, 1]
                correlation_matrix[i, j] = corr
    
    return correlation_matrix

def apply_channel_specific_noise(audio, correlation_matrix, sr, params):
    """Apply noise with consideration for inter-channel relationships."""
    n_channels = audio.shape[0]
    perturbed = np.zeros_like(audio)
    
    # Generate noise for each channel considering correlations
    for ch in range(n_channels):
        # Adjust noise parameters based on channel characteristics
        ch_noise_scale = params['noise_scale'] * (0.8 + 0.4 * np.random.random())
        
        # Generate base noise for this channel
        noise = generate_psychoacoustic_noise(
            audio[ch], sr, params['window_size'], params['hop_size'], 
            ch_noise_scale, params['adaptive_scaling']
        )
        
        # Apply correlation-aware adjustments
        for other_ch in range(n_channels):
            if other_ch != ch and correlation_matrix[ch, other_ch] > 0.7:
                # Channels are highly correlated, make noise more similar
                # to preserve stereo imaging
                pass
        
        perturbed[ch] = apply_noise_to_audio(audio[ch], noise)
    
    return perturbed
```

**Benefits:**
- Improved handling of stereo and multichannel audio
- Better preservation of spatial imaging in mixes
- More natural sound quality in protected audio

---

### 3. `branch:adaptive-temporal-masking`

**Purpose:** Add a temporal masking model that leverages pre- and post-masking phenomena to reduce the need for simultaneous frequency masking.

**Technical Details:**
- Implement temporal masking model based on psychoacoustic research
- Add detection of transients (sudden loud sounds) using envelope analysis
- Create pre-masking window (5-10ms before transients) where less noise is needed
- Implement post-masking window (50-200ms after transients) with gradual decay
- Adjust noise injection to account for temporal masking effects

**Key Files to Modify:**
- `dagger.py`: Add new temporal masking functions
- Modify `generate_psychoacoustic_noise()` to incorporate temporal masking

**Example Implementation:**
```python
def detect_transients(audio, sr, threshold=0.7, window_size=512):
    """Detect transients in audio signal using envelope analysis."""
    import numpy as np
    from scipy.signal import hilbert
    
    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)
    
    # Normalize envelope
    envelope = envelope / np.max(envelope)
    
    # Calculate derivative of envelope to find rapid changes
    env_diff = np.diff(envelope)
    env_diff = np.append(env_diff, 0)  # Append zero to maintain length
    
    # Find positions where derivative exceeds threshold
    transient_positions = np.where(env_diff > threshold)[0]
    
    # Convert sample positions to time (seconds)
    transient_times = transient_positions / sr
    
    return transient_times, transient_positions

def calculate_temporal_masking_window(transient_positions, audio_length, sr, 
                                     pre_mask_ms=10, post_mask_ms=150):
    """Calculate temporal masking windows around detected transients."""
    import numpy as np
    
    # Convert ms to samples
    pre_mask_samples = int(pre_mask_ms * sr / 1000)
    post_mask_samples = int(post_mask_ms * sr / 1000)
    
    # Initialize masking window with ones (no masking effect)
    masking_window = np.ones(audio_length)
    
    # Apply pre and post-masking for each transient
    for position in transient_positions:
        # Pre-masking (ramp up to transient)
        pre_start = max(0, position - pre_mask_samples)
        if pre_start < position:
            # Create linear ramp from 0.2 to 1.0
            pre_ramp = np.linspace(0.2, 1.0, position - pre_start)
            masking_window[pre_start:position] = np.minimum(
                masking_window[pre_start:position], pre_ramp)
        
        # Post-masking (decay after transient)
        post_end = min(audio_length, position + post_mask_samples)
        if position < post_end:
            # Create exponential decay from 1.0 to 0.1
            post_ramp = np.linspace(1.0, 0.1, post_end - position) ** 2
            masking_window[position:post_end] = np.minimum(
                masking_window[position:post_end], post_ramp)
    
    return masking_window
```

**Benefits:**
- Reduced overall noise level needed for effective protection
- More natural sounding results by leveraging human auditory masking
- Better protection during transient-heavy content (percussion, etc.)

---

### 4. `branch:mp3-compression-resilience-test`

**Purpose:** Automatically re-encode WAV to MP3 after noise injection and analyze to assess noise retention.

**Technical Details:**
- Add MP3 encoding and decoding capability using `pydub` or similar library
- Implement spectral analysis to compare noise before and after compression
- Create metrics to quantify noise survival rate across frequency bands
- Add automatic parameter adjustment if noise is stripped by compression
- Implement testing across multiple compression quality levels (128k, 192k, 320k)

**Key Files to Modify:**
- `dagger.py`: Add new compression testing functions
- Add dependency for MP3 encoding/decoding (pydub)

**Example Implementation:**
```python
def encode_decode_mp3(audio_data, sr, bitrate="192k"):
    """Encode audio to MP3 and decode back to WAV to test compression effects."""
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    import io
    import numpy as np
    
    # Convert numpy array to AudioSegment
    audio_float = np.copy(audio_data)
    audio_pcm = (audio_float * 32767).astype(np.int16)
    
    if audio_pcm.ndim == 1:  # Mono
        channels = 1
    else:
        channels = audio_pcm.shape[0]
        audio_pcm = audio_pcm.T  # Convert to (samples, channels)
    
    segment = AudioSegment(
        audio_pcm.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=channels
    )
    
    # Export to MP3 in-memory
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3", bitrate=bitrate)
    buffer.seek(0)
    
    # Import back from MP3
    compressed = AudioSegment.from_file(buffer, format="mp3")
    
    # Convert to numpy array
    compressed_pcm = np.array(compressed.get_array_of_samples())
    
    if compressed.channels == 2:
        compressed_pcm = compressed_pcm.reshape((-1, 2)).T
    
    # Convert back to float in range [-1, 1]
    compressed_float = compressed_pcm.astype(np.float32) / 32767
    
    return compressed_float

def calculate_noise_retention(original, protected, compressed_protected):
    """Calculate what percentage of injected noise survives compression."""
    import numpy as np
    from scipy.signal import stft
    
    # Calculate noise before compression
    original_noise = protected - original
    
    # Calculate noise after compression
    compressed_noise = compressed_protected - original
    
    # Calculate STFTs
    f_original, t_original, Zxx_original = stft(original_noise[0] if original_noise.ndim > 1 else original_noise)
    f_compressed, t_compressed, Zxx_compressed = stft(compressed_noise[0] if compressed_noise.ndim > 1 else compressed_noise)
    
    # Calculate magnitudes
    mag_original = np.abs(Zxx_original)
    mag_compressed = np.abs(Zxx_compressed)
    
    # Calculate retention ratio across frequency bands
    retention = np.zeros(len(f_original))
    for i in range(len(f_original)):
        if np.sum(mag_original[i, :]) > 0:
            retention[i] = np.sum(mag_compressed[i, :]) / np.sum(mag_original[i, :])
        else:
            retention[i] = 1.0
    
    # Calculate overall retention
    overall_retention = np.mean(retention)
    
    return overall_retention, retention
```

**Benefits:**
- Ensures protection survives common distribution formats
- Provides metrics for effectiveness in real-world scenarios
- Allows for optimizing parameters specifically for compression resistance

---

### 5. `branch:perceptual-quality-check`

**Purpose:** Integrate perceptual audio metrics to score audio quality before and after protection.

**Technical Details:**
- Implement log-spectral distance calculation for objective comparison
- Add PEAQ-inspired metrics (Perceptual Evaluation of Audio Quality) using lightweight algorithms
- Create simple quality scoring system (0-100%) based on psychoacoustic principles
- Add automatic parameter adjustment based on quality metrics
- Implement A/B comparison visualization and reporting

**Key Files to Modify:**
- `dagger.py`: Add new perceptual quality functions
- Enhance visualization code to include quality metrics

**Example Implementation:**
```python
def calculate_log_spectral_distance(original, processed, sr, n_fft=2048):
    """Calculate log-spectral distance between original and processed audio."""
    import numpy as np
    from scipy.signal import stft
    
    # Calculate STFTs
    f, t, Zxx_original = stft(original, fs=sr, nperseg=n_fft)
    _, _, Zxx_processed = stft(processed, fs=sr, nperseg=n_fft)
    
    # Calculate power spectra
    psd_original = np.abs(Zxx_original)**2
    psd_processed = np.abs(Zxx_processed)**2
    
    # Avoid log(0) by adding a small constant
    psd_original = np.maximum(psd_original, 1e-10)
    psd_processed = np.maximum(psd_processed, 1e-10)
    
    # Calculate log-spectral distance
    log_diff = 10 * np.log10(psd_processed / psd_original)
    
    # Calculate mean and standard deviation of log differences
    mean_lsd = np.mean(np.abs(log_diff))
    std_lsd = np.std(log_diff)
    
    return mean_lsd, std_lsd

def perceptual_quality_score(original, processed, sr):
    """Calculate perceptual quality score based on psychoacoustic model."""
    import numpy as np
    from scipy.stats import pearsonr
    
    # Calculate basic metrics
    mean_lsd, std_lsd = calculate_log_spectral_distance(original, processed, sr)
    
    # Calculate waveform correlation
    correlation, _ = pearsonr(original, processed)
    
    # Calculate envelope difference
    from scipy.signal import hilbert
    env_original = np.abs(hilbert(original))
    env_processed = np.abs(hilbert(processed))
    env_correlation, _ = pearsonr(env_original, env_processed)
    
    # Simplified PEAQ-inspired score (0-100 scale)
    # Lower LSD and higher correlations are better
    lsd_score = 100 * np.exp(-0.3 * mean_lsd)
    corr_score = 100 * (correlation + 1) / 2  # Map from [-1,1] to [0,100]
    env_score = 100 * (env_correlation + 1) / 2
    
    # Weighted combination
    final_score = 0.4 * lsd_score + 0.3 * corr_score + 0.3 * env_score
    
    # Create detailed metrics dictionary
    metrics = {
        'lsd_mean': mean_lsd,
        'lsd_std': std_lsd,
        'waveform_correlation': correlation,
        'envelope_correlation': env_correlation,
        'lsd_score': lsd_score,
        'correlation_score': corr_score,
        'envelope_score': env_score,
        'final_score': final_score
    }
    
    return final_score, metrics
```

**Benefits:**
- Objective metrics for protection quality
- Balance between protection effectiveness and audio quality
- Tools for fine-tuning parameters to specific content

---

### 6. `branch:genre-aware-preset-generator`

**Purpose:** Provide preset profiles for common music genres with tuned masking parameters.

**Technical Details:**
- Create spectrum analysis functions to detect genre characteristics
- Implement presets for common genres (rock, jazz, electronic, classical, etc.)
- Add parameter tuning specific to frequency distributions of different genres
- Create user-friendly preset selection interface
- Add capability to blend presets for hybrid genres

**Key Files to Modify:**
- `dagger.py`: Add genre detection and preset functions
- Add preset definitions and genre-specific processing

**Example Implementation:**
```python
def analyze_genre_characteristics(audio, sr):
    """Analyze spectral characteristics to suggest genre classification."""
    import numpy as np
    import librosa
    
    # Extract features
    # Spectral centroid - brightness
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    
    # Spectral contrast - difference between peaks and valleys
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).mean(axis=1)
    
    # RMS energy - overall loudness
    rms = librosa.feature.rms(y=audio).mean()
    
    # Zero crossing rate - noisiness/distortion
    zcr = librosa.feature.zero_crossing_rate(y=audio).mean()
    
    # Tempo - beats per minute
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    
    # Onset strength - rhythm articulation
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_strength = np.mean(onset_env)
    
    # Create feature vector
    features = {
        'centroid': float(centroid),
        'contrast_low': float(contrast[0]),
        'contrast_mid': float(np.mean(contrast[1:4])),
        'contrast_high': float(np.mean(contrast[4:])),
        'rms': float(rms),
        'zcr': float(zcr),
        'tempo': float(tempo),
        'onset_strength': float(onset_strength)
    }
    
    # Simple genre classification based on features
    genre_scores = {}
    
    # Rock: high contrast, medium centroid, high RMS
    rock_score = (features['contrast_mid'] * 0.4 + 
                 (features['rms'] / 0.1) * 0.4 + 
                 (1 - abs(features['centroid'] - 2000)/2000) * 0.2)
    genre_scores['rock'] = min(1.0, max(0.0, rock_score))
    
    # Classical: low ZCR, low RMS, low contrast
    classical_score = ((1 - features['zcr'] / 0.1) * 0.4 + 
                      (1 - features['rms'] / 0.1) * 0.4 +
                      (1 - features['contrast_high']) * 0.2)
    genre_scores['classical'] = min(1.0, max(0.0, classical_score))
    
    # Electronic: high centroid, high contrast_high, consistent rhythm
    electronic_score = (features['centroid'] / 3000) * 0.3 + \
                      features['contrast_high'] * 0.3 + \
                      features['onset_strength'] * 0.4
    genre_scores['electronic'] = min(1.0, max(0.0, electronic_score))
    
    # Jazz: medium centroid, low ZCR, medium RMS
    jazz_score = ((1 - abs(features['centroid'] - 1500)/1500) * 0.4 + 
                 (1 - features['zcr'] / 0.1) * 0.3 + 
                 (1 - abs(features['rms'] - 0.05)/0.05) * 0.3)
    genre_scores['jazz'] = min(1.0, max(0.0, jazz_score))
    
    return genre_scores, features

def genre_specific_parameters(genre):
    """Return optimal protection parameters for specific music genre."""
    presets = {
        'rock': {
            'noise_scale': 0.015,
            'window_size': 1024,
            'hop_size': 512,
            'adaptive_scaling': True,
            'frequency_emphasis': 'mid',  # Emphasize mid-range frequencies
            'temporal_threshold': 0.6,    # For transient detection
        },
        'jazz': {
            'noise_scale': 0.008,
            'window_size': 2048,         # Larger window for better frequency resolution
            'hop_size': 1024,
            'adaptive_scaling': True,
            'frequency_emphasis': 'high', # Emphasize high frequencies (cymbals, etc)
            'temporal_threshold': 0.5,
        },
        'classical': {
            'noise_scale': 0.005,        # Lower scale for subtle protection
            'window_size': 4096,         # Largest window for precise frequency analysis
            'hop_size': 2048,
            'adaptive_scaling': True,
            'frequency_emphasis': 'balanced',
            'temporal_threshold': 0.4,    # Lower threshold for subtle transients
        },
        'electronic': {
            'noise_scale': 0.012,
            'window_size': 1024,
            'hop_size': 512,
            'adaptive_scaling': True,
            'frequency_emphasis': 'low',  # Emphasize low (bass) frequencies
            'temporal_threshold': 0.7,    # Higher threshold for pronounced beats
        },
        'ambient': {
            'noise_scale': 0.007,
            'window_size': 4096,
            'hop_size': 2048,
            'adaptive_scaling': True,
            'frequency_emphasis': 'wide', # Wide spectrum approach
            'temporal_threshold': 0.3,    # Low threshold for subtle changes
        },
    }
    
    return presets.get(genre, presets['rock'])  # Default to rock preset
```

**Benefits:**
- Better quality results with content-aware processing
- Simplified workflow with presets for common use cases
- More effective protection tuned to genre characteristics

---

### 7. `branch:batch-folder-cli`

**Purpose:** Extend CLI to recursively process folders of WAV files with enhanced reporting.

**Technical Details:**
- Enhance CLI to support recursive directory traversal
- Add inclusion/exclusion patterns for selective processing
- Implement detailed report generation with protection metrics
- Add progress tracking for large batches
- Create summary statistics with visualization

**Key Files to Modify:**
- `dagger.py`: Enhance command-line argument parsing
- Add reporting and statistics functions

**Example Implementation:**
```python
def recursive_file_processor(root_dir, pattern="*.wav", exclusion_pattern=None):
    """Recursively find audio files in directory structure."""
    import os
    import fnmatch
    
    matches = []
    
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            # Skip excluded files
            if exclusion_pattern and fnmatch.fnmatch(filename, exclusion_pattern):
                continue
            matches.append(os.path.join(root, filename))
    
    return matches

def generate_batch_report(results, output_path):
    """Generate detailed report of batch processing results."""
    import json
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create summary stats
    num_files = len(results)
    total_duration = sum(r.get('duration', 0) for r in results)
    avg_quality_score = np.mean([r.get('quality_score', 0) for r in results])
    avg_noise_retention = np.mean([r.get('noise_retention', 0) for r in results])
    
    # Create report dictionary
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'files_processed': num_files,
            'total_duration_seconds': total_duration,
            'average_quality_score': avg_quality_score,
            'average_noise_retention': avg_noise_retention,
        },
        'file_details': results
    }
    
    # Save JSON report
    json_path = os.path.join(output_path, 'protection_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary visualization
    plt.figure(figsize=(10, 8))
    
    # Plot quality scores
    plt.subplot(2, 1, 1)
    quality_scores = [r.get('quality_score', 0) for r in results]
    plt.bar(range(len(quality_scores)), quality_scores)
    plt.axhline(y=avg_quality_score, color='r', linestyle='-', label=f'Avg: {avg_quality_score:.2f}')
    plt.ylabel('Quality Score')
    plt.xlabel('File Index')
    plt.title('Audio Quality Scores')
    plt.legend()
    
    # Plot noise retention if available
    if any('noise_retention' in r for r in results):
        plt.subplot(2, 1, 2)
        retention_scores = [r.get('noise_retention', 0) for r in results]
        plt.bar(range(len(retention_scores)), retention_scores)
        plt.axhline(y=avg_noise_retention, color='r', linestyle='-', 
                   label=f'Avg: {avg_noise_retention:.2f}')
        plt.ylabel('Noise Retention')
        plt.xlabel('File Index')
        plt.title('MP3 Compression Noise Retention')
        plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_path, 'protection_summary.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    
    return json_path, plot_path
```

**Benefits:**
- Streamlined workflow for batch processing
- Detailed analytics across processed files
- Better visibility into protection effectiveness

---

### 8. `branch:spectrogram-visualizer`

**Purpose:** Add a matplotlib-based spectrogram comparison tool with enhanced visualization features.

**Technical Details:**
- Create enhanced spectrogram visualizations with overlay capability
- Add zoom and region selection for detailed analysis
- Implement difference highlighting to show where noise is concentrated
- Add perceptual masking curve visualization
- Create interactive export options for reporting

**Key Files to Modify:**
- `dagger.py`: Enhance visualization functions
- Add new visualization module for advanced features

**Example Implementation:**
```python
def create_overlay_spectrogram(original, perturbed, noise, sr, output_path=None):
    """Create multi-layer spectrogram showing original, noise, and perturbed audio."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define custom colormaps
    orig_cmap = plt.cm.Blues
    noise_cmap = LinearSegmentedColormap.from_list('noise_cmap', 
                                                  [(0, 'white'), (1, 'red')], N=100)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Calculate spectrograms
    plt.subplot(3, 1, 1)
    plt.title("Original Audio")
    plt.specgram(original, Fs=sr, NFFT=1024, noverlap=512, cmap=orig_cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Frequency [Hz]")
    
    plt.subplot(3, 1, 2)
    plt.title("Noise Layer (Exaggerated for Visibility)")
    # Amplify noise for better visibility
    visible_noise = noise * 5
    plt.specgram(visible_noise, Fs=sr, NFFT=1024, noverlap=512, cmap=noise_cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Frequency [Hz]")
    
    plt.subplot(3, 1, 3)
    plt.title("Protected Audio (Original + Noise)")
    plt.specgram(perturbed, Fs=sr, NFFT=1024, noverlap=512, cmap=orig_cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None

def visualize_masking_thresholds(audio, sr, window_size=1024, output_path=None):
    """Visualize calculated masking thresholds over spectrogram."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    
    # Perform STFT
    frequencies, times, Zxx = stft(audio, fs=sr, nperseg=window_size, 
                                   noverlap=window_size//2)
    magnitude = np.abs(Zxx)
    
    # Calculate average magnitude across time
    avg_magnitude = np.mean(magnitude, axis=1)
    
    # Convert to dB
    magnitude_db = 20 * np.log10(np.maximum(avg_magnitude, 1e-10))
    
    # Calculate hearing threshold for each frequency
    hearing_threshold_curve = np.array([hearing_threshold(f) for f in frequencies])
    
    # Calculate simplified masking threshold (just for visualization)
    # In reality this is calculated per frame in the STFT
    from scipy.ndimage import gaussian_filter1d
    smoothed_magnitude = gaussian_filter1d(magnitude_db, sigma=2)
    masking_threshold = smoothed_magnitude - 20  # Simplified offset
    
    # Ensure masking threshold is never below hearing threshold
    masking_threshold = np.maximum(masking_threshold, hearing_threshold_curve)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot spectrogram
    plt.subplot(2, 1, 1)
    plt.specgram(audio, Fs=sr, NFFT=window_size, noverlap=window_size//2, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Audio Spectrogram")
    plt.ylabel("Frequency [Hz]")
    
    # Plot threshold curves
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, magnitude_db, label="Signal", color='blue')
    plt.plot(frequencies, hearing_threshold_curve, label="Hearing Threshold", color='red')
    plt.plot(frequencies, masking_threshold, label="Masking Threshold", color='green')
    plt.fill_between(frequencies, hearing_threshold_curve, masking_threshold, 
                     color='yellow', alpha=0.3, label="Noise Insertion Zone")
    plt.title("Psychoacoustic Masking Model")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None
```

**Benefits:**
- Better understanding of protection mechanism through visualization
- Easier debugging and fine-tuning of parameters
- Educational tool for demonstrating the psychoacoustic principles
- Improved reporting for professional audio engineers

---

### 9. `branch:audacity-plugin-export`

**Purpose:** Package HarmonyDagger as an Audacity plugin using Nyquist or scripting interface for broader adoption.

**Technical Details:**
- Research Audacity plugin architecture and Nyquist scripting requirements
- Create wrapper functions to interface with Audacity's plugin system
- Package core algorithm for Nyquist script integration
- Implement parameter controls compatible with Audacity's interface
- Create installation package with documentation

**Key Files to Modify:**
- Create new directory `audacity_plugin/` for plugin files
- Create Nyquist script template and wrapper functions
- Add installation and usage documentation

**Example Implementation:**
```python
def export_to_nyquist_script(params, output_path):
    """Export core algorithm as Audacity Nyquist script."""
    # Nyquist script template with embedded parameters
    nyquist_script = f"""
;nyquist plugin
;version 4
;type process
;name "HarmonyDagger Audio Protection"
;author "HarmonyDagger Team"
;copyright "MIT License"

;control noise-scale "Noise Scale" real "dB" 0 -20 0
;control adaptive "Adaptive Scaling" int "Enabled" 1 0 1
;control window "Window Size" int "" 1024 512 4096
;control emphasis "Frequency Emphasis" choice "Balanced,Low,Mid,High" 0

(defun bark-scale (freq)
  (+ (* 13 (atan (* 0.00076 freq)))
     (* 3.5 (atan (expt (/ freq 7500.0) 2)))))

(defun critical-band-width (center-freq)
  (+ 25 (* 75 (expt (+ 1 (* 1.4 (expt (/ center-freq 1000.0) 2))) 0.69))))

(defun hearing-threshold (freq)
  (let* ((f (/ freq 1000.0))
         (threshold (- (* 3.64 (expt f -0.8))
                       (* 6.5 (exp (* -0.6 (expt (- f 3.3) 2)))))))
    threshold))

;; Main protection function
(defun protect-audio (sound)
  (let* ((noise-amt (db-to-linear (+ {params.get('noise_scale', 0.01)} (getf 'noise-scale))))
         (wnd-size (getf 'window))
         (adpt (= (getf 'adaptive) 1))
         (emph-choice (getf 'emphasis)))
    ;; Implementation of psychoacoustic protection...
    ;; (This is simplified - actual implementation would need more Nyquist-specific code)
    (sum sound (mult noise-amt (noise)))))

;; Process the audio
(protect-audio *track*)
    """
    
    # Save script to file
    with open(output_path, 'w') as f:
        f.write(nyquist_script)
    
    print(f"Nyquist plugin script saved to: {output_path}")
    return output_path

def create_audacity_plugin_package(output_dir):
    """Create complete Audacity plugin package with all required files."""
    import os
    import shutil
    
    # Create plugin directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the Nyquist script
    default_params = {
        'noise_scale': 0.01,
        'window_size': 1024,
        'hop_size': 512,
        'adaptive_scaling': True
    }
    
    script_path = os.path.join(output_dir, "HarmonyDagger.ny")
    export_to_nyquist_script(default_params, script_path)
    
    # Create README file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("""# HarmonyDagger Audacity Plugin

This plugin adds psychoacoustic protection to your audio files within Audacity.

## Installation

1. Copy the HarmonyDagger.ny file to Audacity's Plug-Ins folder:
   - Windows: %APPDATA%\\Audacity\\Plug-Ins
   - macOS: ~/Library/Application Support/audacity/Plug-Ins
   - Linux: ~/.audacity-files/Plug-Ins

2. Restart Audacity

3. The plugin will appear under Effect menu > HarmonyDagger Audio Protection

## Usage

1. Select the audio region you want to protect
2. Open the plugin from the Effect menu
3. Adjust parameters as needed
4. Click "Apply"

## Parameters

- Noise Scale: Controls the strength of the protection (lower values = more subtle)
- Adaptive Scaling: Enables dynamic adjustment based on audio content
- Window Size: Controls frequency resolution (larger = more precise frequency analysis)
- Frequency Emphasis: Choose which frequency range to emphasize in the protection

## More Information

Visit https://github.com/yourusername/harmonydagger for the full version with more features.
""")
    
    # Create a simple installation script
    install_script_path = os.path.join(output_dir, "install.py")
    with open(install_script_path, 'w') as f:
        f.write("""#!/usr/bin/env python3
import os
import shutil
import platform
import sys

def install_plugin():
    # Determine Audacity plugin folder based on operating system
    plugin_folder = None
    system = platform.system()
    
    if system == "Windows":
        plugin_folder = os.path.join(os.getenv('APPDATA'), 'Audacity', 'Plug-Ins')
    elif system == "Darwin":  # macOS
        plugin_folder = os.path.expanduser('~/Library/Application Support/audacity/Plug-Ins')
    elif system == "Linux":
        plugin_folder = os.path.expanduser('~/.audacity-files/Plug-Ins')
    
    if not plugin_folder:
        print(f"Unsupported platform: {system}")
        return False
    
    # Create the plugin folder if it doesn't exist
    os.makedirs(plugin_folder, exist_ok=True)
    
    # Copy the plugin file
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HarmonyDagger.ny')
    destination = os.path.join(plugin_folder, 'HarmonyDagger.ny')
    
    try:
        shutil.copy2(source, destination)
        print(f"Successfully installed HarmonyDagger plugin to: {destination}")
        print("Please restart Audacity to use the plugin.")
        return True
    except Exception as e:
        print(f"Error installing plugin: {str(e)}")
        return False

if __name__ == "__main__":
    install_plugin()
""")
    
    # Make the installation script executable
    os.chmod(install_script_path, 0o755)
    
    return output_dir
```

**Benefits:**
- Broader adoption by integration with popular audio editing software
- Simplified workflow for non-technical users
- Expanded user base beyond command-line users
- Easy installation and usage for musicians

"""
File operations and batch processing functions for HarmonyDagger.
"""
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import librosa
import soundfile as sf
from pydub import AudioSegment

from .common import (
    DEFAULT_HOP_SIZE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WINDOW_SIZE,
)
from .core import apply_noise_multichannel

# Set up module logger
logger = logging.getLogger(__name__)


def process_audio_file(
    file_path: str,
    output_path: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    force_mono: bool = False,
    visualize: bool = False,
    visualize_diff: bool = False,
    visualization_path: Optional[str] = None
) -> Tuple[bool, str, float]:
    """
    Process a single audio file with HarmonyDagger.
    
    Args:
        file_path: Path to input audio file
        output_path: Path to save processed audio. If None, creates a path based on input file.
        window_size: STFT window size
        hop_size: STFT hop size
        noise_scale: Scale factor for noise (0.0 to 1.0)
        adaptive_scaling: Whether to use adaptive scaling based on signal strength
        force_mono: Convert stereo audio to mono before processing
        visualize: Whether to generate a spectrogram visualization
        visualize_diff: Whether to generate a difference visualization
        visualization_path: Directory to save visualizations, if None uses the output file directory
        
    Returns:
        Tuple of (success, output_file_path, processing_time_seconds)
    """
    start_time = time.time()
    
    try:
        # Generate output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_protected{ext}"
        
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=force_mono)
        
        # Process audio (handle both mono and multi-channel)
        try:
            if y.ndim > 1:  # Multi-channel
                y_processed = apply_noise_multichannel(
                    y, sr, window_size, hop_size, noise_scale, adaptive_scaling
                )
            else:  # Mono
                y_processed = apply_noise_multichannel(
                    y, sr, window_size, hop_size, noise_scale, adaptive_scaling
                )
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, str(e), time.time() - start_time
        
        # Determine format from output file extension
        _, ext = os.path.splitext(output_path)
        ext = ext.lower()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        
        # Save processed audio based on format
        try:
            if ext == '.mp3':
                # MP3 format requires special handling with pydub
                try:
                    # Check if ffmpeg or avconv is available
                    from pydub.utils import which
                    if which("ffmpeg") is None and which("avconv") is None:
                        # Fall back to WAV if ffmpeg/avconv is not available
                        logger.warning("ffmpeg/avconv not found. Falling back to WAV format.")
                        output_path = os.path.splitext(output_path)[0] + ".wav"
                        sf.write(output_path, y_processed, sr)
                    else:
                        # Convert to MP3 using pydub
                        temp_wav = tempfile.mktemp(suffix='.wav')
                        sf.write(temp_wav, y_processed, sr)
                        
                        # Convert WAV to MP3 using pydub
                        audio = AudioSegment.from_wav(temp_wav)
                        audio.export(output_path, format="mp3", bitrate="192k")
                        
                        # Clean up temporary file
                        os.remove(temp_wav)
                except Exception as e:
                    logger.error(f"Error converting to MP3: {str(e)}. Falling back to WAV format.")
                    # Fall back to WAV format
                    output_path = os.path.splitext(output_path)[0] + ".wav"
                    sf.write(output_path, y_processed, sr)
            elif ext in ['.flac', '.ogg', '.wav']:
                try:
                    # Formats supported directly by soundfile
                    if ext == '.wav':
                        # Use our custom WAV utility for robust WAV file saving
                        from .wav_utils import save_wav_file
                        success = save_wav_file(output_path, y_processed, sr)
                        if success:
                            logger.debug(f"Saved audio as WAV using wav_utils: {output_path}")
                        else:
                            # If that still fails, try soundfile as a fallback
                            logger.warning("Fallback to soundfile for WAV saving")
                            sf.write(output_path, y_processed, sr, format='WAV')
                    elif ext == '.flac':
                        sf.write(output_path, y_processed, sr, format='FLAC')
                        logger.debug(f"Saved audio as FLAC: {output_path}")
                    elif ext == '.ogg':
                        sf.write(output_path, y_processed, sr, format='OGG')
                        logger.debug(f"Saved audio as OGG: {output_path}")
                except Exception as format_error:
                    logger.error(f"Error saving in {ext} format: {str(format_error)}. Falling back to WAV with .wav extension.")
                    output_path = os.path.splitext(output_path)[0] + ".wav"
                    
                    # Use our custom WAV utility as a fallback
                    from .wav_utils import save_wav_file
                    if not save_wav_file(output_path, y_processed, sr):
                        # If that still fails, try soundfile as a last resort
                        sf.write(output_path, y_processed, sr, format='WAV')
            else:
                # Default to WAV for unsupported formats
                logger.warning(f"Unsupported format: {ext}. Defaulting to WAV.")
                if not ext:
                    output_path = output_path + ".wav"
                else:
                    output_path = os.path.splitext(output_path)[0] + ".wav"
                sf.write(output_path, y_processed, sr, format='WAV')
                logger.debug(f"Saved audio in WAV format: {output_path}")
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, str(e), time.time() - start_time
        
        # Generate visualizations if requested
        if visualize or visualize_diff:
            try:
                from .visualization import create_audio_comparison
                
                # Use visualization_path if provided, otherwise use the output file directory
                vis_dir = visualization_path if visualization_path else os.path.dirname(output_path)
                os.makedirs(vis_dir, exist_ok=True)
                
                create_audio_comparison(
                    file_path,
                    output_path,
                    output_dir=vis_dir,
                    visualize_spectrogram=visualize,
                    visualize_diff=visualize_diff
                )
                logger.info(f"Generated visualizations in {vis_dir}")
            except Exception as vis_error:
                logger.error(f"Failed to generate visualizations: {str(vis_error)}")
        
        processing_time = time.time() - start_time
        return True, output_path, processing_time
    
    except Exception as e:
        return False, str(e), time.time() - start_time


def _process_file_for_batch(
    file_path: str,
    output_dir: Optional[str],
    window_size: int,
    hop_size: int,
    noise_scale: float,
    adaptive_scaling: bool,
    force_mono: bool,
    visualize: bool = False,
    visualize_diff: bool = False,
    visualization_path: Optional[str] = None
) -> Tuple[str, Tuple[bool, str, float]]:
    """
    Process a single audio file for batch processing.
    
    This is a helper function for parallel_batch_process.
    It's defined at the module level to ensure it can be pickled for parallel processing.
    
    Args:
        file_path: Path to input audio file
        output_dir: Directory to save processed files (if None, save alongside input file)
        window_size: STFT window size
        hop_size: STFT hop size
        noise_scale: Scale factor for noise
        adaptive_scaling: Whether to use adaptive scaling
        force_mono: Whether to convert to mono before processing
        visualize: Whether to generate a spectrogram visualization
        visualize_diff: Whether to generate a difference visualization
        visualization_path: Directory to save visualizations (defaults to output_dir)
        
    Returns:
        Tuple of (file_path, process_result) where process_result is (success, output_path/error, time)
    """
    if output_dir:
        filename = os.path.basename(file_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base}_protected{ext}")
    else:
        output_path = None
    
    # Use output_dir as visualization_path if not specified
    vis_path = visualization_path if visualization_path else output_dir
        
    return file_path, process_audio_file(
        file_path,
        output_path,
        window_size,
        hop_size,
        noise_scale,
        adaptive_scaling,
        force_mono,
        visualize,
        visualize_diff,
        vis_path
    )


def parallel_batch_process(
    file_paths: List[str],
    output_dir: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    force_mono: bool = False,
    max_workers: Optional[int] = None,
    visualize: bool = False,
    visualize_diff: bool = False,
    visualization_path: Optional[str] = None
) -> Dict[str, Dict[str, Union[bool, str, float]]]:
    """
    Process multiple audio files in parallel using a process pool.
    
    This function enables efficient parallel processing of multiple audio files
    using Python's ProcessPoolExecutor. Each file is processed independently
    in a separate process, allowing for significantly faster batch processing
    on multi-core CPUs.
    
    Args:
        file_paths: List of input audio file paths
        output_dir: Directory to save processed files. If None, saves alongside input files.
        window_size: STFT window size
        hop_size: STFT hop size
        noise_scale: Scale factor for noise (0.0 to 1.0)
        adaptive_scaling: Whether to use adaptive scaling based on signal strength
        force_mono: Convert stereo audio to mono before processing
        max_workers: Maximum number of worker processes. None = auto (uses CPU count)
        visualize: Whether to generate spectrogram visualizations
        visualize_diff: Whether to generate difference visualizations
        visualization_path: Directory to save visualizations. If None, uses output_dir
        
    Returns:
        Dictionary mapping input files to their processing results with format:
        {
            'file_path': {
                'success': bool,
                'output_path' or 'error': str,
                'processing_time': float
            }
        }
        
    Examples:
        >>> audio_files = recursive_find_audio_files('./audio_files')
        >>> results = parallel_batch_process(
        ...     audio_files,
        ...     output_dir='./protected_audio',
        ...     max_workers=4
        ... )
    """
    results = {}
    
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a partial function with fixed parameters
    process_func = partial(
        _process_file_for_batch,
        output_dir=output_dir,
        window_size=window_size,
        hop_size=hop_size,
        noise_scale=noise_scale,
        adaptive_scaling=adaptive_scaling,
        force_mono=force_mono,
        visualize=visualize,
        visualize_diff=visualize_diff,
        visualization_path=visualization_path
    )
    
    # Execute in parallel using a process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(process_func, path): path
            for path in file_paths
        }
        
        for future in as_completed(future_to_path):
            try:
                input_path, (success, output_or_error, proc_time) = future.result()
                results[input_path] = {
                    "success": success,
                    "output_path" if success else "error": output_or_error,
                    "processing_time": proc_time
                }
            except Exception as e:
                input_path = future_to_path[future]
                results[input_path] = {
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                }
    
    return results


def recursive_find_audio_files(
    directory: str,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively find audio files in a directory.
    
    Args:
        directory: Directory path to search
        extensions: List of audio file extensions to include
        
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
                
    return audio_files


def batch_process(
    input_dir: str,
    output_dir: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    adaptive_scaling: bool = True,
    force_mono: bool = False,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    file_extension: str = '.wav',
    visualize: bool = False,
    visualize_diff: bool = False,
    visualization_path: Optional[str] = None
) -> Dict[str, Dict[str, Union[bool, str, float]]]:
    """
    Process all audio files in a directory.
    
    This is a backward-compatible function that maintains the previous API.
    For new code, use parallel_batch_process instead.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed files
        window_size: STFT window size
        hop_size: STFT hop size
        noise_scale: Scale factor for noise (0.0 to 1.0)
        adaptive_scaling: Whether to use adaptive scaling based on signal strength
        force_mono: Convert stereo audio to mono before processing
        parallel: Whether to process files in parallel
        max_workers: Maximum number of worker processes (used only if parallel=True)
        file_extension: File extension to process (.wav, .mp3, etc.)
        visualize: Whether to generate spectrogram visualizations
        visualize_diff: Whether to generate difference visualizations
        visualization_path: Directory to save visualizations. If None, uses output_dir
        
    Returns:
        Dictionary mapping input files to their processing results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files to process
    file_paths = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(file_extension):
            file_paths.append(os.path.join(input_dir, file))
    
    # Process files in parallel or sequentially
    if parallel:
        return parallel_batch_process(
            file_paths,
            output_dir=output_dir,
            window_size=window_size,
            hop_size=hop_size,
            noise_scale=noise_scale,
            adaptive_scaling=adaptive_scaling,
            force_mono=force_mono,
            max_workers=max_workers,
            visualize=visualize,
            visualize_diff=visualize_diff,
            visualization_path=visualization_path
        )
    else:
        results = {}
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base}_protected{ext}")
            
            success, output_or_error, proc_time = process_audio_file(
                file_path,
                output_path,
                window_size=window_size,
                hop_size=hop_size,
                noise_scale=noise_scale,
                adaptive_scaling=adaptive_scaling,
                force_mono=force_mono,
                visualize=visualize,
                visualize_diff=visualize_diff,
                visualization_path=visualization_path
            )
            
            results[file_path] = {
                "success": success,
                "output_path" if success else "error": output_or_error,
                "processing_time": proc_time
            }
        
        return results

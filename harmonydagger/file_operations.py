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
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from .common import (
    DEFAULT_DRY_WET,
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
    visualization_path: Optional[str] = None,
    dry_wet: float = DEFAULT_DRY_WET,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
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
        dry_wet: Mix ratio (0.0 = original, 1.0 = fully protected)
        vocal_mode: Optimize protection for vocal frequencies (300Hz-3kHz)
        use_phase_perturbation: Add phase-based perturbation
        use_temporal_masking: Add temporal forward masking noise

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
            y_processed = apply_noise_multichannel(
                y, sr, window_size, hop_size, noise_scale, adaptive_scaling,
                dry_wet=dry_wet,
                vocal_mode=vocal_mode,
                use_phase_perturbation=use_phase_perturbation,
                use_temporal_masking=use_temporal_masking,
                use_ensemble=use_ensemble,
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
                        # Use our robust WAV saving function
                        from .wav_utils import save_wav_file
                        if not save_wav_file(output_path, y_processed, sr):
                            raise Exception("Failed to save WAV fallback file")
                    else:
                        # Use our robust WAV utility for the temporary file
                        from .wav_utils import save_wav_file
                        temp_wav_path = os.path.join(tempfile.gettempdir(), f"harmonydagger_temp_{int(time.time())}.wav")
                        logger.debug(f"Creating temporary WAV file: {temp_wav_path}")

                        if not save_wav_file(temp_wav_path, y_processed, sr):
                            # If our utility fails, try soundfile directly
                            logger.warning("Using soundfile for temporary WAV")
                            sf.write(temp_wav_path, y_processed, sr, format='WAV')

                        try:
                            logger.debug(f"Converting WAV to MP3: {output_path}")
                            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

                            with open(temp_wav_path, 'rb') as wav_file:
                                wav_data = wav_file.read()

                            raw_audio = AudioSegment(
                                data=wav_data,
                                sample_width=2,
                                frame_rate=sr,
                                channels=2 if y_processed.ndim > 1 else 1
                            )
                            raw_audio.export(output_path, format="mp3", bitrate="192k")
                            logger.debug(f"Successfully created MP3 file: {output_path}")
                        except Exception as mp3_error:
                            logger.error(f"Error in MP3 export: {str(mp3_error)}")
                            logger.warning("Trying alternative MP3 encoding method")
                            try:
                                import subprocess
                                cmd = [
                                    "ffmpeg", "-y", "-i", temp_wav_path,
                                    "-codec:a", "libmp3lame", "-qscale:a", "2",
                                    output_path
                                ]
                                subprocess.run(cmd, check=True, capture_output=True)
                                logger.debug(f"Created MP3 using ffmpeg command: {output_path}")
                            except Exception as ffmpeg_error:
                                logger.error(f"ffmpeg conversion failed: {str(ffmpeg_error)}")
                                raise

                        # Clean up temporary file
                        try:
                            os.remove(temp_wav_path)
                            logger.debug(f"Removed temporary WAV file: {temp_wav_path}")
                        except Exception as rm_error:
                            logger.warning(f"Failed to remove temporary file: {str(rm_error)}")
                except Exception as e:
                    logger.error(f"Error converting to MP3: {str(e)}. Falling back to WAV format.")
                    output_path = os.path.splitext(output_path)[0] + ".wav"
                    from .wav_utils import save_wav_file
                    if not save_wav_file(output_path, y_processed, sr):
                        logger.warning("Trying direct file write approach")
                        try:
                            audio_int16 = np.clip(y_processed * 32767, -32768, 32767).astype(np.int16)
                            with open(output_path, 'wb') as f:
                                import wave
                                wf = wave.open(f, 'wb')
                                wf.setnchannels(1 if y_processed.ndim == 1 else y_processed.shape[0])
                                wf.setsampwidth(2)
                                wf.setframerate(sr)
                                wf.writeframes(audio_int16.tobytes())
                                wf.close()
                            logger.debug(f"Wrote WAV file directly: {output_path}")
                        except Exception as direct_error:
                            logger.error(f"Direct WAV writing failed: {str(direct_error)}")
                            raise
            elif ext in ['.flac', '.ogg', '.wav']:
                try:
                    if ext == '.wav':
                        from .wav_utils import save_wav_file
                        success = save_wav_file(output_path, y_processed, sr)
                        if success:
                            logger.debug(f"Saved audio as WAV using wav_utils: {output_path}")
                        else:
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
                    from .wav_utils import save_wav_file
                    if not save_wav_file(output_path, y_processed, sr):
                        sf.write(output_path, y_processed, sr, format='WAV')
            else:
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
                vis_dir = visualization_path if visualization_path else os.path.dirname(output_path)
                if not vis_dir:
                    vis_dir = '.'
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
    visualization_path: Optional[str] = None,
    dry_wet: float = DEFAULT_DRY_WET,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
) -> Tuple[str, Tuple[bool, str, float]]:
    """
    Process a single audio file for batch processing.

    This is a helper function for parallel_batch_process.
    It's defined at the module level to ensure it can be pickled for parallel processing.
    """
    if output_dir:
        filename = os.path.basename(file_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base}_protected{ext}")
    else:
        output_path = None

    vis_path = visualization_path if visualization_path else output_dir

    return file_path, process_audio_file(
        file_path,
        output_path,
        window_size=window_size,
        hop_size=hop_size,
        noise_scale=noise_scale,
        adaptive_scaling=adaptive_scaling,
        force_mono=force_mono,
        visualize=visualize,
        visualize_diff=visualize_diff,
        visualization_path=vis_path,
        dry_wet=dry_wet,
        vocal_mode=vocal_mode,
        use_phase_perturbation=use_phase_perturbation,
        use_temporal_masking=use_temporal_masking,
        use_ensemble=use_ensemble,
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
    visualization_path: Optional[str] = None,
    dry_wet: float = DEFAULT_DRY_WET,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
) -> Dict[str, Dict[str, Union[bool, str, float]]]:
    """
    Process multiple audio files in parallel using a process pool.
    """
    results = {}

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        visualization_path=visualization_path,
        dry_wet=dry_wet,
        vocal_mode=vocal_mode,
        use_phase_perturbation=use_phase_perturbation,
        use_temporal_masking=use_temporal_masking,
        use_ensemble=use_ensemble,
    )

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
    visualization_path: Optional[str] = None,
    dry_wet: float = DEFAULT_DRY_WET,
    vocal_mode: bool = False,
    use_phase_perturbation: bool = False,
    use_temporal_masking: bool = False,
    use_ensemble: bool = False,
) -> Dict[str, Dict[str, Union[bool, str, float]]]:
    """
    Process all audio files in a directory.

    This is a backward-compatible function that maintains the previous API.
    For new code, use parallel_batch_process instead.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_paths = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(file_extension):
            file_paths.append(os.path.join(input_dir, file))

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
            visualization_path=visualization_path,
            dry_wet=dry_wet,
            vocal_mode=vocal_mode,
            use_phase_perturbation=use_phase_perturbation,
            use_temporal_masking=use_temporal_masking,
            use_ensemble=use_ensemble,
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
                visualization_path=visualization_path,
                dry_wet=dry_wet,
                vocal_mode=vocal_mode,
                use_phase_perturbation=use_phase_perturbation,
                use_temporal_masking=use_temporal_masking,
                use_ensemble=use_ensemble,
            )

            results[file_path] = {
                "success": success,
                "output_path" if success else "error": output_or_error,
                "processing_time": proc_time
            }

        return results

"""
File processing operations for HarmonyDagger.
"""
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .common import (
    DEFAULT_HOP_SIZE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WINDOW_SIZE,
    DIFFERENCE_SUFFIX,
    SPECTROGRAM_SUFFIX,
)
from .core import apply_noise_multichannel
from .visualization import visualize_difference, visualize_spectrograms


def process_audio_file(
    input_file: str,
    output_file: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    force_mono: bool = False,
    adaptive_scaling: bool = True,
    visualize: bool = False,
    visualize_diff: bool = False,
    visualization_path: Optional[str] = None
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Process a single audio file and add psychoacoustic noise.
    """
    start_time = time.time()

    try:
        logging.info(f"Loading input file: {input_file}")
        audio, sr = sf.read(input_file, always_2d=True)
        audio = audio.T  # (channels, samples)

        original_size = os.path.getsize(input_file)

        if force_mono and audio.shape[0] > 1:
            logging.info("Converting to mono...")
            audio = np.mean(audio, axis=0, keepdims=True)

        logging.info(f"Generating psychoacoustic noise with scale: {noise_scale}")
        additional_info = "with adaptive scaling" if adaptive_scaling else ""
        logging.info(f"Processing audio {additional_info}...")

        perturbed = apply_noise_multichannel(
            audio, sr, window_size, hop_size, noise_scale, adaptive_scaling
        )
        perturbed = perturbed.T  # (samples, channels)

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        logging.info(f"Saving perturbed audio to: {output_file}")
        sf.write(output_file, perturbed, sr)

        processing_time = time.time() - start_time
        new_size = os.path.getsize(output_file)
        file_size_ratio = new_size / original_size

        if visualize or visualize_diff:
            if visualization_path:
                os.makedirs(visualization_path, exist_ok=True)

            if visualize:
                spec_output = None
                if visualization_path:
                    base_name = os.path.splitext(os.path.basename(output_file))[0]
                    spec_output = os.path.join(visualization_path, f"{base_name}{SPECTROGRAM_SUFFIX}")
                visualize_spectrograms(audio[0], perturbed.T[0], sr, spec_output)

            if visualize_diff:
                diff_output = None
                if visualization_path:
                    base_name = os.path.splitext(os.path.basename(output_file))[0]
                    diff_output = os.path.join(visualization_path, f"{base_name}{DIFFERENCE_SUFFIX}")
                visualize_difference(audio[0], perturbed.T[0], sr, diff_output)

        logging.info(f"Process completed successfully in {processing_time:.2f} seconds for {input_file}.")
        return processing_time, file_size_ratio, None

    except Exception as e:
        error_msg = f"Error processing {input_file}: {str(e)}"
        logging.error(error_msg)
        return None, None, error_msg


def batch_process(
    input_dir: str,
    output_dir: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    force_mono: bool = False,
    adaptive_scaling: bool = True,
    file_extension: str = ".wav",
    visualize: bool = False,
    visualize_diff: bool = False,
    parallel: bool = False,
    workers: Optional[int] = None
) -> None:
    """
    Process all audio files in a directory.

    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed files
        window_size: STFT window size
        hop_size: STFT hop size
        noise_scale: Noise scaling factor
        force_mono: Convert stereo to mono before processing
        adaptive_scaling: Enable adaptive noise scaling based on signal strength
        file_extension: File extension to process (e.g., '.wav')
        visualize: Generate spectrogram visualizations
        visualize_diff: Generate difference visualizations
        parallel: Use parallel processing (multiprocessing)
        workers: Number of worker processes (None for CPU count)
    """
    os.makedirs(output_dir, exist_ok=True)

    vis_dir = os.path.join(output_dir, "visualizations")
    if visualize or visualize_diff:
        os.makedirs(vis_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(file_extension)]

    if not files:
        logging.warning(f"No {file_extension} files found in {input_dir}")
        return

    logging.info(f"Found {len(files)} {file_extension} files to process")

    if parallel:
        _batch_process_parallel(
            input_dir, output_dir, vis_dir, files,
            window_size, hop_size, noise_scale,
            force_mono, adaptive_scaling, visualize, visualize_diff,
            workers
        )
    else:
        _batch_process_sequential(
            input_dir, output_dir, vis_dir, files,
            window_size, hop_size, noise_scale,
            force_mono, adaptive_scaling, visualize, visualize_diff
        )


def _batch_process_sequential(
    input_dir: str,
    output_dir: str,
    vis_dir: str,
    files: List[str],
    window_size: int,
    hop_size: int,
    noise_scale: float,
    force_mono: bool,
    adaptive_scaling: bool,
    visualize: bool,
    visualize_diff: bool
) -> None:
    """Sequential batch processing implementation."""
    total_time = 0
    successful_processing_count = 0
    failed_files_info = []

    for i, file_name in enumerate(files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        logging.info(f"Processing file {i+1}/{len(files)}: {file_name}")
        proc_time, size_ratio, error_msg = process_audio_file(
            input_path, output_path, window_size, hop_size, noise_scale,
            force_mono, adaptive_scaling, visualize, visualize_diff, vis_dir
        )

        if error_msg is None and proc_time is not None and size_ratio is not None:
            total_time += proc_time
            successful_processing_count += 1
        else:
            failed_files_info.append((file_name, error_msg or "Unknown error"))

    _report_batch_summary(
        successful_processing_count, len(files), total_time,
        output_dir, vis_dir, visualize or visualize_diff, failed_files_info
    )


def _process_file_wrapper(args: Dict[str, Any]) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
    """Wrapper function for multiprocessing."""
    file_name = args["file_name"]
    input_path = args["input_path"]
    output_path = args["output_path"]

    proc_time, size_ratio, error_msg = process_audio_file(
        input_path, output_path,
        args["window_size"], args["hop_size"], args["noise_scale"],
        args["force_mono"], args["adaptive_scaling"],
        args["visualize"], args["visualize_diff"], args["vis_dir"]
    )

    return file_name, proc_time, size_ratio, error_msg


def _batch_process_parallel(
    input_dir: str,
    output_dir: str,
    vis_dir: str,
    files: List[str],
    window_size: int,
    hop_size: int,
    noise_scale: float,
    force_mono: bool,
    adaptive_scaling: bool,
    visualize: bool,
    visualize_diff: bool,
    workers: Optional[int] = None
) -> None:
    """Parallel batch processing implementation using ProcessPoolExecutor."""
    # Determine worker count (default to CPU count)
    if workers is None:
        workers = multiprocessing.cpu_count()

    logging.info(f"Using parallel processing with {workers} workers")
    start_time = time.time()

    # Prepare arguments for each file
    process_args = []
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        process_args.append({
            "file_name": file_name,
            "input_path": input_path,
            "output_path": output_path,
            "window_size": window_size,
            "hop_size": hop_size,
            "noise_scale": noise_scale,
            "force_mono": force_mono,
            "adaptive_scaling": adaptive_scaling,
            "visualize": visualize,
            "visualize_diff": visualize_diff,
            "vis_dir": vis_dir
        })

    # Process files in parallel
    successful_processing_count = 0
    failed_files_info = []
    total_processing_time = 0.0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (file_name, proc_time, size_ratio, error_msg) in enumerate(
            executor.map(_process_file_wrapper, process_args)
        ):
            logging.info(f"Completed {i+1}/{len(files)}: {file_name}")

            if error_msg is None and proc_time is not None and size_ratio is not None:
                successful_processing_count += 1
                total_processing_time += proc_time
            else:
                failed_files_info.append((file_name, error_msg or "Unknown error"))

    total_wall_time = time.time() - start_time
    logging.info(f"Parallel processing completed in {total_wall_time:.2f} seconds wall time")
    logging.info(f"Accumulated processing time: {total_processing_time:.2f} seconds")
    logging.info(f"Speed improvement: {total_processing_time / max(1, total_wall_time):.2f}x")

    _report_batch_summary(
        successful_processing_count, len(files), total_processing_time,
        output_dir, vis_dir, visualize or visualize_diff, failed_files_info
    )


def _report_batch_summary(
    successful_count: int,
    total_count: int,
    total_time: float,
    output_dir: str,
    vis_dir: str,
    has_visualizations: bool,
    failed_files_info: List[Tuple[str, str]]
) -> None:
    """Report batch processing summary."""
    logging.info("\n----- Batch Processing Summary -----")
    logging.info(f"Successfully processed files: {successful_count}/{total_count}")
    logging.info(f"Total processing time for successful files: {total_time:.2f} seconds")
    logging.info(f"Processed files saved to: {output_dir}")
    if has_visualizations:
        logging.info(f"Visualizations saved to: {vis_dir}")

    if failed_files_info:
        logging.warning(f"\nEncountered errors with {len(failed_files_info)} file(s):")
        for file_name, err_msg in failed_files_info:
            logging.warning(f"  - {file_name}: {err_msg}")
    logging.info("----- End of Batch Processing Summary -----")

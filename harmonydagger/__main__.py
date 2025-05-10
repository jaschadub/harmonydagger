#!/usr/bin/env python3
"""
Command-line interface for HarmonyDagger.
"""
import argparse
import logging
import os
import sys
from typing import List

from .common import DEFAULT_HOP_SIZE, DEFAULT_NOISE_SCALE, DEFAULT_WINDOW_SIZE
from .file_operations import batch_process, process_audio_file


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HarmonyDagger: Psychoacoustic noise generator for audio files'
    )

    # Input/output options
    parser.add_argument('input',
                      help='Input audio file or directory')
    parser.add_argument('output',
                      help='Output file or directory')

    # Processing parameters
    parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE,
                      help=f'STFT window size (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--hop-size', type=int, default=DEFAULT_HOP_SIZE,
                      help=f'STFT hop size (default: {DEFAULT_HOP_SIZE})')
    parser.add_argument('--noise-scale', type=float, default=DEFAULT_NOISE_SCALE,
                      help=f'Noise scaling factor (default: {DEFAULT_NOISE_SCALE})')

    # Audio options
    parser.add_argument('--force-mono', action='store_true',
                      help='Convert stereo files to mono before processing')
    parser.add_argument('--adaptive-scaling', action='store_true', default=True,
                      help='Use adaptive noise scaling (default: True)')
    parser.add_argument('--no-adaptive-scaling', action='store_false', dest='adaptive_scaling',
                      help='Disable adaptive noise scaling')

    # Batch processing
    parser.add_argument('--ext', default='.wav',
                      help='File extension to process in batch mode (default: .wav)')
    parser.add_argument('--parallel', action='store_true',
                      help='Use parallel processing for batch operations')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes for parallel processing (default: CPU count)')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                      help='Generate spectrogram visualizations')
    parser.add_argument('--visualize-diff', action='store_true',
                      help='Generate difference visualizations')

    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')

    return parser.parse_args(args)

def main() -> int:
    """Main entry point for the command-line interface."""
    args = parse_args(sys.argv[1:])
    setup_logging(args.verbose)

    # Determine if input is a file or directory
    is_batch = os.path.isdir(args.input)

    try:
        if is_batch:
            logging.info(f"Batch processing directory: {args.input}")
            os.makedirs(args.output, exist_ok=True)

            batch_process(
                args.input,
                args.output,
                window_size=args.window_size,
                hop_size=args.hop_size,
                noise_scale=args.noise_scale,
                force_mono=args.force_mono,
                adaptive_scaling=args.adaptive_scaling,
                file_extension=args.ext,
                visualize=args.visualize,
                visualize_diff=args.visualize_diff,
                parallel=args.parallel,
                workers=args.workers
            )
        else:
            logging.info(f"Processing single file: {args.input}")
            vis_path = os.path.dirname(args.output) if args.visualize or args.visualize_diff else None

            _, _, error = process_audio_file(
                args.input,
                args.output,
                window_size=args.window_size,
                hop_size=args.hop_size,
                noise_scale=args.noise_scale,
                force_mono=args.force_mono,
                adaptive_scaling=args.adaptive_scaling,
                visualize=args.visualize,
                visualize_diff=args.visualize_diff,
                visualization_path=vis_path
            )

            if error:
                logging.error(f"Processing failed: {error}")
                return 1

        return 0

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

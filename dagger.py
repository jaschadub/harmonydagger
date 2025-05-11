#!/usr/bin/env python3
"""
HarmonyDagger: Making Music Unlearnable for Generative AI

This tool implements the HarmonyCloak approach, introducing imperceptible
psychoacoustic noise into audio files to protect them from being learned
by generative AI models, while preserving perceptual quality for human listeners.

Based on research paper: "HarmonyCloak: Making Music Unlearnable for Generative AI"
by Syed Irfan Ali Meerza, Lichao Sun, and Jian Liu
"""

import argparse
import logging
import os
import sys

# Ensure the package directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import constants from the common module explicitly
from harmonydagger.common import (
    DEFAULT_HOP_SIZE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WINDOW_SIZE,
)

# Import file operation functions which orchestrate the rest
from harmonydagger.file_operations import (
    parallel_batch_process,
    process_audio_file,
    recursive_find_audio_files,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Note: Other modules like psychoacoustics, core, visualization are used by
# file_operations and do not need to be directly imported here anymore.

# ===== Main Execution =====

def main(args=None):
    """
    Main execution function that processes command line arguments.

    Args:
        args: Command line arguments parsed by argparse
    """
    # If no args provided, parse from command line
    if args is None:
        args = parse_args()

    # Single file processing
    if not args.batch_mode:
        logging.info(f"Starting single file processing for: {args.input_file}")
        success, output_path, proc_time = process_audio_file(
            args.input_file,
            args.output_file,
            args.window_size,
            args.hop_size,
            args.noise_scale,
            args.adaptive_scaling,
            args.force_mono
        )
        
        if success:
            logging.info(f"Successfully processed file in {proc_time:.2f} seconds")
            logging.info(f"Saved to: {output_path}")
            
            # Add visualization if requested
            if args.visualize or args.visualize_diff:
                try:
                    from harmonydagger.visualization import create_audio_comparison
                    create_audio_comparison(
                        args.input_file,
                        output_path,
                        visualize_diff=args.visualize_diff
                    )
                except Exception as e:
                    logging.error(f"Error creating visualizations: {str(e)}")
        else:
            logging.error(f"Error processing file: {output_path}")
            return 1

    # Batch processing mode
    else:
        if not args.output_dir:
            logging.error("Error: --output_dir is required for batch processing")
            return 1

        logging.info(f"Starting batch processing from input_dir: {args.input_dir}")
        
        # Find audio files
        extensions = [args.file_extension] if args.file_extension.startswith('.') else [f".{args.file_extension}"]
        input_files = recursive_find_audio_files(args.input_dir, extensions)
        
        if not input_files:
            logging.error(f"No {args.file_extension} files found in {args.input_dir}")
            return 1
            
        logging.info(f"Found {len(input_files)} audio files to process")
        
        # Process files in parallel
        results = parallel_batch_process(
            input_files,
            args.output_dir,
            args.window_size,
            args.hop_size,
            args.noise_scale,
            args.adaptive_scaling,
            args.force_mono,
            args.max_workers
        )
        
        # Report results
        successful = sum(1 for result in results.values() if result['success'])
        failed = len(results) - successful
        total_time = sum(result['processing_time'] for result in results.values())
        
        logging.info(f"Batch processing complete: {successful} succeeded, {failed} failed")
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Create visualizations if requested (for successful files only)
        if args.visualize or args.visualize_diff:
            try:
                from harmonydagger.visualization import create_audio_comparison
                for input_path, result in results.items():
                    if result['success']:
                        output_path = result['output_path']
                        create_audio_comparison(
                            input_path,
                            output_path,
                            visualize_diff=args.visualize_diff
                        )
            except Exception as e:
                logging.error(f"Error creating visualizations: {str(e)}")
                
    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HarmonyDagger - Make Music Unlearnable for Generative AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output arguments
    input_group = parser.add_argument_group("Input/Output")
    input_group.add_argument("--batch_mode", action="store_true",
                            help="Process all audio files in a directory")
    input_group.add_argument("--input_dir", type=str,
                            help="Directory containing input audio files (for batch mode)")
    input_group.add_argument("--output_dir", type=str,
                            help="Directory to save processed files (for batch mode)")
    input_group.add_argument("--file_extension", type=str, default=".wav",
                            help="File extension to process in batch mode")

    # Make input_file and output_file optional for batch mode
    input_group.add_argument("input_file", type=str, nargs="?",
                            help="Path to input audio file")
    input_group.add_argument("output_file", type=str, nargs="?",
                            help="Path to save perturbed audio file")

    # Processing parameters
    param_group = parser.add_argument_group("Processing Parameters")
    param_group.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE,
                            help="STFT window size")
    param_group.add_argument("--hop_size", type=int, default=DEFAULT_HOP_SIZE,
                            help="STFT hop size")
    param_group.add_argument("--noise_scale", type=float, default=DEFAULT_NOISE_SCALE,
                            help="Noise scale factor (0-1)")
    param_group.add_argument("--adaptive_scaling", action="store_true",
                            help="Use adaptive noise scaling based on signal strength")
    param_group.add_argument("--force_mono", action="store_true",
                            help="Force stereo input to mono before processing")
    param_group.add_argument("--max_workers", type=int, default=None,
                            help="Maximum number of worker processes for batch processing (default: auto)")

    # Visualization options
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument("--visualize", action="store_true",
                          help="Show spectrogram comparison of original and perturbed audio")
    vis_group.add_argument("--visualize_diff", action="store_true",
                          help="Visualize the difference between original and perturbed audio")

    args = parser.parse_args()

    # Validate arguments
    if args.batch_mode:
        if not args.input_dir:
            parser.error("--input_dir is required for batch mode")
    else:
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required for single file mode")
            
    return args


if __name__ == "__main__":
    sys.exit(main())

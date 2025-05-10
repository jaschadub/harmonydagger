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
# import numpy as np # No longer directly used in this file
# import librosa # No longer directly used in this file
# import soundfile as sf # No longer directly used in this file
# from scipy.signal import stft, istft # Moved to core.py
# import matplotlib.pyplot as plt # Moved to visualization.py
# import os # No longer directly used in this file
# import time # No longer directly used in this file
import logging
# from typing import Tuple, List, Union, Optional # No longer directly used for function signatures here
# from numpy.typing import NDArray # No longer directly used in this file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import constants from the common module
from harmonydagger.common import *

# Import psychoacoustic functions
from harmonydagger.psychoacoustics import (
    hearing_threshold,
    bark_scale,
    critical_band_width,
    magnitude_to_db,
    db_to_magnitude
)

# Import core processing functions
from harmonydagger.core import (
    generate_psychoacoustic_noise,
    apply_noise_multichannel,
    apply_noise_to_audio
)

# Import visualization functions
from harmonydagger.visualization import (
    visualize_spectrograms,
    visualize_difference
)

# Import file operation functions
from harmonydagger.file_operations import (
    process_audio_file,
    batch_process
)

# ===== Main Execution =====

def main(args):
    """
    Main execution function that processes command line arguments.
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Single file processing
    if not args.batch_mode:
        logging.info(f"Starting single file processing for: {args.input_file}")
        process_audio_file(
            args.input_file,
            args.output_file,
            args.window_size,
            args.hop_size,
            args.noise_scale,
            args.force_mono,
            args.adaptive_scaling,
            args.visualize,
            args.visualize_diff
        )
    
    # Batch processing mode
    else:
        if not args.output_dir:
            logging.error("Error: --output_dir is required for batch processing")
            parser.error("--output_dir is required for batch processing") # Keep parser error for CLI exit
            return
        
        logging.info(f"Starting batch processing from input_dir: {args.input_dir}")
        batch_process(
            args.input_dir,
            args.output_dir,
            args.window_size,
            args.hop_size,
            args.noise_scale,
            args.force_mono,
            args.adaptive_scaling,
            args.file_extension,
            args.visualize,
            args.visualize_diff
        )


if __name__ == "__main__":
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
    
    main(args)

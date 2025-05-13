#!/usr/bin/env python3
"""
Command-line interface for HarmonyDagger.
"""
import argparse
import sys
import time
from pathlib import Path

from harmonydagger import __version__
from harmonydagger.common import setup_logger
from harmonydagger.file_operations import process_audio_file

logger = setup_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HarmonyDagger: Protect audio against AI voice cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("input", help="Input audio file or directory containing audio files")
    parser.add_argument("-o", "--output", help="Output file or directory (default: input_protected.wav)")
    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=2048,
        help="STFT window size"
    )
    parser.add_argument(
        "-s",
        "--hop-size",
        type=int,
        default=512,
        help="STFT hop size"
    )
    parser.add_argument(
        "-n",
        "--noise-scale",
        type=float,
        default=0.1,
        help="Noise scale (0-1)"
    )
    parser.add_argument(
        "-a",
        "--adaptive-scaling",
        action="store_true",
        help="Use adaptive noise scaling based on signal strength"
    )
    parser.add_argument(
        "-m",
        "--force-mono",
        action="store_true",
        help="Convert stereo to mono before processing"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel processing jobs (for batch processing)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["wav", "mp3", "flac", "ogg", "all"],
        default="all",
        help="Specify audio format to process (when processing directories)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"HarmonyDagger {__version__}"
    )

    args = parser.parse_args()

    # Configure logger verbosity
    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    input_path = Path(args.input)
    
    # Handle single file processing
    if input_path.is_file():
        logger.info(f"Processing file: {input_path}")
        
        # Set default output path if not specified
        if args.output is None:
            output_name = f"{input_path.stem}_protected{input_path.suffix}"
            output_path = input_path.parent / output_name
        else:
            output_path = Path(args.output)
            
            # If output is a directory, create a filename inside it
            if output_path.is_dir():
                output_path = output_path / f"{input_path.stem}_protected{input_path.suffix}"

        start_time = time.time()
        
        try:
            # Check if file exists
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                return 1
                
            # Check if file is empty
            if input_path.stat().st_size == 0:
                logger.error(f"Input file is empty: {input_path}")
                return 1
                
            # Verify file is a valid audio file
            try:
                import soundfile as sf
                file_info = sf.info(str(input_path))
                logger.debug(f"Audio file info: {file_info}")
            except Exception as sf_error:
                logger.error(f"Failed to read audio file: {str(sf_error)}")
                return 1
            
            # Log details about the process
            logger.debug(f"Processing parameters:")
            logger.debug(f"  Window size: {args.window_size}")
            logger.debug(f"  Hop size: {args.hop_size}")
            logger.debug(f"  Noise scale: {args.noise_scale}")
            logger.debug(f"  Adaptive scaling: {args.adaptive_scaling}")
            logger.debug(f"  Force mono: {args.force_mono}")
            
            success, out_path, processing_time = process_audio_file(
                str(input_path),
                str(output_path),
                window_size=args.window_size,
                hop_size=args.hop_size,
                noise_scale=args.noise_scale,
                adaptive_scaling=args.adaptive_scaling,
                force_mono=args.force_mono,
                visualize=args.verbose,  # Generate visualizations when in verbose mode
                visualize_diff=args.verbose,
            )
            
            if success:
                logger.info(f"Successfully processed file in {processing_time:.2f}s")
                logger.info(f"Output saved to: {out_path}")
                return 0
            else:
                logger.error(f"Processing failed: {out_path}")
                return 1
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            if args.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return 1

    # Handle batch directory processing
    elif input_path.is_dir():
        # Ensure output directory exists if specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path / "protected"
            output_dir.mkdir(exist_ok=True)
            
        logger.info(f"Processing all audio files in: {input_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Determine which file extensions to process based on format argument
        if args.format == "all":
            audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]
            logger.info("Processing all supported audio formats (WAV, MP3, FLAC, OGG)")
        else:
            audio_extensions = [f".{args.format}"]
            logger.info(f"Processing only {args.format.upper()} files")
            
        # Find audio files with specified extensions (case-insensitive)
        audio_files = []
        for ext in audio_extensions:
            # Try lowercase extension
            files = list(input_path.glob(f"*{ext}"))
            audio_files.extend(files)
            
            # Try uppercase extension
            upper_ext = ext.upper()
            upper_files = list(input_path.glob(f"*{upper_ext}"))
            audio_files.extend(upper_files)
            
            if files or upper_files:
                logger.info(f"Found {len(files) + len(upper_files)} files with extension {ext}")
        
        if not audio_files:
            logger.error(f"No audio files found in {input_path}")
            return 1
            
        logger.info(f"Found {len(audio_files)} audio file(s)")
        
        # Process files sequentially or in parallel
        if args.jobs > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            
            # Limit jobs to CPU count
            jobs = min(args.jobs, multiprocessing.cpu_count())
            logger.info(f"Using {jobs} parallel processing jobs")
            
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = []
                
                for audio_file in audio_files:
                    output_path = output_dir / f"{audio_file.stem}_protected{audio_file.suffix}"
                    
                    futures.append(
                        executor.submit(
                            process_audio_file,
                            str(audio_file),
                            str(output_path),
                            args.window_size,
                            args.hop_size,
                            args.noise_scale,
                            args.adaptive_scaling,
                            args.force_mono,
                        )
                    )
                
                success_count = 0
                for future in futures:
                    success, _, _ = future.result()
                    if success:
                        success_count += 1
                        
                logger.info(f"Successfully processed {success_count}/{len(audio_files)} files")
                return 0 if success_count == len(audio_files) else 1
        else:
            # Process sequentially
            start_time = time.time()
            success_count = 0
            
            for audio_file in audio_files:
                output_path = output_dir / f"{audio_file.stem}_protected{audio_file.suffix}"
                logger.info(f"Processing: {audio_file.name}")
                
                success, _, _ = process_audio_file(
                    str(audio_file),
                    str(output_path),
                    window_size=args.window_size,
                    hop_size=args.hop_size,
                    noise_scale=args.noise_scale,
                    adaptive_scaling=args.adaptive_scaling,
                    force_mono=args.force_mono,
                )
                
                if success:
                    success_count += 1
                    
            total_time = time.time() - start_time
            logger.info(f"Batch processing complete in {total_time:.2f}s")
            logger.info(f"Successfully processed {success_count}/{len(audio_files)} files")
            return 0 if success_count == len(audio_files) else 1
    
    else:
        logger.error(f"Input path not found: {input_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

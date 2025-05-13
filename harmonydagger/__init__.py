"""
HarmonyDagger: A tool for protecting audio against AI voice cloning.

This package provides functions and tools to process audio files and add
psychoacoustically-masked noise that helps protect against voice cloning while
remaining imperceptible to human listeners.
"""

from .core import (
    apply_noise_multichannel,
    apply_noise_to_audio,
    generate_psychoacoustic_noise,
)
from .file_operations import (
    parallel_batch_process,
    process_audio_file,
    recursive_find_audio_files,
)

__version__ = "0.3.1"

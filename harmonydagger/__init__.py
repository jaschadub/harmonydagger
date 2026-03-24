"""
HarmonyDagger: A tool for protecting audio against AI

This package provides functions and tools to process audio files and add
psychoacoustically-masked noise that helps protect while
remaining imperceptible to human listeners.
"""

from .benchmark import compute_snr, generate_benchmark_report
from .core import (
    apply_noise_multichannel,
    apply_noise_to_audio,
    generate_protected_audio,
    generate_psychoacoustic_noise,
)
from .file_operations import (
    parallel_batch_process,
    process_audio_file,
    recursive_find_audio_files,
)
from .phase import generate_phase_perturbation
from .robustness import augment_and_check_survival, simulate_mp3_compression
from .temporal_masking import apply_temporal_masking
from .verify import compute_feature_similarity, verify_protection
from .vocal_mode import apply_vocal_emphasis, compute_vocal_emphasis_curve

__version__ = "0.4.0"

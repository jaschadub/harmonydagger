"""
Common constants for the HarmonyDagger package.
"""
import logging


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name for the logger, typically __name__
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Constants for psychoacoustic modeling
REFERENCE_PRESSURE = 20e-6  # Reference pressure in air (20 μPa)
MASKING_CURVE_SLOPE = 0.8   # Slope for frequency masking (dB/Bark)
DB_LOG_EPSILON = 1e-10      # Epsilon for log10 to avoid log(0)
HZ_TO_KHZ = 1000.0          # Conversion factor for Hz to kHz

# Constants for hearing_threshold function (ISO 226:2003 simplified)
HEARING_THRESH_F_POW = -0.8
HEARING_THRESH_C1 = 3.64
HEARING_THRESH_C2 = -6.5
HEARING_THRESH_EXP_C1 = -0.6
HEARING_THRESH_F_OFFSET = 3.3

# Constants for bark_scale function
BARK_SCALE_C1 = 13.0
BARK_SCALE_C2 = 0.00076
BARK_SCALE_C3 = 3.5
BARK_SCALE_F_DIV = 7500.0

# Constants for critical_band_width function (Zwicker's model simplified)
CBW_C1 = 25.0
CBW_C2 = 75.0
CBW_C3 = 1.4
CBW_F_POW = 0.69

# Constants for adaptive noise scaling
ADAPTIVE_SCALE_NORM_MIN = 0.5
ADAPTIVE_SCALE_NORM_RANGE = 1.0 # Max will be MIN + RANGE = 1.5
ADAPTIVE_SIGNAL_STRENGTH_DIV = 60.0
NOISE_UPPER_BOUND_FACTOR = 0.8 # Noise magnitude upper bound relative to signal

# Default STFT parameters (can be overridden by CLI)
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_HOP_SIZE = 512
DEFAULT_NOISE_SCALE = 0.01

# Visualization defaults
DEFAULT_FIGSIZE_SPECTROGRAM = (12, 8)
DEFAULT_FIGSIZE_DIFFERENCE = (12, 10)
DEFAULT_VIS_NFFT = 1024
DEFAULT_VIS_NOVERLAP = 512
DEFAULT_VIS_DPI = 300
SPECTROGRAM_SUFFIX = "_spectrogram.png"
DIFFERENCE_SUFFIX = "_difference.png"

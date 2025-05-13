"""
Utility functions for WAV file operations.
"""
import logging
import numpy as np
import os
import struct
import wave

logger = logging.getLogger(__name__)

def save_wav_file(file_path: str, audio: np.ndarray, sample_rate: int) -> bool:
    """
    Save audio data to a WAV file using wave module (instead of soundfile).
    
    Args:
        file_path: Path to save the WAV file
        audio: Audio data as numpy array (values should be in [-1.0, 1.0] range)
        sample_rate: Sample rate in Hz
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Convert float64 to int16 for WAV file
        # Scale to full int16 range while clipping to avoid overflow
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        # Handle mono or stereo
        num_channels = 1
        if audio.ndim > 1:
            num_channels = audio.shape[0]
            # Transpose if needed to make compatible with wave module
            if audio_int16.shape[0] < audio_int16.shape[1]:
                audio_int16 = audio_int16.T
        
        # Open wave file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            
            # Write data
            if num_channels == 1:
                wf.writeframes(audio_int16.tobytes())
            else:
                # For multi-channel, handle interleaving
                wf.writeframes(audio_int16.tobytes())
        
        logger.debug(f"Successfully saved WAV file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving WAV file: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def read_wav_file(file_path: str) -> tuple:
    """
    Read audio data from a WAV file using wave module.
    
    Args:
        file_path: Path to the WAV file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get file properties
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read raw audio data
            raw_data = wf.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit audio
                data = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit audio
                data = np.frombuffer(raw_data, dtype=np.int32)
            else:
                data = np.frombuffer(raw_data, dtype=np.uint8)
                
            # Normalize to [-1.0, 1.0]
            if sample_width == 1:  # 8-bit unsigned
                data = data.astype(np.float64) / 128.0 - 1.0
            elif sample_width == 2:  # 16-bit signed
                data = data.astype(np.float64) / 32768.0
            else:  # 32-bit signed
                data = data.astype(np.float64) / 2147483648.0
                
            # Reshape for multiple channels
            if n_channels > 1:
                data = data.reshape(-1, n_channels).T
                
            return data, sample_rate
            
    except Exception as e:
        logger.error(f"Error reading WAV file {file_path}: {str(e)}")
        raise

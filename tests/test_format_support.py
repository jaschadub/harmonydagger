"""
Tests for audio format support in HarmonyDagger.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from harmonydagger.file_operations import process_audio_file


class AudioFormatSupportTest(unittest.TestCase):
    """Test support for different audio formats."""

    def setUp(self):
        """Create test audio data."""
        self.sample_rate = 44100
        # Create 1 second of audio data (sine wave)
        t = np.linspace(0, 1.0, self.sample_rate, False)
        self.test_audio = 0.5 * np.sin(2.0 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create WAV reference file
        self.wav_path = self.temp_path / "test_audio.wav"
        sf.write(self.wav_path, self.test_audio, self.sample_rate)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_wav_processing(self):
        """Test processing of WAV files."""
        output_path = self.temp_path / "test_output.wav"
        
        success, path, _ = process_audio_file(
            str(self.wav_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify the output file can be read
        processed_audio, sr = sf.read(output_path)
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(len(processed_audio), len(self.test_audio))

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/ffmpeg"),
        reason="Requires ffmpeg for MP3 support"
    )
    def test_mp3_processing(self):
        """Test processing of MP3 files."""
        # Create an MP3 file from the WAV using pydub
        from pydub import AudioSegment
        
        mp3_path = self.temp_path / "test_audio.mp3"
        AudioSegment.from_wav(str(self.wav_path)).export(str(mp3_path), format="mp3", bitrate="192k")
        
        output_path = self.temp_path / "test_output.mp3"
        
        success, path, _ = process_audio_file(
            str(mp3_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify by converting back to WAV and comparing
        output_audio = AudioSegment.from_mp3(str(output_path))
        temp_wav = self.temp_path / "temp_converted.wav"
        output_audio.export(str(temp_wav), format="wav")
        
        # Check if the file can be read and has approximately the same length
        processed_audio, sr = sf.read(temp_wav)
        self.assertEqual(sr, self.sample_rate)
        
        # MP3 compression may change the exact sample count, so check approximate duration
        duration_original = len(self.test_audio) / self.sample_rate
        duration_processed = len(processed_audio) / sr
        self.assertAlmostEqual(duration_original, duration_processed, delta=0.1)
    
    def test_flac_processing(self):
        """Test processing of FLAC files."""
        # Create a FLAC file
        flac_path = self.temp_path / "test_audio.flac"
        sf.write(flac_path, self.test_audio, self.sample_rate, format='FLAC')
        
        output_path = self.temp_path / "test_output.flac"
        
        success, path, _ = process_audio_file(
            str(flac_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify the output FLAC file can be read
        processed_audio, sr = sf.read(output_path)
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(len(processed_audio), len(self.test_audio))
    
    def test_ogg_processing(self):
        """Test processing of OGG files."""
        # Create an OGG file
        ogg_path = self.temp_path / "test_audio.ogg"
        sf.write(ogg_path, self.test_audio, self.sample_rate, format='OGG')
        
        output_path = self.temp_path / "test_output.ogg"
        
        success, path, _ = process_audio_file(
            str(ogg_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify the output OGG file can be read
        processed_audio, sr = sf.read(output_path)
        self.assertEqual(sr, self.sample_rate)
        # OGG is lossy, so the sample count might differ slightly
        self.assertAlmostEqual(len(processed_audio) / sr, len(self.test_audio) / self.sample_rate, delta=0.1)
    
    def test_format_conversion(self):
        """Test conversion between different formats."""
        # Process WAV to MP3
        output_path = self.temp_path / "wav_to_mp3.mp3"
        
        success, path, _ = process_audio_file(
            str(self.wav_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        # Check if either MP3 output or fallback WAV file exists (depending on ffmpeg availability)
        output_base = os.path.splitext(output_path)[0]
        self.assertTrue(
            os.path.exists(output_path) or os.path.exists(f"{output_base}.wav"),
            "Neither MP3 nor fallback WAV file was created"
        )
        
        # Process WAV to FLAC
        output_path = self.temp_path / "wav_to_flac.flac"
        
        success, path, _ = process_audio_file(
            str(self.wav_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Process WAV to OGG
        output_path = self.temp_path / "wav_to_ogg.ogg"
        
        success, path, _ = process_audio_file(
            str(self.wav_path),
            str(output_path),
            window_size=1024,
            hop_size=256,
            noise_scale=0.1
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()

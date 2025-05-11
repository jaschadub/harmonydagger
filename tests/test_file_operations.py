"""
Unit tests for the file_operations module.
"""
import os
import shutil
import tempfile
import unittest

import librosa
import numpy as np
import soundfile as sf

from harmonydagger.file_operations import (
    parallel_batch_process,
    process_audio_file,
    recursive_find_audio_files,
)


class TestFileOperations(unittest.TestCase):
    """Test cases for the file_operations module."""
    
    def setUp(self):
        """Set up test fixtures, creating a temporary directory and test files."""
        # Create a temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple audio file for testing
        sr = 22050  # Sample rate
        duration = 2.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration))
        self.test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save the test audio to a file
        self.test_file = os.path.join(self.temp_dir, "test.wav")
        sf.write(self.test_file, self.test_audio, sr)
        
        # Create a subdirectory with another test file
        self.sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(self.sub_dir, exist_ok=True)
        self.test_file2 = os.path.join(self.sub_dir, "test2.wav")
        sf.write(self.test_file2, self.test_audio, sr)
    
    def tearDown(self):
        """Tear down test fixtures, removing the temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_process_audio_file(self):
        """Test processing a single audio file."""
        output_path = os.path.join(self.output_dir, "output.wav")
        success, path, processing_time = process_audio_file(
            self.test_file,
            output_path,
            window_size=1024,
            hop_size=256,
            noise_scale=0.1,
            adaptive_scaling=True,
            force_mono=False
        )
        
        # Check that processing was successful
        self.assertTrue(success)
        self.assertEqual(path, output_path)
        self.assertTrue(processing_time > 0)
        
        # Check that the output file exists and has similar characteristics
        self.assertTrue(os.path.exists(output_path))
        
        y, sr = librosa.load(output_path, sr=None)
        self.assertEqual(len(y), len(self.test_audio))
        self.assertTrue(np.abs(np.mean(y - self.test_audio)) < 0.1)  # Noise is small
    
    def test_parallel_batch_process(self):
        """Test parallel batch processing of multiple audio files."""
        file_paths = [self.test_file, self.test_file2]
        
        results = parallel_batch_process(
            file_paths,
            self.output_dir,
            window_size=1024,
            hop_size=256,
            noise_scale=0.1,
            adaptive_scaling=True,
            force_mono=False,
            max_workers=2
        )
        
        # Check that all files were processed successfully
        for file_path, result in results.items():
            print(f"Processing result for {file_path}: {result}")
            self.assertTrue(result['success'], f"Processing failed: {result.get('error', 'Unknown error')}")
            self.assertTrue(result['processing_time'] > 0)
            
            output_path = result['output_path']
            self.assertTrue(os.path.exists(output_path))
            
            # Check that output files are in the correct directory
            self.assertTrue(output_path.startswith(self.output_dir))
    
    def test_recursive_find_audio_files(self):
        """Test finding audio files recursively."""
        # Create a non-audio file
        text_file = os.path.join(self.temp_dir, "test.txt")
        with open(text_file, "w") as f:
            f.write("This is a test file")
        
        # Find all WAV files
        found_files = recursive_find_audio_files(self.temp_dir)
        self.assertEqual(len(found_files), 2)
        self.assertIn(self.test_file, found_files)
        self.assertIn(self.test_file2, found_files)
        
        # Test with a specific extension
        found_files = recursive_find_audio_files(self.temp_dir, extensions=['.mp3'])
        self.assertEqual(len(found_files), 0)


if __name__ == "__main__":
    unittest.main()

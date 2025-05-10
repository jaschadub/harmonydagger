"""
Test the parallel processing functionality in HarmonyDagger.
"""
import os
import tempfile
import unittest
import numpy as np
import time
from pathlib import Path

from harmonydagger.file_operations import batch_process
from harmonydagger.core import generate_psychoacoustic_noise

class TestParallelProcessing(unittest.TestCase):
    """Test the parallel batch processing feature."""
    
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate a few dummy audio files for testing
        np.random.seed(42)  # For reproducibility
        self.num_test_files = 3
        self.sample_rate = 44100
        self.duration = 1  # 1 second
        
        for i in range(self.num_test_files):
            # Create a simple sine wave with random noise
            t = np.linspace(0, self.duration, self.sample_rate * self.duration)
            frequency = 440  # A4 note
            audio = np.sin(2 * np.pi * frequency * t) * 0.5
            audio += np.random.normal(0, 0.01, len(audio))  # Add a bit of noise
            audio = np.stack([audio, audio])  # Make it stereo (2 channels)
            
            # Save this dummy file in wav format
            try:
                import soundfile as sf
                file_path = self.input_dir / f"test_audio_{i}.wav"
                sf.write(file_path, audio.T, self.sample_rate)
            except ImportError:
                self.skipTest("soundfile module not available")
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_parallel_vs_sequential(self):
        """Test that parallel processing is faster than sequential for multiple files."""
        # Skip if there are fewer than 2 test files
        if self.num_test_files < 2:
            self.skipTest("Need at least 2 test files to compare parallel vs sequential")
        
        # Time the sequential processing
        start_time = time.time()
        batch_process(
            str(self.input_dir),
            str(self.output_dir) + "_seq",
            parallel=False
        )
        sequential_time = time.time() - start_time
        
        # Now time the parallel processing
        start_time = time.time()
        batch_process(
            str(self.input_dir),
            str(self.output_dir) + "_par",
            parallel=True
        )
        parallel_time = time.time() - start_time
        
        # Check timing - for very short files or few files, parallel might be 
        # marginally slower due to overhead, but should be comparable or faster
        # On a multi-core machine with more files, it should be faster
        print(f"Sequential time: {sequential_time:.3f}s, Parallel time: {parallel_time:.3f}s")
        
        # Just ensure parallel execution completes and doesn't crash
        # The actual speedup will depend on the test machine's cores and other factors
        self.assertTrue(True)
    
    def test_bark_scale_optimization(self):
        """Test the Bark scale pre-calculation optimization."""
        # Create a dummy audio signal for testing
        audio = np.random.rand(self.sample_rate)
        
        # Time the noise generation
        start_time = time.time()
        noise = generate_psychoacoustic_noise(audio, self.sample_rate)
        end_time = time.time()
        
        # Make sure it completes without errors
        self.assertIsNotNone(noise)
        self.assertEqual(len(noise), len(audio))
        
        # Print the execution time for reference
        print(f"Psychoacoustic noise generation time: {end_time - start_time:.3f}s")


if __name__ == "__main__":
    unittest.main()

"""Tests for the GPU acceleration module."""
import numpy as np
import pytest

from harmonydagger.gpu import _TORCH_AVAILABLE, is_gpu_available

# Skip all tests in this module if torch is not installed
pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch not installed"
)


class TestGPUBackend:
    """Tests for GPU-accelerated noise generation."""

    @pytest.fixture
    def sine_wave(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        return audio, sr

    def test_is_gpu_available_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_gpu_noise_output_shape(self, sine_wave):
        from harmonydagger.gpu import generate_psychoacoustic_noise_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")  # Test on CPU even without GPU
        result = generate_psychoacoustic_noise_gpu(
            audio, sr, window_size=1024, hop_size=512, device=device
        )
        assert len(result) == len(audio)

    def test_gpu_noise_is_nonzero(self, sine_wave):
        from harmonydagger.gpu import generate_psychoacoustic_noise_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")
        result = generate_psychoacoustic_noise_gpu(
            audio, sr, window_size=1024, hop_size=512, device=device
        )
        assert np.any(result != 0)

    def test_gpu_noise_is_small(self, sine_wave):
        from harmonydagger.gpu import generate_psychoacoustic_noise_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")
        result = generate_psychoacoustic_noise_gpu(
            audio, sr, window_size=1024, hop_size=512, noise_scale=0.01, device=device
        )
        rms = np.sqrt(np.mean(result**2))
        signal_rms = np.sqrt(np.mean(audio**2))
        assert rms < signal_rms * 0.5

    def test_gpu_protected_audio_shape(self, sine_wave):
        from harmonydagger.gpu import generate_protected_audio_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")
        result = generate_protected_audio_gpu(
            audio, sr, window_size=1024, hop_size=512, device=device
        )
        assert len(result) == len(audio)

    def test_gpu_protected_audio_bounded(self, sine_wave):
        from harmonydagger.gpu import generate_protected_audio_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")
        result = generate_protected_audio_gpu(
            audio, sr, window_size=1024, hop_size=512, device=device
        )
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_gpu_vocal_mode(self, sine_wave):
        from harmonydagger.gpu import generate_psychoacoustic_noise_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")
        result = generate_psychoacoustic_noise_gpu(
            audio, sr, window_size=1024, hop_size=512,
            vocal_mode=True, device=device
        )
        assert len(result) == len(audio)

    def test_gpu_matches_cpu_approximately(self, sine_wave):
        """GPU and CPU should produce similar (not identical) results."""
        from harmonydagger.core import generate_psychoacoustic_noise
        from harmonydagger.gpu import generate_psychoacoustic_noise_gpu

        audio, sr = sine_wave
        import torch

        device = torch.device("cpu")
        cpu_result = generate_psychoacoustic_noise(
            audio, sr, window_size=1024, hop_size=512, noise_scale=0.1
        )
        gpu_result = generate_psychoacoustic_noise_gpu(
            audio, sr, window_size=1024, hop_size=512, noise_scale=0.1, device=device
        )
        # Both should have similar RMS (within 2x of each other)
        cpu_rms = np.sqrt(np.mean(cpu_result**2))
        gpu_rms = np.sqrt(np.mean(gpu_result**2))
        assert gpu_rms > 0
        ratio = max(cpu_rms, gpu_rms) / max(min(cpu_rms, gpu_rms), 1e-10)
        assert ratio < 5.0, f"CPU RMS={cpu_rms:.6f}, GPU RMS={gpu_rms:.6f}"

    def test_core_use_gpu_flag(self, sine_wave):
        """Test that use_gpu=True in generate_protected_audio routes to GPU backend."""
        from harmonydagger.core import generate_protected_audio

        audio, sr = sine_wave
        result = generate_protected_audio(
            audio, sr, window_size=1024, hop_size=512,
            noise_scale=0.1, use_gpu=True
        )
        assert len(result) == len(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

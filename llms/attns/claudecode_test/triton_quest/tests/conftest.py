"""
Shared pytest fixtures for the Triton Quest test suite.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_progress_file(tmp_path, monkeypatch):
    """Create a temporary progress file for isolated testing."""
    progress_file = tmp_path / ".progress.json"

    # Monkeypatch the PROGRESS_FILE in quest module
    import quest
    monkeypatch.setattr(quest, "PROGRESS_FILE", progress_file)

    return progress_file


@pytest.fixture
def progress_with_level1_complete(temp_progress_file):
    """Create a progress file with level 1 completed."""
    progress = {
        "completed_levels": [1],
        "hints_used": {"1": 2}
    }
    with open(temp_progress_file, "w") as f:
        json.dump(progress, f)
    return temp_progress_file


@pytest.fixture
def progress_with_multiple_levels(temp_progress_file):
    """Create a progress file with multiple levels completed."""
    progress = {
        "completed_levels": [1, 2, 3],
        "hints_used": {"1": 6, "2": 3, "3": 1}
    }
    with open(temp_progress_file, "w") as f:
        json.dump(progress, f)
    return temp_progress_file


@pytest.fixture
def cuda_available():
    """Check if CUDA is available, skip test if not."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def sample_tensors_1d(cuda_available):
    """Create sample 1D CUDA tensors for testing."""
    import torch
    size = 1024
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    return x, y


@pytest.fixture
def sample_tensors_2d(cuda_available):
    """Create sample 2D CUDA tensors for testing."""
    import torch
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    return a, b


@pytest.fixture
def sample_attention_tensors(cuda_available):
    """Create sample attention tensors (Q, K, V) for testing."""
    import torch
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device='cuda')
    k = torch.randn(B, H, S, D, device='cuda')
    v = torch.randn(B, H, S, D, device='cuda')
    return q, k, v


# Pytest markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: tests that require CUDA GPU"
    )

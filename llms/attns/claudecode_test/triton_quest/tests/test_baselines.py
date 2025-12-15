"""
Tests for baseline implementations (requires CUDA).
"""

import pytest
import torch
import torch.nn.functional as F
import math


pytestmark = pytest.mark.cuda


class TestVectorAddBaseline:
    """Test vector_add_baseline correctness."""

    def test_basic_addition(self, cuda_available):
        """Test basic vector addition."""
        from utils.baseline import vector_add_baseline

        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = torch.tensor([4.0, 5.0, 6.0], device='cuda')

        result = vector_add_baseline(x, y)
        expected = torch.tensor([5.0, 7.0, 9.0], device='cuda')

        assert torch.allclose(result, expected)

    def test_random_vectors(self, sample_tensors_1d):
        """Test with random tensors."""
        from utils.baseline import vector_add_baseline

        x, y = sample_tensors_1d
        result = vector_add_baseline(x, y)
        expected = x + y

        assert torch.allclose(result, expected)


class TestMatmulBaseline:
    """Test matmul_baseline correctness."""

    def test_basic_matmul(self, cuda_available):
        """Test basic matrix multiplication."""
        from utils.baseline import matmul_baseline

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')

        result = matmul_baseline(a, b)
        expected = torch.matmul(a, b)

        assert torch.allclose(result, expected)

    def test_random_matrices(self, sample_tensors_2d):
        """Test with random tensors."""
        from utils.baseline import matmul_baseline

        a, b = sample_tensors_2d
        result = matmul_baseline(a, b)
        expected = torch.matmul(a, b)

        assert torch.allclose(result, expected)


class TestSoftmaxBaseline:
    """Test softmax_baseline correctness."""

    def test_basic_softmax(self, cuda_available):
        """Test basic softmax."""
        from utils.baseline import softmax_baseline

        x = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
        result = softmax_baseline(x, dim=-1)
        expected = F.softmax(x, dim=-1)

        assert torch.allclose(result, expected)

    def test_numerical_stability(self, cuda_available):
        """Test numerical stability with large values."""
        from utils.baseline import softmax_baseline

        x = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')
        result = softmax_baseline(x, dim=-1)

        # Should not have NaN or Inf
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        # Should sum to 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(1, device='cuda'))

    def test_softmax_sums_to_one(self, cuda_available):
        """Test that softmax output sums to 1."""
        from utils.baseline import softmax_baseline

        x = torch.randn(4, 128, device='cuda')
        result = softmax_baseline(x, dim=-1)

        row_sums = result.sum(dim=-1)
        expected_sums = torch.ones(4, device='cuda')

        assert torch.allclose(row_sums, expected_sums, atol=1e-5)


class TestAttentionScoresBaseline:
    """Test attention_scores_baseline correctness."""

    def test_shape_output(self, sample_attention_tensors):
        """Test that output shape is correct."""
        from utils.baseline import attention_scores_baseline

        q, k, v = sample_attention_tensors
        B, H, S, D = q.shape

        result = attention_scores_baseline(q, k)

        assert result.shape == (B, H, S, S)

    def test_scaling(self, cuda_available):
        """Test that scaling is applied correctly."""
        from utils.baseline import attention_scores_baseline

        B, H, S, D = 1, 1, 4, 64
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')

        # With default scale
        result_default = attention_scores_baseline(q, k)

        # Manual calculation with scale
        scale = 1.0 / math.sqrt(D)
        result_manual = torch.matmul(q, k.transpose(-2, -1)) * scale

        assert torch.allclose(result_default, result_manual, rtol=1e-4)

    def test_custom_scale(self, cuda_available):
        """Test with custom scale factor."""
        from utils.baseline import attention_scores_baseline

        B, H, S, D = 1, 1, 4, 64
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')
        custom_scale = 0.5

        result = attention_scores_baseline(q, k, scale=custom_scale)
        expected = torch.matmul(q, k.transpose(-2, -1)) * custom_scale

        assert torch.allclose(result, expected)


class TestCausalAttentionBaseline:
    """Test causal_attention_baseline correctness."""

    def test_output_shape(self, sample_attention_tensors):
        """Test that output shape matches input Q shape."""
        from utils.baseline import causal_attention_baseline

        q, k, v = sample_attention_tensors

        result = causal_attention_baseline(q, k, v)

        assert result.shape == q.shape

    def test_causal_property(self, cuda_available):
        """Test that future positions don't affect past positions."""
        from utils.baseline import causal_attention_baseline

        B, H, S, D = 1, 1, 8, 32
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')
        v = torch.randn(B, H, S, D, device='cuda')

        result1 = causal_attention_baseline(q, k, v)

        # Modify future positions in K and V
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, :, 4:, :] = torch.randn(B, H, 4, D, device='cuda')
        v_modified[:, :, 4:, :] = torch.randn(B, H, 4, D, device='cuda')

        result2 = causal_attention_baseline(q, k_modified, v_modified)

        # First 4 positions should be the same (causality)
        assert torch.allclose(result1[:, :, :4, :], result2[:, :, :4, :], rtol=1e-4)


class TestFlashAttentionBaseline:
    """Test flash_attention_baseline matches causal baseline."""

    def test_matches_causal_baseline(self, sample_attention_tensors):
        """Test that flash baseline matches regular causal baseline."""
        from utils.baseline import flash_attention_baseline, causal_attention_baseline

        q, k, v = sample_attention_tensors

        result_flash = flash_attention_baseline(q, k, v)
        result_causal = causal_attention_baseline(q, k, v)

        assert torch.allclose(result_flash, result_causal, rtol=1e-3, atol=1e-3)

    def test_output_shape(self, sample_attention_tensors):
        """Test that output shape matches input Q shape."""
        from utils.baseline import flash_attention_baseline

        q, k, v = sample_attention_tensors

        result = flash_attention_baseline(q, k, v)

        assert result.shape == q.shape

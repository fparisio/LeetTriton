"""
PyTorch baseline implementations for correctness and performance testing.
These are the "ground truth" that your Triton kernels must match.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import time


def vector_add_baseline(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Level 1: Simple vector addition."""
    return x + y


def matmul_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Level 2: Matrix multiplication."""
    return torch.matmul(a, b)


def softmax_baseline(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Level 3: Softmax along a dimension."""
    return F.softmax(x, dim=dim)


def attention_scores_baseline(
    q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None
) -> torch.Tensor:
    """Level 4: Compute attention scores (Q @ K^T) * scale."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    return torch.matmul(q, k.transpose(-2, -1)) * scale


def causal_mask_baseline(seq_len: int, device: torch.device) -> torch.Tensor:
    """Level 5: Create a causal mask (lower triangular)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def masked_softmax_baseline(
    scores: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Level 5: Softmax with optional causal masking."""
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return F.softmax(scores, dim=-1)


def causal_attention_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Full causal attention: softmax((Q @ K^T) * scale + mask) @ V

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        v: Value tensor of shape (batch, heads, seq_len, head_dim)
        scale: Optional scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Attention output of shape (batch, heads, seq_len, head_dim)
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    seq_len = q.shape[-2]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Handle NaN from all-masked positions (shouldn't happen with causal mask)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Apply attention to values
    output = torch.matmul(attn_weights, v)

    return output


def flash_attention_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Use PyTorch's native scaled_dot_product_attention as the Flash Attention baseline.
    This uses the most efficient implementation available (Flash Attention if supported).
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    )


class Benchmarker:
    """Utility class for benchmarking kernel performance."""

    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters

    def benchmark(self, fn, *args, **kwargs) -> Tuple[float, float]:
        """
        Benchmark a function and return (mean_ms, std_ms).
        """
        # Warmup
        for _ in range(self.warmup_iters):
            fn(*args, **kwargs)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.bench_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn(*args, **kwargs)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times = torch.tensor(times)
        return times.mean().item(), times.std().item()

    def compare(self, baseline_fn, test_fn, *args, **kwargs) -> dict:
        """
        Compare test function against baseline.
        Returns dict with timing info and speedup.
        """
        baseline_mean, baseline_std = self.benchmark(baseline_fn, *args, **kwargs)
        test_mean, test_std = self.benchmark(test_fn, *args, **kwargs)

        speedup = baseline_mean / test_mean

        return {
            "baseline_ms": baseline_mean,
            "baseline_std": baseline_std,
            "test_ms": test_mean,
            "test_std": test_std,
            "speedup": speedup,
        }

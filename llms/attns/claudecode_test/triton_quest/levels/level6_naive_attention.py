"""
=============================================================================
                    LEVEL 6: NAIVE CAUSAL ATTENTION
                    "Putting It All Together"
=============================================================================

QUEST OBJECTIVE:
    Implement complete causal attention using your previous kernels:
    output = softmax(Q @ K^T / sqrt(d) + causal_mask) @ V

WHAT YOU'LL LEARN:
    - Composing multiple kernels
    - Full attention pipeline
    - Memory allocation patterns
    - Performance bottlenecks of naive approach

THE FULL PICTURE:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

    Step 1: Compute scores = Q @ K^T / sqrt(d_k)    [Level 4]
    Step 2: Apply causal mask and softmax           [Level 5]
    Step 3: Compute output = weights @ V            [Level 2-ish]

MEMORY ANALYSIS (why this is "naive"):
    For Q, K, V of shape (B, H, S, D):
    - Scores matrix: (B, H, S, S) <- O(S^2) memory!
    - For S=4096, D=64, that's 4096*4096 = 16M floats per head
    - This is the bottleneck that Flash Attention solves

YOUR TASK:
    Combine your previous kernels to implement full causal attention.
    You can either:
    A) Call your existing kernels sequentially (easier)
    B) Write a new kernel that does score @ V after masked softmax (good practice)

    This level accepts either approach - the goal is to see it work end-to-end!

HINTS (reveal progressively if stuck):
    Hint 1: Use attention_scores() from level 4 to compute Q @ K^T * scale
    Hint 2: Use causal_softmax() from level 5 to apply mask and softmax
    Hint 3: The final matmul is: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
    Hint 4: You can use torch.matmul or write another Triton kernel for step 3
    Hint 5: Make sure to handle the reshaping correctly!

=============================================================================
"""

import torch
import triton
import triton.language as tl
import math

# Import your previous implementations
# from level4_attention_scores import attention_scores
# from level5_causal_mask import causal_softmax


def naive_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Naive causal attention implementation.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        v: Value tensor (batch, heads, seq_len, head_dim)
        scale: Scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Attention output (batch, heads, seq_len, head_dim)

    TODO: Implement this using your previous kernels!

    Steps:
    1. Compute attention scores: scores = (Q @ K^T) * scale
    2. Apply causal mask and softmax: weights = causal_softmax(scores)
    3. Compute output: output = weights @ V
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape

    batch, heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Compute attention scores
    # scores = ?  # Shape: (batch, heads, seq_len, seq_len)


    # Step 2: Apply causal mask and softmax
    # weights = ?  # Shape: (batch, heads, seq_len, seq_len)


    # Step 3: Compute output
    # output = ?  # Shape: (batch, heads, seq_len, head_dim)


    # return output

    pass  # Remove this when you implement


# =============================================================================
# BONUS: Write a fused weights @ V kernel for extra practice!
# =============================================================================

@triton.jit
def attention_output_kernel(
    weights_ptr, v_ptr, out_ptr,
    batch_heads, seq_len, head_dim,
    stride_wbh, stride_ws1, stride_ws2,
    stride_vbh, stride_vs, stride_vd,
    stride_obh, stride_os, stride_od,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    BONUS: Compute attention output = weights @ V

    weights: (B*H, S, S)
    V: (B*H, S, D)
    output: (B*H, S, D)

    This is essentially a batched matmul, similar to level 2.
    Each program computes a (BLOCK_S, BLOCK_D) tile of output.
    """
    # BONUS IMPLEMENTATION - Optional!
    pass


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 6: Naive Causal Attention")
    print("=" * 50)

    def reference_causal_attention(q, k, v, scale=None):
        """PyTorch reference implementation."""
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)  # Handle any NaN from all-masked rows
        return torch.matmul(weights, v)

    test_cases = [
        (1, 1, 64, 32),     # Small
        (2, 4, 128, 64),    # Multi-batch, multi-head
        (1, 8, 256, 64),    # Larger seq_len
        (4, 4, 100, 32),    # Non-power-of-2
    ]

    for batch, heads, seq_len, head_dim in test_cases:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

        expected = reference_causal_attention(q, k, v)

        try:
            actual = naive_causal_attention(q, k, v)

            if actual is None:
                print(f"  [SKIP] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Not implemented")
                continue

            if torch.allclose(expected, actual, rtol=1e-2, atol=1e-2):
                print(f"  [PASS] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                print(f"  [FAIL] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Max diff = {max_diff}")
        except Exception as e:
            print(f"  [ERROR] B={batch}, H={heads}, S={seq_len}, D={head_dim}: {e}")

    print("\nIf all tests pass, run: python quest.py check 6")
    print("\n" + "=" * 50)
    print("MEMORY WARNING:")
    print("Notice how naive attention allocates O(S^2) memory for scores.")
    print("For seq_len=4096, that's ~64MB per batch*head in float32!")
    print("Level 7+ will teach you how Flash Attention avoids this.")
    print("=" * 50)

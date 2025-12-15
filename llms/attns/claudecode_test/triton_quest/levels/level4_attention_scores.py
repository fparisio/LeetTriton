"""
=============================================================================
                    LEVEL 4: ATTENTION SCORES
                    "The First Step to Attention"
=============================================================================

QUEST OBJECTIVE:
    Compute attention scores: S = (Q @ K^T) * scale
    Where Q is (batch, heads, seq_len, head_dim)
          K is (batch, heads, seq_len, head_dim)
          S is (batch, heads, seq_len, seq_len)

WHAT YOU'LL LEARN:
    - Batched matrix operations
    - Handling 4D tensors in Triton
    - Scaling factor: 1/sqrt(head_dim)
    - Transposing K for the matmul

ATTENTION BASICS:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    This level focuses on the first part: Q @ K^T / sqrt(d_k)
    - Q (Query): What we're looking for
    - K (Key): What we match against
    - K^T: Transpose of K to make dimensions align
    - sqrt(d_k): Scaling to prevent softmax saturation

DIMENSION ANALYSIS:
    Q: (B, H, S, D)  - batch, heads, seq_len, head_dim
    K: (B, H, S, D)
    K^T: (B, H, D, S)
    Q @ K^T: (B, H, S, S)  <- This is our attention score matrix!

STRATEGY:
    For this level, you can "flatten" batch and heads dimensions.
    Process each (seq_len x head_dim) @ (head_dim x seq_len) separately.
    Total of B * H independent matmuls.

YOUR TASK:
    Modify your matmul kernel to handle the batched attention scores.

HINTS (reveal progressively if stuck):
    Hint 1: Flatten batch*heads into a single dimension for simpler indexing
    Hint 2: Each program handles one (row, col) tile within one batch*head
    Hint 3: Use a 3D grid: (num_row_tiles, num_col_tiles, batch*heads)
    Hint 4: Don't forget to transpose K! Either transpose before or adjust strides
    Hint 5: Multiply final result by scale = 1/sqrt(head_dim)

=============================================================================
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def attention_scores_kernel(
    # Pointers
    q_ptr, k_ptr, out_ptr,
    # Dimensions
    batch_heads,    # B * H (flattened)
    seq_len,        # S
    head_dim,       # D
    scale,          # 1/sqrt(D)
    # Strides for Q (B*H, S, D)
    stride_qbh, stride_qs, stride_qd,
    # Strides for K (B*H, S, D)
    stride_kbh, stride_ks, stride_kd,
    # Strides for output (B*H, S, S)
    stride_obh, stride_os1, stride_os2,
    # Block sizes
    BLOCK_S: tl.constexpr,  # Block size for seq_len
    BLOCK_D: tl.constexpr,  # Block size for head_dim
):
    """
    Compute attention scores: out = (Q @ K^T) * scale

    For each batch*head, compute S[i,j] = sum_d(Q[i,d] * K[j,d]) * scale

    TODO: Implement this kernel!

    Grid: (num_row_blocks, num_col_blocks, batch_heads)
    Each program computes a BLOCK_S x BLOCK_S tile of the output.
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get program IDs
    # pid_row = ?  (which row block of output)
    # pid_col = ?  (which col block of output)
    # pid_bh = ?   (which batch*head)


    # Step 2: Calculate row and column offsets for this tile
    # offs_row = ?  (shape: BLOCK_S)
    # offs_col = ?  (shape: BLOCK_S)


    # Step 3: Initialize accumulator
    # acc = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)


    # Step 4: Calculate base pointers for this batch*head
    # q_batch_ptr = q_ptr + pid_bh * stride_qbh
    # k_batch_ptr = k_ptr + pid_bh * stride_kbh


    # Step 5: Loop over head_dim in chunks of BLOCK_D
    # for d in range(0, head_dim, BLOCK_D):
    #     - Load Q tile: (BLOCK_S, BLOCK_D) from rows offs_row, cols d:d+BLOCK_D
    #     - Load K tile: (BLOCK_S, BLOCK_D) from rows offs_col, cols d:d+BLOCK_D
    #       NOTE: We want K^T, so we transpose after loading (or load transposed)
    #     - Compute: acc += Q_tile @ K_tile^T


    # Step 6: Apply scale
    # acc = acc * scale


    # Step 7: Store result
    # Calculate output pointers and store with masking


    pass  # Remove this when you implement


def attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Compute attention scores: (Q @ K^T) * scale

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        scale: Scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Attention scores (batch, heads, seq_len, seq_len)
    """
    assert q.is_cuda and k.is_cuda
    assert q.shape == k.shape

    batch, heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Reshape to (batch*heads, seq_len, head_dim)
    q_flat = q.reshape(batch * heads, seq_len, head_dim)
    k_flat = k.reshape(batch * heads, seq_len, head_dim)

    # Allocate output (batch*heads, seq_len, seq_len)
    out_flat = torch.empty(batch * heads, seq_len, seq_len, device=q.device, dtype=q.dtype)

    # Block sizes
    BLOCK_S = 64
    BLOCK_D = 64

    # Grid
    grid = (
        triton.cdiv(seq_len, BLOCK_S),  # row blocks
        triton.cdiv(seq_len, BLOCK_S),  # col blocks
        batch * heads,                   # batch*heads
    )

    attention_scores_kernel[grid](
        q_flat, k_flat, out_flat,
        batch * heads, seq_len, head_dim, scale,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    # Reshape back to (batch, heads, seq_len, seq_len)
    return out_flat.view(batch, heads, seq_len, seq_len)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 4: Attention Scores")
    print("=" * 50)

    test_cases = [
        (1, 1, 64, 64),     # Single batch, single head
        (2, 4, 128, 64),    # Multi-batch, multi-head
        (1, 8, 256, 32),    # More heads
        (4, 4, 100, 64),    # Non-power-of-2 seq_len
    ]

    for batch, heads, seq_len, head_dim in test_cases:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

        scale = 1.0 / math.sqrt(head_dim)
        expected = torch.matmul(q, k.transpose(-2, -1)) * scale

        try:
            actual = attention_scores(q, k, scale)

            if torch.allclose(expected, actual, rtol=1e-2, atol=1e-2):
                print(f"  [PASS] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                print(f"  [FAIL] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Max diff = {max_diff}")
        except Exception as e:
            print(f"  [ERROR] B={batch}, H={heads}, S={seq_len}, D={head_dim}: {e}")

    print("\nIf all tests pass, run: python quest.py check 4")

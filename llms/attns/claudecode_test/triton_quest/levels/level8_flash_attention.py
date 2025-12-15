"""
=============================================================================
                    LEVEL 8: FLASH ATTENTION
                    "The Final Boss"
=============================================================================

QUEST OBJECTIVE:
    Implement the full Flash Attention algorithm with 2D tiling.
    This is the state-of-the-art memory-efficient attention algorithm.

WHAT YOU'LL LEARN:
    - Full Flash Attention algorithm
    - 2D tiling (tiles over both Q and K/V)
    - Maximum GPU utilization
    - Production-level kernel optimization

FLASH ATTENTION OVERVIEW:
    Instead of processing one query row at a time (Level 7),
    Flash Attention processes BLOCKS of queries together.

    This improves parallelism and memory access patterns:
    - Each program handles a BLOCK_M x BLOCK_N tile of computation
    - Better cache utilization
    - More work per program = better GPU utilization

THE ALGORITHM:
    For each block of queries Q_block (BLOCK_M rows):
    1. Initialize: max_vec, sum_vec, acc_matrix (all per-row)
    2. For each K/V block:
       a. Load K_block, V_block
       b. Compute S_block = Q_block @ K_block^T * scale (BLOCK_M x BLOCK_N)
       c. Apply causal mask to S_block
       d. Online softmax update (per row):
          - new_max = max(old_max, rowmax(S_block))
          - correction = exp(old_max - new_max)
          - Scale previous acc and sum
          - exp_S = exp(S_block - new_max)
          - sum += rowsum(exp_S)
          - acc += exp_S @ V_block
    3. output = acc / sum

KEY INSIGHT - Per-Row State:
    Each row of Q has its own:
    - Running max (for numerical stability)
    - Running sum (for normalization)
    - Running accumulator (partial output)

    With BLOCK_M query rows, we track BLOCK_M maxes, sums, accumulators.

YOUR TASK:
    Implement Flash Attention with 2D tiling.
    This is challenging - take your time!

HINTS (reveal progressively if stuck):
    Hint 1: Grid is 2D: (num_Q_blocks, batch*heads)
    Hint 2: Each program handles BLOCK_M query rows
    Hint 3: max_vec, sum_vec have shape (BLOCK_M,) - one per query row
    Hint 4: acc has shape (BLOCK_M, head_dim) - output accumulator
    Hint 5: S_block = Q_block @ K_block^T has shape (BLOCK_M, BLOCK_N)
    Hint 6: Use tl.max(S_block, axis=1) for per-row max
    Hint 7: Causal mask: for Q row i, K col j is valid if (q_offset + i) >= (k_offset + j)

=============================================================================
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_heads, seq_len, head_dim, scale,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_obh, stride_os, stride_od,
    BLOCK_M: tl.constexpr,  # Block size for queries
    BLOCK_N: tl.constexpr,  # Block size for keys/values
    BLOCK_D: tl.constexpr,  # Block size for head_dim
):
    """
    Flash Attention kernel with full 2D tiling.

    Grid: (num_q_blocks, batch_heads)
    Each program processes BLOCK_M query rows.

    TODO: Implement the full Flash Attention algorithm!
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get program indices
    # q_block_idx = tl.program_id(0)  # Which block of queries
    # batch_head_idx = tl.program_id(1)


    # Step 2: Calculate query row offsets for this block
    # q_offs = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    # d_offs = tl.arange(0, BLOCK_D)


    # Step 3: Initialize per-row state
    # max_vec = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    # sum_vec = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)


    # Step 4: Calculate base pointers for this batch*head
    # q_base = q_ptr + batch_head_idx * stride_qbh
    # k_base = k_ptr + batch_head_idx * stride_kbh
    # v_base = v_ptr + batch_head_idx * stride_vbh


    # Step 5: Load Q block (BLOCK_M, BLOCK_D)
    # Create 2D indices for Q
    # q_ptrs = q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    # q_mask = (q_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)
    # q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)


    # Step 6: Iterate over K/V blocks
    # For causal attention, we only need to go up to the diagonal
    # max_k_block = (q_block_idx * BLOCK_M + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    # max_k_block = min(max_k_block, triton.cdiv(seq_len, BLOCK_N))

    # for k_block_idx in range(0, max_k_block):
    #     k_offs = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    #
    #     # Step 6a: Load K block (BLOCK_N, BLOCK_D)
    #     # k_ptrs = ...
    #     # k_block = tl.load(...)
    #
    #     # Step 6b: Compute S_block = Q_block @ K_block^T (BLOCK_M, BLOCK_N)
    #     # s_block = tl.dot(q_block, tl.trans(k_block)) * scale
    #
    #     # Step 6c: Apply causal mask
    #     # For each (q_row, k_col), valid if q_offs[q_row] >= k_offs[k_col]
    #     # causal_mask = q_offs[:, None] >= k_offs[None, :]
    #     # s_block = tl.where(causal_mask, s_block, float('-inf'))
    #
    #     # Also mask out-of-bounds
    #     # boundary_mask = (q_offs[:, None] < seq_len) & (k_offs[None, :] < seq_len)
    #     # s_block = tl.where(boundary_mask, s_block, float('-inf'))
    #
    #     # Step 6d: Online softmax update
    #     # row_max = tl.max(s_block, axis=1)  # Per-row max, shape (BLOCK_M,)
    #     # new_max = tl.maximum(max_vec, row_max)
    #
    #     # Correction factor for rescaling
    #     # correction = tl.exp(max_vec - new_max)
    #
    #     # Rescale previous sum and acc
    #     # sum_vec = sum_vec * correction
    #     # acc = acc * correction[:, None]
    #
    #     # Update max
    #     # max_vec = new_max
    #
    #     # Compute exp(s - max)
    #     # exp_s = tl.exp(s_block - max_vec[:, None])
    #
    #     # Update sum
    #     # sum_vec += tl.sum(exp_s, axis=1)
    #
    #     # Step 6e: Load V block (BLOCK_N, BLOCK_D)
    #     # v_ptrs = ...
    #     # v_block = tl.load(...)
    #
    #     # Accumulate: acc += exp_s @ V_block
    #     # acc += tl.dot(exp_s.to(v_block.dtype), v_block)


    # Step 7: Normalize output
    # output = acc / sum_vec[:, None]


    # Step 8: Store output (BLOCK_M, BLOCK_D)
    # out_base = out_ptr + batch_head_idx * stride_obh
    # out_ptrs = out_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    # out_mask = (q_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)
    # tl.store(out_ptrs, output, mask=out_mask)


    pass  # Remove this when you implement


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention implementation.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        v: Value tensor (batch, heads, seq_len, head_dim)
        scale: Scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Attention output (batch, heads, seq_len, head_dim)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape

    batch, heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Reshape to (batch*heads, seq_len, head_dim)
    q_flat = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    out_flat = torch.empty_like(q_flat)

    # Block sizes - tunable!
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Grid: (num_q_blocks, batch*heads)
    num_q_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (num_q_blocks, batch * heads)

    flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, out_flat,
        batch * heads, seq_len, head_dim, scale,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return out_flat.view(batch, heads, seq_len, head_dim)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 8: Flash Attention")
    print("=" * 50)
    print("THE FINAL BOSS")
    print("=" * 50)

    def reference_causal_attention(q, k, v, scale=None):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)
        return torch.matmul(weights, v)

    test_cases = [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (1, 8, 256, 64),
        (4, 4, 100, 32),
        (2, 8, 512, 64),   # Larger test
    ]

    all_passed = True
    for batch, heads, seq_len, head_dim in test_cases:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

        expected = reference_causal_attention(q, k, v)

        try:
            actual = flash_attention(q, k, v)

            if actual is None:
                print(f"  [SKIP] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Not implemented")
                all_passed = False
                continue

            if torch.allclose(expected, actual, rtol=1e-2, atol=1e-2):
                print(f"  [PASS] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                mean_diff = (expected - actual).abs().mean().item()
                print(f"  [FAIL] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Max={max_diff:.4f}, Mean={mean_diff:.4f}")
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] B={batch}, H={heads}, S={seq_len}, D={head_dim}: {e}")
            all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("""
    ===============================================
    |                                             |
    |     CONGRATULATIONS, TRITON MASTER!         |
    |                                             |
    |        You have conquered all 8 levels      |
    |        and implemented Flash Attention      |
    |        from scratch!                        |
    |                                             |
    |              *    *    *                    |
    |           *     *     *                     |
    |        *    VICTORY    *                    |
    |           *     *     *                     |
    |              *    *    *                    |
    |                                             |
    ===============================================
        """)
        print("=" * 60)
        print("\nNext steps to continue your journey:")
        print("  1. Add autotuning with @triton.autotune")
        print("  2. Implement backward pass for training")
        print("  3. Add support for different head dimensions")
        print("  4. Profile with nsys and optimize further")
        print("  5. Compare performance against PyTorch's SDPA")

    print("\nIf all tests pass, run: python quest.py check 8")

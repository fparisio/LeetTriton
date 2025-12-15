"""
=============================================================================
                    LEVEL 7: FUSED ATTENTION
                    "The First Optimization"
=============================================================================

QUEST OBJECTIVE:
    Fuse attention into a single kernel: avoid materializing the full S x S matrix.
    For each output position, compute attention on-the-fly.

WHAT YOU'LL LEARN:
    - Kernel fusion benefits
    - Online/streaming computations
    - Tiling strategies for attention
    - Memory bandwidth optimization

THE MEMORY PROBLEM:
    Naive attention stores full (S x S) scores matrix.
    For S=4096: 4096 * 4096 * 4 bytes = 64MB per batch*head!

    With fusion, we never store the full matrix.
    We compute tiles of scores, apply softmax, multiply by V tile,
    and accumulate - all before moving to the next tile.

TILING STRATEGY:
    For each output row (query position):
    1. Iterate over K/V in tiles (blocks of key positions)
    2. Compute partial scores for this tile
    3. Apply causal mask (if needed for this tile)
    4. Keep track of running max and sum for online softmax
    5. Accumulate weighted V values
    6. Rescale at the end

ONLINE SOFTMAX (Key Insight):
    Instead of computing softmax(x) = exp(x_i - max(x)) / sum(exp(x - max(x)))
    over all elements at once, we can compute it incrementally!

    For two blocks A and B:
    - Compute max_A, sum_A for block A
    - Compute max_B, sum_B for block B
    - New max = max(max_A, max_B)
    - Rescale: sum_A' = sum_A * exp(max_A - new_max)
    - Rescale: sum_B' = sum_B * exp(max_B - new_max)
    - Total sum = sum_A' + sum_B'

YOUR TASK:
    Implement fused attention with online softmax.
    Process one row of output at a time, iterating over K/V tiles.

HINTS (reveal progressively if stuck):
    Hint 1: Outer loop over query rows, inner loop over K/V tiles
    Hint 2: For each query row, maintain: running_max, running_sum, acc_output
    Hint 3: When processing new tile: compute scores, find tile_max
    Hint 4: Update running_max, rescale previous acc and sum
    Hint 5: Compute exp(scores - running_max), add to running_sum
    Hint 6: Accumulate: acc += exp_scores @ V_tile (after rescaling)
    Hint 7: Final output = acc / running_sum

=============================================================================
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_heads, seq_len, head_dim, scale,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_obh, stride_os, stride_od,
    BLOCK_S: tl.constexpr,  # Block size for seq_len (K/V tiles)
    BLOCK_D: tl.constexpr,  # Block size for head_dim
):
    """
    Fused causal attention kernel with online softmax.

    Each program processes ONE query row and produces ONE output row.
    It iterates over all K/V positions in tiles, using online softmax.

    Grid: (seq_len, batch_heads)

    TODO: Implement this kernel!

    Algorithm for each query row q_i:
    1. Initialize: max_so_far = -inf, sum_so_far = 0, acc = zeros(head_dim)
    2. For each K/V tile j = 0, BLOCK_S, 2*BLOCK_S, ...:
       a. Load K tile: k[j:j+BLOCK_S, :]
       b. Compute scores: s = q_i @ k^T * scale  (shape: BLOCK_S)
       c. Apply causal mask: s[k] = -inf where j+k > i
       d. Find tile_max = max(s)
       e. new_max = max(max_so_far, tile_max)
       f. Rescale: sum_so_far *= exp(max_so_far - new_max)
       g. Rescale: acc *= exp(max_so_far - new_max)
       h. Update max_so_far = new_max
       i. exp_s = exp(s - max_so_far)
       j. sum_so_far += sum(exp_s)
       k. Load V tile: v[j:j+BLOCK_S, :]
       l. acc += exp_s @ v  (weighted sum)
    3. Output = acc / sum_so_far
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get program indices
    # query_idx = tl.program_id(0)  # Which query row
    # batch_head_idx = tl.program_id(1)  # Which batch*head


    # Step 2: Initialize for online softmax
    # max_so_far = float('-inf')  # Running maximum
    # sum_so_far = 0.0            # Running sum of exp(scores - max)
    # acc = tl.zeros((BLOCK_D,), dtype=tl.float32)  # Accumulated output


    # Step 3: Load query vector (this row's query)
    # q_row_ptr = q_ptr + batch_head_idx * stride_qbh + query_idx * stride_qs
    # d_offs = tl.arange(0, BLOCK_D)
    # q = tl.load(q_row_ptr + d_offs * stride_qd, mask=d_offs < head_dim)


    # Step 4: Iterate over K/V tiles
    # for tile_start in range(0, seq_len, BLOCK_S):
    #     # Calculate which key positions this tile covers
    #     key_offs = tile_start + tl.arange(0, BLOCK_S)
    #
    #     # Apply causal mask: only attend to positions <= query_idx
    #     causal_mask = key_offs <= query_idx
    #     valid_mask = (key_offs < seq_len) & causal_mask
    #
    #     # Skip entirely if all positions are masked (optimization)
    #     if tile_start > query_idx:
    #         break  # All future positions - skip rest
    #
    #     # Load K tile: (BLOCK_S, head_dim) -> we need (head_dim, BLOCK_S) for matmul
    #     # Actually, we compute dot product differently in Triton for 1D query
    #
    #     # Compute scores for this tile
    #     # scores = sum_d(q[d] * k[key, d]) for each key in tile
    #
    #     # Apply causal mask
    #     # scores = tl.where(causal_mask, scores, float('-inf'))
    #
    #     # Online softmax update
    #     # tile_max = tl.max(scores)
    #     # new_max = tl.maximum(max_so_far, tile_max)
    #
    #     # Rescale previous accumulations
    #     # correction = tl.exp(max_so_far - new_max)
    #     # sum_so_far = sum_so_far * correction
    #     # acc = acc * correction
    #     # max_so_far = new_max
    #
    #     # Compute exp scores with new max
    #     # exp_scores = tl.exp(scores - max_so_far)
    #     # exp_scores = tl.where(valid_mask, exp_scores, 0.0)
    #
    #     # Update running sum
    #     # sum_so_far += tl.sum(exp_scores)
    #
    #     # Load V tile and accumulate
    #     # For each key position k, add exp_scores[k] * V[k, :]
    #     # acc += exp_scores @ V_tile


    # Step 5: Normalize by sum
    # output = acc / sum_so_far


    # Step 6: Store output
    # out_row_ptr = out_ptr + batch_head_idx * stride_obh + query_idx * stride_os
    # tl.store(out_row_ptr + d_offs * stride_od, output, mask=d_offs < head_dim)


    pass  # Remove this when you implement


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Fused causal attention implementation.

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
    q_flat = q.reshape(batch * heads, seq_len, head_dim)
    k_flat = k.reshape(batch * heads, seq_len, head_dim)
    v_flat = v.reshape(batch * heads, seq_len, head_dim)
    out_flat = torch.empty_like(q_flat)

    # Block sizes
    BLOCK_S = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Grid: one program per (query_row, batch*head)
    grid = (seq_len, batch * heads)

    fused_attention_kernel[grid](
        q_flat, k_flat, v_flat, out_flat,
        batch * heads, seq_len, head_dim, scale,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return out_flat.view(batch, heads, seq_len, head_dim)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 7: Fused Attention")
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
    ]

    for batch, heads, seq_len, head_dim in test_cases:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

        expected = reference_causal_attention(q, k, v)

        try:
            actual = fused_attention(q, k, v)

            if actual is None:
                print(f"  [SKIP] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Not implemented")
                continue

            if torch.allclose(expected, actual, rtol=1e-2, atol=1e-2):
                print(f"  [PASS] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                mean_diff = (expected - actual).abs().mean().item()
                print(f"  [FAIL] B={batch}, H={heads}, S={seq_len}, D={head_dim}: Max={max_diff:.4f}, Mean={mean_diff:.4f}")
        except Exception as e:
            print(f"  [ERROR] B={batch}, H={heads}, S={seq_len}, D={head_dim}: {e}")

    print("\n" + "=" * 50)
    print("MEMORY COMPARISON:")
    print("Naive (Level 6): Stores full S x S matrix")
    print("Fused (Level 7): Only stores O(S * D) at a time")
    print("For S=4096, D=64: 64MB vs ~1MB per batch*head!")
    print("=" * 50)
    print("\nIf all tests pass, run: python quest.py check 7")

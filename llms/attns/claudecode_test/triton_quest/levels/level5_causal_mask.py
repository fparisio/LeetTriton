"""
=============================================================================
                    LEVEL 5: CAUSAL MASKING
                    "Preventing Time Travel"
=============================================================================

QUEST OBJECTIVE:
    Apply causal masking to attention scores, then softmax.
    output = softmax(scores + causal_mask)
    Where causal_mask is -inf for positions j > i (future positions)

WHAT YOU'LL LEARN:
    - Causal (autoregressive) masking
    - Combining operations: mask + softmax
    - Efficient mask generation on-the-fly
    - Handling -inf in computations

CAUSAL ATTENTION:
    In autoregressive models (like GPT), position i can only attend to
    positions 0, 1, ..., i (past and present, not future).

    The causal mask looks like (for seq_len=4):
        [0,   -inf, -inf, -inf]
        [0,    0,   -inf, -inf]
        [0,    0,    0,   -inf]
        [0,    0,    0,    0  ]

    When added to scores and passed through softmax, -inf becomes 0.

EFFICIENCY INSIGHT:
    Instead of creating a full mask tensor, we can generate the mask
    on-the-fly by comparing row and column indices:
    - If col_idx <= row_idx: valid (mask = 0)
    - If col_idx > row_idx: invalid (mask = -inf)

YOUR TASK:
    Implement causal masked softmax in a single kernel.
    This combines levels 3 (softmax) and adds causal masking.

HINTS (reveal progressively if stuck):
    Hint 1: Process row-by-row (each program handles one row)
    Hint 2: For row i, valid columns are 0, 1, ..., i
    Hint 3: Generate mask: mask = col_offs <= row_idx
    Hint 4: Apply mask before softmax: use tl.where(mask, score, -inf)
    Hint 5: Rest is same as softmax: max, exp, sum, divide

=============================================================================
"""

import torch
import triton
import triton.language as tl


@triton.jit
def causal_softmax_kernel(
    input_ptr,
    output_ptr,
    seq_len,          # Number of columns (and rows, it's square)
    stride_batch,     # Stride to next batch*head
    stride_row,       # Stride to next row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply causal mask and softmax to attention scores.

    Input: (batch*heads, seq_len, seq_len) - attention scores
    Output: (batch*heads, seq_len, seq_len) - attention weights

    TODO: Implement causal masked softmax!

    Each program handles one row of one batch*head.
    For row i:
    1. Load the row of scores
    2. Apply causal mask: set scores[j] = -inf where j > i
    3. Apply numerically stable softmax
    4. Store result
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get batch*head index and row index
    # We use a 2D grid: (rows, batch*heads)
    # row_idx = ?
    # batch_idx = ?


    # Step 2: Calculate pointer to start of this row
    # row_ptr = input_ptr + batch_idx * stride_batch + row_idx * stride_row


    # Step 3: Create column offsets
    # col_offs = tl.arange(0, BLOCK_SIZE)


    # Step 4: Create causal mask
    # valid positions: col_offs <= row_idx
    # causal_mask = ?


    # Step 5: Create boundary mask (for seq_len not divisible by BLOCK_SIZE)
    # boundary_mask = col_offs < seq_len


    # Step 6: Load scores
    # Use combined mask for loading, use -inf for masked positions
    # scores = tl.load(?, mask=?, other=float('-inf'))


    # Step 7: Apply causal mask
    # Set future positions to -inf
    # scores = tl.where(causal_mask, scores, float('-inf'))


    # Step 8: Numerically stable softmax
    # max_val = tl.max(scores, axis=0)
    # exp_scores = tl.exp(scores - max_val)
    # sum_exp = tl.sum(exp_scores, axis=0)
    # output = exp_scores / sum_exp


    # Step 9: Handle edge case: row 0 attends only to itself
    # If all positions are -inf (shouldn't happen with causal), output 0


    # Step 10: Store result
    # out_ptr = output_ptr + batch_idx * stride_batch + row_idx * stride_row
    # tl.store(?, ?, mask=?)


    pass  # Remove this when you implement


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    """
    Apply causal masking and softmax to attention scores.

    Args:
        scores: Attention scores (batch, heads, seq_len, seq_len)

    Returns:
        Attention weights (batch, heads, seq_len, seq_len)
    """
    assert scores.is_cuda

    batch, heads, seq_len, _ = scores.shape
    assert scores.shape[-1] == seq_len, "Scores must be square in last two dims"

    # Reshape to (batch*heads, seq_len, seq_len)
    scores_flat = scores.reshape(batch * heads, seq_len, seq_len)
    output_flat = torch.empty_like(scores_flat)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)

    # Grid: (seq_len rows, batch*heads)
    grid = (seq_len, batch * heads)

    causal_softmax_kernel[grid](
        scores_flat, output_flat,
        seq_len,
        scores_flat.stride(0),
        scores_flat.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_flat.view(batch, heads, seq_len, seq_len)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 5: Causal Masked Softmax")
    print("=" * 50)

    def reference_causal_softmax(scores):
        """PyTorch reference implementation."""
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(scores, dim=-1)

    test_cases = [
        (1, 1, 64),     # Small
        (2, 4, 128),    # Multi-batch, multi-head
        (1, 8, 256),    # Larger seq_len
        (4, 4, 100),    # Non-power-of-2
    ]

    for batch, heads, seq_len in test_cases:
        scores = torch.randn(batch, heads, seq_len, seq_len, device='cuda')

        expected = reference_causal_softmax(scores)

        try:
            actual = causal_softmax(scores)

            # Check for NaN
            if torch.isnan(actual).any():
                print(f"  [FAIL] B={batch}, H={heads}, S={seq_len}: Contains NaN!")
                continue

            if torch.allclose(expected, actual, rtol=1e-3, atol=1e-3):
                print(f"  [PASS] B={batch}, H={heads}, S={seq_len}: Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                print(f"  [FAIL] B={batch}, H={heads}, S={seq_len}: Max diff = {max_diff}")
        except Exception as e:
            print(f"  [ERROR] B={batch}, H={heads}, S={seq_len}: {e}")

    # Verify causal property
    print("\n  Verifying causal property...")
    scores = torch.randn(1, 1, 4, 4, device='cuda')
    try:
        weights = causal_softmax(scores)
        upper_tri = torch.triu(weights[0, 0], diagonal=1)
        if upper_tri.abs().max() < 1e-6:
            print("  [PASS] Causal mask correctly zeros future positions!")
        else:
            print(f"  [FAIL] Future positions not zero: {upper_tri}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    print("\nIf all tests pass, run: python quest.py check 5")

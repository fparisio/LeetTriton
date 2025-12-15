"""
=============================================================================
                    LEVEL 3: SOFTMAX
                    "The Heart of Attention"
=============================================================================

QUEST OBJECTIVE:
    Implement row-wise softmax: output[i,j] = exp(x[i,j] - max_i) / sum_j(exp(x[i,j] - max_i))

WHAT YOU'LL LEARN:
    - Row-wise operations (each program handles one row)
    - Numerical stability (the max trick)
    - Reduction operations: tl.max(), tl.sum()
    - Why naive softmax fails and how to fix it

NUMERICAL STABILITY:
    Naive softmax: softmax(x) = exp(x) / sum(exp(x))
    Problem: exp(x) overflows for large x (e.g., x > 88 for float32)

    Stable softmax:
    1. Find max_val = max(x)
    2. Compute exp(x - max_val)  # Now all values <= 1, no overflow!
    3. Divide by sum(exp(x - max_val))

    This is mathematically equivalent because:
    exp(x - max) / sum(exp(x - max)) = exp(x) / sum(exp(x))

TRITON CONCEPTS:
    1. REDUCTIONS: tl.max(x, axis=0) finds max along axis
       tl.sum(x, axis=0) sums along axis

    2. ROW-WISE PROCESSING: Each program handles one complete row.
       This is different from matmul where programs handle tiles.

    3. MULTIPLE PASSES: Softmax requires multiple passes over data:
       - Pass 1: Find max
       - Pass 2: Compute exp(x - max) and sum
       - Pass 3: Divide by sum (can combine with pass 2)

YOUR TASK:
    Implement numerically stable row-wise softmax.

HINTS (reveal progressively if stuck):
    Hint 1: Each program processes one row: row_idx = tl.program_id(0)
    Hint 2: Load the entire row (with masking): x = tl.load(row_ptr + offs, mask=offs < n_cols)
    Hint 3: Find row max: max_val = tl.max(x, axis=0) - returns scalar!
    Hint 4: Subtract max and exp: numerator = tl.exp(x - max_val)
    Hint 5: Sum the numerators: denominator = tl.sum(numerator, axis=0)
    Hint 6: Divide: output = numerator / denominator
    Hint 7: Store with masking

=============================================================================
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,           # Number of columns (row length)
    input_row_stride, # Stride between rows in input
    output_row_stride,# Stride between rows in output
    BLOCK_SIZE: tl.constexpr,  # Must be >= n_cols
):
    """
    Triton kernel for row-wise softmax.

    TODO: Implement numerically stable softmax!

    Each program handles ONE row. The algorithm:
    1. Load the row
    2. Find the maximum value (for numerical stability)
    3. Subtract max and compute exp
    4. Sum the exp values
    5. Divide each exp value by the sum
    6. Store the result
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get row index (which row is this program processing?)
    # row_idx = ?


    # Step 2: Calculate pointer to start of this row
    # row_start_ptr = ?


    # Step 3: Create column offsets and load mask
    # col_offs = ?  (0, 1, 2, ..., BLOCK_SIZE-1)
    # mask = ?      (which columns are valid?)


    # Step 4: Load the row
    # row = tl.load(?, mask=?, other=float('-inf'))
    # Note: Use float('-inf') for out-of-bounds so they don't affect max


    # Step 5: Find maximum for numerical stability
    # max_val = ?


    # Step 6: Compute numerator: exp(x - max)
    # numerator = ?


    # Step 7: Compute denominator: sum of numerators
    # denominator = ?


    # Step 8: Compute softmax output
    # output = ?


    # Step 9: Store result
    # tl.store(?, ?, mask=?)


    pass  # Remove this when you implement the kernel


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function that launches the Triton softmax kernel.
    Computes softmax along the last dimension.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"

    # Handle arbitrary dimensions by reshaping to 2D
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])

    n_rows, n_cols = x_2d.shape

    # BLOCK_SIZE must be at least n_cols (one row per program)
    # Round up to nearest power of 2 for efficiency
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output
    output = torch.empty_like(x_2d)

    # Grid: one program per row
    grid = (n_rows,)

    # Launch kernel
    softmax_kernel[grid](
        x_2d, output,
        n_cols,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape back to original shape
    return output.view(original_shape)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 3: Softmax")
    print("=" * 50)

    # Test cases
    test_cases = [
        (4, 128),     # Small
        (32, 256),    # Medium
        (64, 1024),   # Larger
        (8, 100),     # Non-power-of-2 cols
    ]

    for n_rows, n_cols in test_cases:
        x = torch.randn(n_rows, n_cols, device='cuda')

        expected = torch.softmax(x, dim=-1)
        try:
            actual = softmax(x)

            if torch.allclose(expected, actual, rtol=1e-4, atol=1e-4):
                print(f"  [PASS] ({n_rows}x{n_cols}): Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                print(f"  [FAIL] ({n_rows}x{n_cols}): Max diff = {max_diff}")
        except Exception as e:
            print(f"  [ERROR] ({n_rows}x{n_cols}): {e}")

    # Test numerical stability with large values
    print("\n  Testing numerical stability...")
    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')
    expected = torch.softmax(x_large, dim=-1)
    try:
        actual = softmax(x_large)
        if torch.allclose(expected, actual, rtol=1e-4, atol=1e-4):
            print(f"  [PASS] Large values: Numerically stable!")
        else:
            print(f"  [FAIL] Large values: Not numerically stable")
    except Exception as e:
        print(f"  [ERROR] Large values: {e}")

    print("\nIf all tests pass, run: python quest.py check 3")

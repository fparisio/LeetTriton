"""
=============================================================================
                    LEVEL 2: MATRIX MULTIPLICATION
                    "The Gateway to GPU Mastery"
=============================================================================

QUEST OBJECTIVE:
    Implement matrix multiplication: C = A @ B
    Where A is (M, K) and B is (K, N), producing C of shape (M, N)

WHAT YOU'LL LEARN:
    - 2D grid of programs (blocks)
    - Tiled matrix multiplication
    - Accumulator patterns
    - Memory access patterns and coalescing basics
    - Loop over K dimension

TRITON CONCEPTS:
    1. 2D GRIDS: You'll launch a 2D grid of programs.
       - program_id(0) = which row-block of C
       - program_id(1) = which col-block of C

    2. TILING: Each program computes a BLOCK_M x BLOCK_N tile of C.
       To do this, it needs to load tiles from A and B and accumulate.

    3. K-DIMENSION LOOP: Since we tile, we iterate over K in chunks
       of BLOCK_K, loading tiles and accumulating partial results.

    4. tl.dot(): Triton's matrix multiply for tiles. Takes two 2D
       tensors and returns their matrix product.

MEMORY LAYOUT:
    A is (M, K) - row major: A[i,j] at offset i*K + j
    B is (K, N) - row major: B[i,j] at offset i*N + j
    C is (M, N) - row major: C[i,j] at offset i*N + j

YOUR TASK:
    Fill in the kernel below. Focus on understanding tiling!

HINTS (reveal progressively if stuck):
    Hint 1: Get 2D program IDs: pid_m = tl.program_id(0), pid_n = tl.program_id(1)
    Hint 2: Create offset ranges: offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    Hint 3: Initialize accumulator: acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Hint 4: Loop over K: for k in range(0, K, BLOCK_K)
    Hint 5: Load A tile: a = tl.load(a_ptrs, mask=...), where a_ptrs points to current tile
    Hint 6: Load B tile similarly, use tl.dot(a, b) to multiply, add to acc
    Hint 7: After loop, store acc to C with proper masking

=============================================================================
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (how many elements to skip for next row)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A @ B.

    TODO: Implement this kernel!

    The algorithm:
    1. Each program computes one BLOCK_M x BLOCK_N tile of C
    2. To compute this tile, we iterate over K in chunks of BLOCK_K
    3. In each iteration, load a BLOCK_M x BLOCK_K tile of A
       and a BLOCK_K x BLOCK_N tile of B
    4. Multiply them and accumulate into result
    5. After all K iterations, store the tile to C
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get program IDs (which tile of C are we computing?)
    # pid_m = ?  (row block index)
    # pid_n = ?  (col block index)


    # Step 2: Calculate starting row and column indices for this tile
    # offs_m = ?  (row indices within this tile, shape: BLOCK_M)
    # offs_n = ?  (col indices within this tile, shape: BLOCK_N)


    # Step 3: Calculate pointers to first tiles of A and B
    # A tile starts at row offs_m, col 0
    # B tile starts at row 0, col offs_n
    # a_ptrs shape: (BLOCK_M, BLOCK_K)
    # b_ptrs shape: (BLOCK_K, BLOCK_N)


    # Step 4: Initialize accumulator
    # acc = ?  (shape: BLOCK_M x BLOCK_N, dtype: float32)


    # Step 5: Loop over K dimension in chunks of BLOCK_K
    # for k in range(0, K, BLOCK_K):
    #     - Load tile of A (with masking for boundaries)
    #     - Load tile of B (with masking for boundaries)
    #     - Multiply using tl.dot() and accumulate
    #     - Advance pointers by BLOCK_K


    # Step 6: Store result to C
    # Calculate C pointers and masks, then store acc


    pass  # Remove this when you implement the kernel


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function that launches the Triton matmul kernel.
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.shape[1] == b.shape[0], f"Inner dimensions must match: {a.shape[1]} vs {b.shape[0]}"

    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Block sizes - these are tunable!
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid: how many programs to launch
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # stride_am, stride_ak
        b.stride(0), b.stride(1),  # stride_bk, stride_bn
        c.stride(0), c.stride(1),  # stride_cm, stride_cn
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 2: Matrix Multiplication")
    print("=" * 50)

    # Test with different sizes
    test_cases = [
        (64, 64, 64),    # Small, perfect alignment
        (128, 256, 128), # Medium
        (100, 100, 100), # Non-power-of-2
        (512, 512, 512), # Larger
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device='cuda')
        b = torch.randn(K, N, device='cuda')

        expected = torch.matmul(a, b)
        try:
            actual = matmul(a, b)

            if torch.allclose(expected, actual, rtol=1e-2, atol=1e-2):
                print(f"  [PASS] ({M}x{K}) @ ({K}x{N}): Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                print(f"  [FAIL] ({M}x{K}) @ ({K}x{N}): Max diff = {max_diff}")
        except Exception as e:
            print(f"  [ERROR] ({M}x{K}) @ ({K}x{N}): {e}")

    print("\nIf all tests pass, run: python quest.py check 2")

"""
=============================================================================
                    LEVEL 1: VECTOR ADDITION
                    "The Hello World of Triton"
=============================================================================

QUEST OBJECTIVE:
    Implement vector addition: output[i] = x[i] + y[i]

WHAT YOU'LL LEARN:
    - Triton kernel basics: @triton.jit decorator
    - Program IDs: tl.program_id()
    - Memory loads and stores: tl.load() and tl.store()
    - Block-based parallelism: BLOCK_SIZE concept
    - Pointer arithmetic in Triton
    - Masking for boundary conditions

TRITON CONCEPTS:
    1. PROGRAMS: In Triton, work is divided into "programs". Each program
       processes a chunk (block) of data. Think of it like CUDA thread blocks.

    2. PROGRAM_ID: tl.program_id(axis) gives you which program you are.
       For 1D work, use axis=0.

    3. BLOCK_SIZE: How many elements each program processes. Typically
       a power of 2 (64, 128, 256, 512, 1024).

    4. POINTER ARITHMETIC: In Triton, you work with pointers + offsets.
       If ptr points to array start, ptr + offsets accesses elements.

    5. MASKING: When your data size isn't divisible by BLOCK_SIZE,
       you need masks to avoid out-of-bounds access.

YOUR TASK:
    Fill in the kernel below. The wrapper function is provided.

HINTS (reveal progressively if stuck):
    Hint 1: Calculate which block you are with tl.program_id(0)
    Hint 2: Create offsets using tl.arange(0, BLOCK_SIZE)
    Hint 3: Your global offset is: block_id * BLOCK_SIZE + local_offsets
    Hint 4: Create a mask: offsets < n (where n is total elements)
    Hint 5: Load x and y using tl.load(ptr + offsets, mask=mask)
    Hint 6: Store result using tl.store(ptr + offsets, result, mask=mask)

=============================================================================
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to output vector
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per program (compile-time constant)
):
    """
    Triton kernel for vector addition.

    TODO: Implement this kernel!

    Steps:
    1. Get the program ID (which block are we?)
    2. Calculate the starting offset for this block
    3. Create a range of offsets for elements in this block
    4. Create a mask for boundary checking
    5. Load x and y values
    6. Compute sum
    7. Store result
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Step 1: Get the program ID


    # Step 2-3: Calculate offsets for this block


    # Step 4: Create boundary mask


    # Step 5: Load data


    # Step 6: Compute


    # Step 7: Store result


    pass  # Remove this when you implement the kernel


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function that launches the Triton kernel.
    This is provided for you - focus on the kernel!
    """
    # Validate inputs
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Input shapes must match"

    # Allocate output
    output = torch.empty_like(x)

    # Calculate grid size
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    # Grid: how many programs to launch
    # We need ceil(n_elements / BLOCK_SIZE) programs
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    vector_add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# SELF-TEST (run this file directly to test your implementation)
# =============================================================================
if __name__ == "__main__":
    print("Testing Level 1: Vector Addition")
    print("=" * 50)

    # Test with different sizes
    test_sizes = [1024, 1000, 8192, 65536]

    for size in test_sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')

        expected = x + y
        try:
            actual = vector_add(x, y)

            if torch.allclose(expected, actual, rtol=1e-5, atol=1e-5):
                print(f"  [PASS] Size {size}: Correct!")
            else:
                max_diff = (expected - actual).abs().max().item()
                print(f"  [FAIL] Size {size}: Max diff = {max_diff}")
        except Exception as e:
            print(f"  [ERROR] Size {size}: {e}")

    print("\nIf all tests pass, run: python quest.py check 1")

# Level implementations for the Triton Quest

LEVELS = {
    1: {
        "name": "Vector Addition",
        "subtitle": "The Hello World of Triton",
        "module": "level1_vector_add",
        "description": "Learn Triton basics: program IDs, blocks, loads, and stores.",
    },
    2: {
        "name": "Matrix Multiplication",
        "subtitle": "The Gateway to GPU Mastery",
        "module": "level2_matmul",
        "description": "Master 2D tiling and the fundamental building block of deep learning.",
    },
    3: {
        "name": "Softmax",
        "subtitle": "The Heart of Attention",
        "module": "level3_softmax",
        "description": "Learn row-wise operations and numerical stability tricks.",
    },
    4: {
        "name": "Attention Scores",
        "subtitle": "The First Step to Attention",
        "module": "level4_attention_scores",
        "description": "Compute Q @ K^T with batched operations.",
    },
    5: {
        "name": "Causal Masking",
        "subtitle": "Preventing Time Travel",
        "module": "level5_causal_mask",
        "description": "Apply causal masks for autoregressive attention.",
    },
    6: {
        "name": "Naive Causal Attention",
        "subtitle": "Putting It All Together",
        "module": "level6_naive_attention",
        "description": "Combine your kernels into full attention.",
    },
    7: {
        "name": "Fused Attention",
        "subtitle": "The First Optimization",
        "module": "level7_fused_attention",
        "description": "Learn online softmax and kernel fusion.",
    },
    8: {
        "name": "Flash Attention",
        "subtitle": "The Final Boss",
        "module": "level8_flash_attention",
        "description": "Master the state-of-the-art attention algorithm.",
    },
}

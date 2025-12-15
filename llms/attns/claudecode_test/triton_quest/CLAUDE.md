# Triton Attention Quest

This is a gamified learning system for teaching Triton GPU programming by implementing Flash Attention from scratch.

## Project Structure

- `quest.py` - Main CLI for progress tracking, testing, hints
- `levels/` - 8 progressive challenge levels (level1 through level8)
- `utils/baseline.py` - PyTorch reference implementations
- `utils/testing.py` - Test harness for validation
- `tests/` - pytest unit test suite
- `.claude/skills/triton-tutor.md` - Comprehensive tutor knowledge base

## Available Commands

- `/tutor [question]` - Get Socratic tutoring help with your implementation

## When Helping Students

If a student asks for help with their Triton implementation:

1. Read `.claude/skills/triton-tutor.md` for detailed level knowledge and teaching guidelines
2. Be a Socratic tutor - guide with questions, don't give solutions
3. Read their actual implementation file to provide specific feedback
4. Focus on building understanding, not just fixing code

## Levels Overview

1. **Vector Addition** - Triton basics: program IDs, blocks, masking
2. **Matrix Multiplication** - 2D tiling, accumulators, dot products
3. **Softmax** - Row-wise ops, numerical stability
4. **Attention Scores** - Batched Q @ K^T, scaling
5. **Causal Masking** - On-the-fly mask generation
6. **Naive Attention** - Kernel composition
7. **Fused Attention** - Online softmax, streaming
8. **Flash Attention** - Full 2D tiled algorithm

# Triton Attention Quest

A gamified learning journey to master Triton by implementing Flash Attention from scratch.

## Quick Start

```bash
cd triton_quest
python quest.py          # See progress and next steps
python quest.py start 1  # Begin Level 1
```

## Commands

| Command | Description |
|---------|-------------|
| `python quest.py` | Show progress overview |
| `python quest.py start <N>` | Get instructions for level N |
| `python quest.py check <N>` | Verify your implementation |
| `python quest.py hint <N>` | Get a progressive hint |
| `python quest.py benchmark` | Compare performance vs PyTorch |
| `python quest.py reset` | Start over |

## The 8 Levels

1. **Vector Addition** - Triton basics: program IDs, blocks, loads/stores
2. **Matrix Multiplication** - 2D tiling, accumulator patterns
3. **Softmax** - Row-wise operations, numerical stability
4. **Attention Scores** - Batched Q @ K^T computation
5. **Causal Masking** - On-the-fly mask generation
6. **Naive Attention** - Compose kernels end-to-end
7. **Fused Attention** - Online softmax, kernel fusion
8. **Flash Attention** - Full 2D tiled algorithm (Final Boss)

## How to Play

1. Open the level file in `levels/`
2. Read the docstrings (they explain everything)
3. Implement the TODO sections
4. Test locally: `python levels/level1_vector_add.py`
5. Submit: `python quest.py check 1`
6. Stuck? `python quest.py hint 1`

Levels unlock sequentially. Complete one to access the next.

## Getting AI Help (Claude Code Tutor)

When you're stuck, use the built-in **Claude Code tutor** for Socratic guidance:

```bash
# In Claude Code, use the /tutor command:
/tutor My softmax gives NaN for large values
/tutor My kernel works but is slower than PyTorch
/tutor What does "incompatible shapes" mean?
```

The tutor will:
- Read your actual implementation
- Analyze for common issues specific to your level
- Guide you with questions (not give away answers)
- Explain the "why" behind GPU programming patterns

**Files:**
- `.claude/skills/triton-tutor.md` - Comprehensive tutor knowledge base
- `.claude/commands/tutor.md` - Slash command definition
- `CLAUDE.md` - Project context for Claude Code

## Running Tests

The project includes a comprehensive pytest test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run only non-GPU tests (faster, CI-friendly)
python -m pytest tests/ -v --ignore=tests/test_baselines.py

# Run specific test file
python -m pytest tests/test_cli.py -v
```

**Test coverage:**
- `test_cli.py` - CLI command functionality
- `test_progress.py` - Progress tracking (save/load/reset)
- `test_hints.py` - Hint system
- `test_levels.py` - Level unlocking logic
- `test_baselines.py` - Baseline implementations (requires CUDA)
- `test_testing_framework.py` - Test harness verification

## Requirements

- Python 3.8+
- PyTorch with CUDA
- Triton

## Project Structure

```
triton_quest/
├── quest.py              # Main CLI
├── CLAUDE.md             # Claude Code project context
├── levels/               # Your challenge files (edit these!)
│   ├── level1_vector_add.py
│   ├── level2_matmul.py
│   ├── level3_softmax.py
│   ├── level4_attention_scores.py
│   ├── level5_causal_mask.py
│   ├── level6_naive_attention.py
│   ├── level7_fused_attention.py
│   └── level8_flash_attention.py
├── utils/
│   ├── baseline.py       # PyTorch reference implementations
│   └── testing.py        # Test harness
├── tests/                # pytest test suite (76 tests)
│   ├── conftest.py       # Shared fixtures
│   ├── test_cli.py
│   ├── test_progress.py
│   ├── test_hints.py
│   ├── test_levels.py
│   ├── test_baselines.py
│   └── test_testing_framework.py
└── .claude/
    ├── skills/
    │   └── triton-tutor.md   # AI tutor knowledge base
    └── commands/
        └── tutor.md          # /tutor slash command
```

---

## Implementation Notes (for Claude Code context)

### What This Project Is

This is a **Test-Driven Development learning system** for Triton GPU programming. The user wanted to learn Triton by implementing causal attention from scratch, with:

- PyTorch baselines as ground truth for correctness testing
- Progressive difficulty levels (gamified as a "quest")
- Hints system that reveals progressively
- Performance benchmarking against PyTorch's SDPA

### Design Decisions

1. **8 Levels of Progression**: Each level builds on the previous, teaching one core concept needed for Flash Attention.

2. **Template Files**: Each level file contains extensive docstrings explaining the concepts, algorithm steps, and hints - but the actual kernel implementation is left as TODO for the user.

3. **Self-Contained Tests**: Each level file can be run directly (`python levels/levelN_*.py`) for quick iteration, plus formal verification via `quest.py check N`.

4. **Online Softmax**: Levels 7-8 teach the key Flash Attention insight - computing softmax incrementally without materializing the full S×S attention matrix.

5. **Progressive Hints**: The hint system reveals 6 hints per level, from conceptual to near-solution, to help users get unstuck without giving away the answer immediately.

### Key Files

- `quest.py`: Main CLI with progress tracking, level checking, hints, benchmarking
- `utils/baseline.py`: PyTorch reference implementations for all operations
- `utils/testing.py`: Test harness with correctness checking and benchmarking utilities
- `levels/__init__.py`: Level metadata (names, descriptions, modules)

### What's NOT Implemented (by design)

The kernel implementations in `levels/` are intentionally incomplete - they contain:
- Full docstrings explaining the algorithm
- Function signatures and wrapper code
- TODO comments marking where the user should write code
- The actual Triton kernel code is left blank for the user to implement

### Future Extensions (if user wants)

- Add backward pass implementations for training
- Add autotuning with `@triton.autotune`
- Add multi-query attention (MQA) / grouped-query attention (GQA) variants
- Add profiling integration with `nsys`

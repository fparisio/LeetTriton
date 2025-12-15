#!/usr/bin/env python3
"""
=============================================================================
                    TRITON ATTENTION QUEST
                    A Gamified Learning Journey
=============================================================================

Welcome, brave adventurer! Your quest is to learn Triton by implementing
causal attention from scratch, progressing through 8 increasingly challenging
levels.

USAGE:
    python quest.py              # Show status and next level
    python quest.py status       # Show progress overview
    python quest.py start <N>    # Show instructions for level N
    python quest.py check <N>    # Check if level N is complete
    python quest.py hint <N>     # Get a hint for level N
    python quest.py benchmark    # Run performance benchmarks
    python quest.py reset        # Reset all progress

=============================================================================
"""

import sys
import os
import json
import importlib
import torch
import math
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from levels import LEVELS
from utils.baseline import (
    vector_add_baseline,
    matmul_baseline,
    softmax_baseline,
    attention_scores_baseline,
    causal_attention_baseline,
    flash_attention_baseline,
    Benchmarker,
)
from utils.testing import QuestTester, TestStatus, print_level_result, LevelResult, TestResult


# =============================================================================
# Progress Tracking
# =============================================================================

PROGRESS_FILE = Path(__file__).parent / ".progress.json"


def load_progress() -> dict:
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_levels": [], "hints_used": {}}


def save_progress(progress: dict):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def reset_progress():
    """Reset all progress."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    print("Progress reset! Ready for a fresh start.")


# =============================================================================
# ASCII Art and Display
# =============================================================================

BANNER = r"""
================================================================================

  ████████╗██████╗ ██╗████████╗ ██████╗ ███╗   ██╗     ██████╗ ██╗   ██╗███████╗
  ╚══██╔══╝██╔══██╗██║╚══██╔══╝██╔═══██╗████╗  ██║    ██╔═══██╗██║   ██║██╔════╝
     ██║   ██████╔╝██║   ██║   ██║   ██║██╔██╗ ██║    ██║   ██║██║   ██║█████╗
     ██║   ██╔══██╗██║   ██║   ██║   ██║██║╚██╗██║    ██║▄▄ ██║██║   ██║██╔══╝
     ██║   ██║  ██║██║   ██║   ╚██████╔╝██║ ╚████║    ╚██████╔╝╚██████╔╝███████╗
     ╚═╝   ╚═╝  ╚═╝╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝     ╚══▀▀═╝  ╚═════╝ ╚══════╝

              Learn Triton by Building Flash Attention from Scratch
================================================================================
"""

LEVEL_COMPLETE_ART = r"""
    +----------------------------------+
    |                                  |
    |         LEVEL COMPLETE!          |
    |              ****                |
    |             ******               |
    |            ********              |
    |           **********             |
    |                                  |
    +----------------------------------+
"""

QUEST_COMPLETE_ART = r"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       ★ ★ ★ ★ ★  QUEST COMPLETE!  ★ ★ ★ ★ ★                  ║
    ║                                                               ║
    ║            You are now a TRITON MASTER!                       ║
    ║                                                               ║
    ║       You've implemented Flash Attention from scratch,        ║
    ║       mastering GPU programming along the way.                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
"""


def print_banner():
    print(BANNER)


def print_status():
    """Print current quest status."""
    progress = load_progress()
    completed = set(progress.get("completed_levels", []))

    print("\n" + "=" * 60)
    print("                    QUEST PROGRESS")
    print("=" * 60 + "\n")

    for level_num, level_info in LEVELS.items():
        status = "✓" if level_num in completed else " "
        lock = "" if level_num == 1 or (level_num - 1) in completed else "🔒"
        print(f"  [{status}] Level {level_num}: {level_info['name']} {lock}")
        print(f"      └─ {level_info['subtitle']}")

    completed_count = len(completed)
    total_count = len(LEVELS)
    progress_bar = "█" * completed_count + "░" * (total_count - completed_count)
    print(f"\n  Progress: [{progress_bar}] {completed_count}/{total_count}")

    if completed_count == total_count:
        print(QUEST_COMPLETE_ART)
    elif completed_count == 0:
        print("\n  Your journey begins! Run: python quest.py start 1")
    else:
        next_level = min([l for l in LEVELS.keys() if l not in completed])
        print(f"\n  Next challenge awaits! Run: python quest.py start {next_level}")


def print_level_instructions(level: int):
    """Print instructions for a specific level."""
    if level not in LEVELS:
        print(f"Error: Level {level} does not exist. Valid levels: 1-8")
        return

    progress = load_progress()
    completed = set(progress.get("completed_levels", []))

    # Check if level is unlocked
    if level > 1 and (level - 1) not in completed:
        print(f"\n🔒 Level {level} is locked!")
        print(f"   Complete Level {level - 1} first to unlock.")
        return

    level_info = LEVELS[level]

    print("\n" + "=" * 60)
    print(f"  LEVEL {level}: {level_info['name'].upper()}")
    print(f"  \"{level_info['subtitle']}\"")
    print("=" * 60)
    print(f"\n  {level_info['description']}")
    print(f"\n  📝 Your task file: levels/{level_info['module']}.py")
    print(f"\n  Instructions:")
    print(f"  1. Open levels/{level_info['module']}.py")
    print(f"  2. Read the docstrings and comments carefully")
    print(f"  3. Implement the TODO sections")
    print(f"  4. Test with: python levels/{level_info['module']}.py")
    print(f"  5. When ready: python quest.py check {level}")
    print(f"\n  Stuck? Get a hint: python quest.py hint {level}")
    print("=" * 60)


# =============================================================================
# Level Checking
# =============================================================================

def check_level(level: int) -> bool:
    """Check if a level implementation is correct."""
    if level not in LEVELS:
        print(f"Error: Level {level} does not exist.")
        return False

    progress = load_progress()
    completed = set(progress.get("completed_levels", []))

    if level > 1 and (level - 1) not in completed:
        print(f"\n🔒 Level {level} is locked! Complete Level {level - 1} first.")
        return False

    level_info = LEVELS[level]
    print(f"\n🔍 Checking Level {level}: {level_info['name']}...")
    print("=" * 50)

    tester = QuestTester()
    tests = []

    try:
        # Import the student's implementation
        module = importlib.import_module(f"levels.{level_info['module']}")
        importlib.reload(module)  # Reload to get latest changes

        if level == 1:
            tests = check_level_1(module, tester)
        elif level == 2:
            tests = check_level_2(module, tester)
        elif level == 3:
            tests = check_level_3(module, tester)
        elif level == 4:
            tests = check_level_4(module, tester)
        elif level == 5:
            tests = check_level_5(module, tester)
        elif level == 6:
            tests = check_level_6(module, tester)
        elif level == 7:
            tests = check_level_7(module, tester)
        elif level == 8:
            tests = check_level_8(module, tester)

    except Exception as e:
        tests = [TestResult(
            name="import",
            status=TestStatus.ERROR,
            message=f"Failed to import module: {e}"
        )]

    # Determine if level passed
    passed = all(t.status == TestStatus.PASSED for t in tests)

    result = LevelResult(
        level=level,
        level_name=level_info['name'],
        passed=passed,
        tests=tests,
    )

    print_level_result(result)

    if passed:
        # Update progress
        if level not in completed:
            progress["completed_levels"].append(level)
            save_progress(progress)
            print(LEVEL_COMPLETE_ART)
            if level < 8:
                print(f"\n  🎉 Level {level + 1} unlocked!")
                print(f"  Run: python quest.py start {level + 1}")
            else:
                print(QUEST_COMPLETE_ART)

    return passed


def check_level_1(module, tester: QuestTester):
    """Check Level 1: Vector Addition."""
    tests = []
    sizes = [1024, 1000, 8192, 65536]

    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        expected = vector_add_baseline(x, y)

        actual, error = tester.run_with_catch(module.vector_add, x, y)

        if error:
            tests.append(TestResult(f"size_{size}", TestStatus.ERROR, error[:100]))
        else:
            tests.append(tester.check_correctness(expected, actual, f"size_{size}"))

    return tests


def check_level_2(module, tester: QuestTester):
    """Check Level 2: Matrix Multiplication."""
    tests = []
    cases = [(64, 64, 64), (128, 256, 128), (100, 100, 100), (512, 512, 512)]

    for M, N, K in cases:
        a = torch.randn(M, K, device='cuda')
        b = torch.randn(K, N, device='cuda')
        expected = matmul_baseline(a, b)

        actual, error = tester.run_with_catch(module.matmul, a, b)

        if error:
            tests.append(TestResult(f"({M},{K})x({K},{N})", TestStatus.ERROR, error[:100]))
        else:
            tests.append(tester.check_correctness(expected, actual, f"({M},{K})x({K},{N})"))

    return tests


def check_level_3(module, tester: QuestTester):
    """Check Level 3: Softmax."""
    tests = []
    cases = [(4, 128), (32, 256), (64, 1024), (8, 100)]

    for n_rows, n_cols in cases:
        x = torch.randn(n_rows, n_cols, device='cuda')
        expected = softmax_baseline(x, dim=-1)

        actual, error = tester.run_with_catch(module.softmax, x)

        if error:
            tests.append(TestResult(f"({n_rows},{n_cols})", TestStatus.ERROR, error[:100]))
        else:
            tests.append(tester.check_correctness(expected, actual, f"({n_rows},{n_cols})"))

    # Numerical stability test
    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')
    expected = softmax_baseline(x_large, dim=-1)
    actual, error = tester.run_with_catch(module.softmax, x_large)

    if error:
        tests.append(TestResult("numerical_stability", TestStatus.ERROR, error[:100]))
    elif torch.isnan(actual).any() or torch.isinf(actual).any():
        tests.append(TestResult("numerical_stability", TestStatus.FAILED, "Contains NaN or Inf!"))
    else:
        tests.append(tester.check_correctness(expected, actual, "numerical_stability"))

    return tests


def check_level_4(module, tester: QuestTester):
    """Check Level 4: Attention Scores."""
    tests = []
    cases = [(1, 1, 64, 64), (2, 4, 128, 64), (1, 8, 256, 32)]

    for B, H, S, D in cases:
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')
        scale = 1.0 / math.sqrt(D)

        expected = attention_scores_baseline(q, k, scale)
        actual, error = tester.run_with_catch(module.attention_scores, q, k, scale)

        if error:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.ERROR, error[:100]))
        else:
            tests.append(tester.check_correctness(expected, actual, f"B{B}_H{H}_S{S}_D{D}"))

    return tests


def check_level_5(module, tester: QuestTester):
    """Check Level 5: Causal Masking."""
    tests = []
    cases = [(1, 1, 64), (2, 4, 128), (1, 8, 256)]

    for B, H, S in cases:
        scores = torch.randn(B, H, S, S, device='cuda')

        # Reference implementation
        mask = torch.tril(torch.ones(S, S, device='cuda'))
        expected = torch.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
        expected = torch.nan_to_num(expected)

        actual, error = tester.run_with_catch(module.causal_softmax, scores)

        if error:
            tests.append(TestResult(f"B{B}_H{H}_S{S}", TestStatus.ERROR, error[:100]))
        else:
            tests.append(tester.check_correctness(expected, actual, f"B{B}_H{H}_S{S}"))

    # Causal property test
    scores = torch.randn(1, 1, 4, 4, device='cuda')
    actual, error = tester.run_with_catch(module.causal_softmax, scores)

    if error:
        tests.append(TestResult("causal_property", TestStatus.ERROR, error[:100]))
    elif actual is not None:
        upper_tri = torch.triu(actual[0, 0], diagonal=1)
        if upper_tri.abs().max() < 1e-6:
            tests.append(TestResult("causal_property", TestStatus.PASSED, "Future positions correctly masked"))
        else:
            tests.append(TestResult("causal_property", TestStatus.FAILED, "Future positions not properly masked"))

    return tests


def check_level_6(module, tester: QuestTester):
    """Check Level 6: Naive Causal Attention."""
    tests = []
    cases = [(1, 1, 64, 32), (2, 4, 128, 64), (1, 8, 256, 64)]

    for B, H, S, D in cases:
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')
        v = torch.randn(B, H, S, D, device='cuda')

        expected = causal_attention_baseline(q, k, v)
        actual, error = tester.run_with_catch(module.naive_causal_attention, q, k, v)

        if error:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.ERROR, error[:100]))
        elif actual is None:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.FAILED, "Returned None"))
        else:
            tests.append(tester.check_correctness(expected, actual, f"B{B}_H{H}_S{S}_D{D}"))

    return tests


def check_level_7(module, tester: QuestTester):
    """Check Level 7: Fused Attention."""
    tests = []
    cases = [(1, 1, 64, 32), (2, 4, 128, 64), (1, 8, 256, 64)]

    for B, H, S, D in cases:
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')
        v = torch.randn(B, H, S, D, device='cuda')

        expected = causal_attention_baseline(q, k, v)
        actual, error = tester.run_with_catch(module.fused_attention, q, k, v)

        if error:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.ERROR, error[:100]))
        elif actual is None:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.FAILED, "Returned None"))
        else:
            tests.append(tester.check_correctness(expected, actual, f"B{B}_H{H}_S{S}_D{D}"))

    return tests


def check_level_8(module, tester: QuestTester):
    """Check Level 8: Flash Attention."""
    tests = []
    cases = [(1, 1, 64, 32), (2, 4, 128, 64), (1, 8, 256, 64), (2, 8, 512, 64)]

    for B, H, S, D in cases:
        q = torch.randn(B, H, S, D, device='cuda')
        k = torch.randn(B, H, S, D, device='cuda')
        v = torch.randn(B, H, S, D, device='cuda')

        expected = causal_attention_baseline(q, k, v)
        actual, error = tester.run_with_catch(module.flash_attention, q, k, v)

        if error:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.ERROR, error[:100]))
        elif actual is None:
            tests.append(TestResult(f"B{B}_H{H}_S{S}_D{D}", TestStatus.FAILED, "Returned None"))
        else:
            tests.append(tester.check_correctness(expected, actual, f"B{B}_H{H}_S{S}_D{D}"))

    return tests


# =============================================================================
# Hints System
# =============================================================================

HINTS = {
    1: [
        "Start with tl.program_id(0) to get which block you are.",
        "Use tl.arange(0, BLOCK_SIZE) to create a range of offsets.",
        "Your global offset is: pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "Create a mask: offsets < n_elements to handle boundaries.",
        "Load with tl.load(ptr + offsets, mask=mask), store with tl.store().",
        "Full pattern: pid = tl.program_id(0); offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE); mask = offs < n; x = tl.load(x_ptr + offs, mask=mask); ...",
    ],
    2: [
        "Use a 2D grid: pid_m = tl.program_id(0), pid_n = tl.program_id(1).",
        "Each program computes a BLOCK_M x BLOCK_N tile of output.",
        "Create 2D offsets: offs_m[:, None], offs_n[None, :] for broadcasting.",
        "Initialize accumulator: acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).",
        "Loop over K: for k in range(0, K, BLOCK_K): load tiles, acc += tl.dot(a, b).",
        "A pointer: a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak",
    ],
    3: [
        "Each program handles one row: row_idx = tl.program_id(0).",
        "Load entire row with: row = tl.load(ptr + col_offs, mask=col_offs < n_cols, other=-inf).",
        "The -inf for out-of-bounds ensures they don't affect max.",
        "Find max: max_val = tl.max(row, axis=0). This returns a scalar!",
        "Compute: numerator = tl.exp(row - max_val); denominator = tl.sum(numerator, axis=0).",
        "Output: output = numerator / denominator. Store with mask.",
    ],
    4: [
        "Use a 3D grid: (row_blocks, col_blocks, batch*heads).",
        "This is similar to matmul, but Q @ K^T (note the transpose).",
        "Load K normally, then use tl.trans(k_block) or adjust your dot product.",
        "Don't forget to multiply by scale at the end!",
        "Loop over head_dim in chunks of BLOCK_D.",
    ],
    5: [
        "Each program handles one row: row_idx = tl.program_id(0).",
        "For row i, valid columns are 0, 1, ..., i (causal mask).",
        "Generate mask on-the-fly: causal_mask = col_offs <= row_idx.",
        "Apply mask before softmax: scores = tl.where(causal_mask, scores, -inf).",
        "Rest is same as softmax: max -> exp -> sum -> divide.",
    ],
    6: [
        "You can compose your previous implementations!",
        "Call: scores = attention_scores(q, k, scale)",
        "Call: weights = causal_softmax(scores)",
        "Final: output = weights @ v (can use torch.matmul for now)",
        "Make sure shapes are correct throughout.",
    ],
    7: [
        "Process one query row at a time, iterate over K/V tiles.",
        "Maintain: running_max, running_sum, acc (accumulated output).",
        "For each K/V tile: compute partial scores, update softmax state.",
        "Online softmax: new_max = max(old_max, tile_max); rescale previous values.",
        "exp_scores = exp(scores - running_max); sum += sum(exp_scores).",
        "acc += exp_scores @ V_tile (outer product accumulation).",
    ],
    8: [
        "Grid is 2D: (num_q_blocks, batch*heads). Each program handles BLOCK_M query rows.",
        "Track per-row state: max_vec(BLOCK_M,), sum_vec(BLOCK_M,), acc(BLOCK_M, D).",
        "S_block = tl.dot(Q_block, tl.trans(K_block)) * scale gives (BLOCK_M, BLOCK_N).",
        "row_max = tl.max(S_block, axis=1) gives per-row max.",
        "Correction: exp(old_max - new_max) to rescale previous accumulations.",
        "Causal mask: q_offs[:, None] >= k_offs[None, :] creates 2D mask.",
        "Final: acc += tl.dot(exp_s.to(v_block.dtype), v_block)",
    ],
}


def show_hint(level: int):
    """Show progressive hints for a level."""
    if level not in HINTS:
        print(f"No hints available for level {level}")
        return

    progress = load_progress()
    hints_used = progress.get("hints_used", {})
    level_key = str(level)

    current_hint = hints_used.get(level_key, 0)
    hints = HINTS[level]

    if current_hint >= len(hints):
        print(f"\n💡 You've seen all {len(hints)} hints for Level {level}!")
        print("If you're still stuck, try re-reading the docstrings carefully.")
        return

    print(f"\n💡 Hint {current_hint + 1}/{len(hints)} for Level {level}:")
    print(f"   {hints[current_hint]}")

    # Update hint counter
    hints_used[level_key] = current_hint + 1
    progress["hints_used"] = hints_used
    save_progress(progress)

    if current_hint + 1 < len(hints):
        print(f"\n   More hints available: python quest.py hint {level}")


# =============================================================================
# Benchmarking
# =============================================================================

def run_benchmarks():
    """Run performance benchmarks comparing implementations to PyTorch."""
    print("\n" + "=" * 60)
    print("                 PERFORMANCE BENCHMARKS")
    print("=" * 60)

    progress = load_progress()
    completed = set(progress.get("completed_levels", []))

    benchmarker = Benchmarker(warmup_iters=10, bench_iters=50)

    # Test configuration
    B, H, S, D = 2, 8, 1024, 64
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

    print(f"\nConfig: batch={B}, heads={H}, seq_len={S}, head_dim={D}")
    print(f"Data type: float16\n")

    # PyTorch baseline
    baseline_time, baseline_std = benchmarker.benchmark(
        lambda: flash_attention_baseline(q.float(), k.float(), v.float()).half()
    )
    print(f"PyTorch SDPA (baseline): {baseline_time:.3f} ms ± {baseline_std:.3f}")

    # Check completed implementations
    implementations = [
        (6, "naive_causal_attention", "Naive Attention"),
        (7, "fused_attention", "Fused Attention"),
        (8, "flash_attention", "Flash Attention"),
    ]

    for level, func_name, display_name in implementations:
        if level in completed:
            try:
                module = importlib.import_module(f"levels.{LEVELS[level]['module']}")
                func = getattr(module, func_name)

                time_ms, std_ms = benchmarker.benchmark(
                    lambda: func(q.float(), k.float(), v.float())
                )
                speedup = baseline_time / time_ms
                indicator = "🚀" if speedup > 1 else "🐢"
                print(f"Level {level} ({display_name}): {time_ms:.3f} ms ± {std_ms:.3f} [{speedup:.2f}x] {indicator}")
            except Exception as e:
                print(f"Level {level} ({display_name}): Error - {e}")
        else:
            print(f"Level {level} ({display_name}): 🔒 Not completed")

    print("\n" + "=" * 60)


# =============================================================================
# Main CLI
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print_banner()
        print_status()
        return

    command = sys.argv[1].lower()

    if command == "status":
        print_banner()
        print_status()

    elif command == "start":
        if len(sys.argv) < 3:
            print("Usage: python quest.py start <level>")
            return
        try:
            level = int(sys.argv[2])
            print_level_instructions(level)
        except ValueError:
            print("Error: Level must be a number (1-8)")

    elif command == "check":
        if len(sys.argv) < 3:
            print("Usage: python quest.py check <level>")
            return
        try:
            level = int(sys.argv[2])
            check_level(level)
        except ValueError:
            print("Error: Level must be a number (1-8)")

    elif command == "hint":
        if len(sys.argv) < 3:
            print("Usage: python quest.py hint <level>")
            return
        try:
            level = int(sys.argv[2])
            show_hint(level)
        except ValueError:
            print("Error: Level must be a number (1-8)")

    elif command == "benchmark":
        run_benchmarks()

    elif command == "reset":
        reset_progress()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python quest.py [status|start|check|hint|benchmark|reset]")


if __name__ == "__main__":
    main()

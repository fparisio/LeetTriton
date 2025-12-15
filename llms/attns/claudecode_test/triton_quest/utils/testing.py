"""
Test harness for the Triton Quest.
Validates correctness and measures performance against baselines.
"""

import torch
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import traceback

from .baseline import Benchmarker


class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str
    details: Optional[dict] = None


@dataclass
class LevelResult:
    level: int
    level_name: str
    passed: bool
    tests: List[TestResult]
    performance: Optional[dict] = None


class QuestTester:
    """Main test harness for validating Triton implementations."""

    def __init__(
        self,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        device: str = "cuda",
    ):
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.benchmarker = Benchmarker()

    def check_correctness(
        self,
        expected: torch.Tensor,
        actual: torch.Tensor,
        test_name: str = "correctness",
    ) -> TestResult:
        """Check if two tensors are close enough."""
        try:
            if actual is None:
                return TestResult(
                    name=test_name,
                    status=TestStatus.ERROR,
                    message="Your function returned None!",
                )

            if expected.shape != actual.shape:
                return TestResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Shape mismatch: expected {expected.shape}, got {actual.shape}",
                )

            if not torch.allclose(expected, actual, rtol=self.rtol, atol=self.atol):
                max_diff = (expected - actual).abs().max().item()
                mean_diff = (expected - actual).abs().mean().item()

                # Find position of max difference
                diff = (expected - actual).abs()
                max_idx = diff.argmax().item()

                return TestResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Values don't match! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}",
                    details={
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "expected_sample": expected.flatten()[max_idx].item(),
                        "actual_sample": actual.flatten()[max_idx].item(),
                    },
                )

            return TestResult(
                name=test_name,
                status=TestStatus.PASSED,
                message="Correctness verified!",
            )

        except Exception as e:
            return TestResult(
                name=test_name,
                status=TestStatus.ERROR,
                message=f"Error during comparison: {str(e)}",
            )

    def run_with_catch(
        self, fn: Callable, *args, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Run a function and catch any exceptions."""
        try:
            result = fn(*args, **kwargs)
            return result, None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg

    def test_shapes(
        self,
        fn: Callable,
        baseline_fn: Callable,
        shapes: List[Tuple],
        test_name: str = "shape_test",
        input_generator: Optional[Callable] = None,
    ) -> List[TestResult]:
        """Test function across multiple input shapes."""
        results = []

        for shape in shapes:
            if input_generator:
                inputs = input_generator(shape, self.device)
            else:
                inputs = (torch.randn(shape, device=self.device),)

            expected = baseline_fn(*inputs)
            actual, error = self.run_with_catch(fn, *inputs)

            if error:
                results.append(
                    TestResult(
                        name=f"{test_name}_shape_{shape}",
                        status=TestStatus.ERROR,
                        message=f"Error: {error}",
                    )
                )
            else:
                result = self.check_correctness(
                    expected, actual, f"{test_name}_shape_{shape}"
                )
                results.append(result)

        return results

    def benchmark_against_baseline(
        self,
        fn: Callable,
        baseline_fn: Callable,
        *args,
        **kwargs,
    ) -> dict:
        """Benchmark test function against baseline."""
        return self.benchmarker.compare(baseline_fn, fn, *args, **kwargs)


def print_test_result(result: TestResult, verbose: bool = False):
    """Pretty print a test result."""
    status_symbols = {
        TestStatus.PASSED: "\033[92m[PASS]\033[0m",
        TestStatus.FAILED: "\033[91m[FAIL]\033[0m",
        TestStatus.ERROR: "\033[93m[ERR!]\033[0m",
        TestStatus.SKIPPED: "\033[90m[SKIP]\033[0m",
    }

    symbol = status_symbols[result.status]
    print(f"  {symbol} {result.name}: {result.message}")

    if verbose and result.details:
        for key, value in result.details.items():
            print(f"         {key}: {value}")


def print_level_result(result: LevelResult):
    """Pretty print level results with ASCII art."""
    if result.passed:
        print(f"\n{'='*60}")
        print(f"\033[92m  LEVEL {result.level} COMPLETE: {result.level_name}\033[0m")
        print(f"{'='*60}")
        print("""
    +----------------+
    |   LEVEL UP!    |
    |      ***       |
    |     *****      |
    |    *******     |
    |   *********    |
    +----------------+
        """)
    else:
        print(f"\n{'='*60}")
        print(f"\033[91m  LEVEL {result.level} INCOMPLETE: {result.level_name}\033[0m")
        print(f"{'='*60}")

    print("\nTest Results:")
    for test in result.tests:
        print_test_result(test)

    if result.performance:
        print("\nPerformance:")
        print(f"  Baseline: {result.performance['baseline_ms']:.3f} ms")
        print(f"  Your impl: {result.performance['test_ms']:.3f} ms")
        speedup = result.performance['speedup']
        if speedup > 1:
            print(f"  \033[92mSpeedup: {speedup:.2f}x faster!\033[0m")
        else:
            print(f"  \033[93mSpeedup: {speedup:.2f}x (slower than baseline)\033[0m")


def generate_attention_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Q, K, V tensors for attention testing."""
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v

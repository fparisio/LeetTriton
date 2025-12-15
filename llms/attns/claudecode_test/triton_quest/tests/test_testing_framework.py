"""
Tests for the testing framework (QuestTester and related classes).
"""

import pytest
import torch


class TestTestStatus:
    """Test TestStatus enum."""

    def test_status_values_exist(self):
        """Test that all expected status values exist."""
        from utils.testing import TestStatus

        assert hasattr(TestStatus, 'PASSED')
        assert hasattr(TestStatus, 'FAILED')
        assert hasattr(TestStatus, 'ERROR')
        assert hasattr(TestStatus, 'SKIPPED')

    def test_status_values(self):
        """Test status string values."""
        from utils.testing import TestStatus

        assert TestStatus.PASSED.value == "PASSED"
        assert TestStatus.FAILED.value == "FAILED"
        assert TestStatus.ERROR.value == "ERROR"
        assert TestStatus.SKIPPED.value == "SKIPPED"


class TestTestResult:
    """Test TestResult dataclass."""

    def test_create_basic_result(self):
        """Test creating a basic test result."""
        from utils.testing import TestResult, TestStatus

        result = TestResult(
            name="test_name",
            status=TestStatus.PASSED,
            message="Test passed!"
        )

        assert result.name == "test_name"
        assert result.status == TestStatus.PASSED
        assert result.message == "Test passed!"
        assert result.details is None

    def test_create_result_with_details(self):
        """Test creating a result with details."""
        from utils.testing import TestResult, TestStatus

        details = {"max_diff": 0.001, "mean_diff": 0.0001}
        result = TestResult(
            name="test_name",
            status=TestStatus.FAILED,
            message="Test failed!",
            details=details
        )

        assert result.details == details


class TestLevelResult:
    """Test LevelResult dataclass."""

    def test_create_level_result(self):
        """Test creating a level result."""
        from utils.testing import LevelResult, TestResult, TestStatus

        tests = [
            TestResult("test1", TestStatus.PASSED, "OK"),
            TestResult("test2", TestStatus.PASSED, "OK"),
        ]

        result = LevelResult(
            level=1,
            level_name="Vector Addition",
            passed=True,
            tests=tests
        )

        assert result.level == 1
        assert result.level_name == "Vector Addition"
        assert result.passed is True
        assert len(result.tests) == 2
        assert result.performance is None

    def test_level_result_with_performance(self):
        """Test level result with performance data."""
        from utils.testing import LevelResult, TestResult, TestStatus

        tests = [TestResult("test1", TestStatus.PASSED, "OK")]
        performance = {"baseline_ms": 1.0, "test_ms": 0.5, "speedup": 2.0}

        result = LevelResult(
            level=1,
            level_name="Vector Addition",
            passed=True,
            tests=tests,
            performance=performance
        )

        assert result.performance == performance


class TestQuestTester:
    """Test QuestTester class."""

    @pytest.fixture
    def tester(self):
        """Create a QuestTester instance."""
        from utils.testing import QuestTester
        return QuestTester()

    def test_tester_default_tolerances(self, tester):
        """Test default tolerance values."""
        assert tester.rtol == 1e-3
        assert tester.atol == 1e-3

    def test_tester_custom_tolerances(self):
        """Test custom tolerance values."""
        from utils.testing import QuestTester

        tester = QuestTester(rtol=1e-4, atol=1e-4)
        assert tester.rtol == 1e-4
        assert tester.atol == 1e-4


@pytest.mark.cuda
class TestQuestTesterCorrectness:
    """Test QuestTester correctness checking (requires CUDA)."""

    @pytest.fixture
    def tester(self):
        """Create a QuestTester instance."""
        from utils.testing import QuestTester
        return QuestTester()

    def test_check_correctness_pass(self, tester, cuda_available):
        """Test correctness check with matching tensors."""
        from utils.testing import TestStatus

        expected = torch.randn(100, device='cuda')
        actual = expected.clone()

        result = tester.check_correctness(expected, actual, "test")

        assert result.status == TestStatus.PASSED

    def test_check_correctness_fail_values(self, tester, cuda_available):
        """Test correctness check with mismatched values."""
        from utils.testing import TestStatus

        expected = torch.randn(100, device='cuda')
        actual = expected + 1.0  # Add large offset

        result = tester.check_correctness(expected, actual, "test")

        assert result.status == TestStatus.FAILED
        assert "diff" in result.message.lower()

    def test_check_correctness_fail_shape(self, tester, cuda_available):
        """Test correctness check with mismatched shapes."""
        from utils.testing import TestStatus

        expected = torch.randn(100, device='cuda')
        actual = torch.randn(200, device='cuda')

        result = tester.check_correctness(expected, actual, "test")

        assert result.status == TestStatus.FAILED
        assert "shape" in result.message.lower()

    def test_check_correctness_none_actual(self, tester, cuda_available):
        """Test correctness check when actual is None."""
        from utils.testing import TestStatus

        expected = torch.randn(100, device='cuda')

        result = tester.check_correctness(expected, None, "test")

        assert result.status == TestStatus.ERROR
        assert "None" in result.message


@pytest.mark.cuda
class TestQuestTesterRunWithCatch:
    """Test QuestTester run_with_catch method (requires CUDA)."""

    @pytest.fixture
    def tester(self):
        """Create a QuestTester instance."""
        from utils.testing import QuestTester
        return QuestTester()

    def test_run_with_catch_success(self, tester, cuda_available):
        """Test run_with_catch with successful function."""

        def add(x, y):
            return x + y

        result, error = tester.run_with_catch(add, 1, 2)

        assert result == 3
        assert error is None

    def test_run_with_catch_exception(self, tester, cuda_available):
        """Test run_with_catch with failing function."""

        def failing_fn():
            raise ValueError("Test error")

        result, error = tester.run_with_catch(failing_fn)

        assert result is None
        assert error is not None
        assert "ValueError" in error
        assert "Test error" in error

    def test_run_with_catch_with_kwargs(self, tester, cuda_available):
        """Test run_with_catch with keyword arguments."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result, error = tester.run_with_catch(greet, "World", greeting="Hi")

        assert result == "Hi, World!"
        assert error is None


class TestPrintFunctions:
    """Test print helper functions."""

    def test_print_test_result(self, capsys):
        """Test print_test_result function."""
        from utils.testing import print_test_result, TestResult, TestStatus

        result = TestResult("my_test", TestStatus.PASSED, "All good!")
        print_test_result(result)

        captured = capsys.readouterr()
        assert "my_test" in captured.out
        assert "All good!" in captured.out

    def test_print_level_result_passed(self, capsys):
        """Test print_level_result for passed level."""
        from utils.testing import print_level_result, LevelResult, TestResult, TestStatus

        tests = [TestResult("test1", TestStatus.PASSED, "OK")]
        result = LevelResult(
            level=1,
            level_name="Vector Addition",
            passed=True,
            tests=tests
        )

        print_level_result(result)

        captured = capsys.readouterr()
        assert "LEVEL 1" in captured.out
        assert "COMPLETE" in captured.out

    def test_print_level_result_failed(self, capsys):
        """Test print_level_result for failed level."""
        from utils.testing import print_level_result, LevelResult, TestResult, TestStatus

        tests = [TestResult("test1", TestStatus.FAILED, "Not OK")]
        result = LevelResult(
            level=1,
            level_name="Vector Addition",
            passed=False,
            tests=tests
        )

        print_level_result(result)

        captured = capsys.readouterr()
        assert "LEVEL 1" in captured.out
        assert "INCOMPLETE" in captured.out

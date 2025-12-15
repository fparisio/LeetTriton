"""
Tests for the CLI functionality of quest.py.
"""

import pytest
import sys
from io import StringIO
from unittest.mock import patch


class TestCLICommands:
    """Test CLI command parsing and execution."""

    def test_main_no_args_shows_banner(self, temp_progress_file, capsys):
        """Test that running with no args shows the banner and status."""
        import quest
        with patch.object(sys, 'argv', ['quest.py']):
            quest.main()

        captured = capsys.readouterr()
        assert "Learn Triton" in captured.out  # Banner text
        assert "QUEST PROGRESS" in captured.out

    def test_status_command(self, temp_progress_file, capsys):
        """Test the status command."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'status']):
            quest.main()

        captured = capsys.readouterr()
        assert "QUEST PROGRESS" in captured.out
        assert "Level 1" in captured.out

    def test_start_valid_level(self, temp_progress_file, capsys):
        """Test start command with valid level."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start', '1']):
            quest.main()

        captured = capsys.readouterr()
        assert "LEVEL 1" in captured.out
        assert "VECTOR ADDITION" in captured.out

    def test_start_locked_level(self, temp_progress_file, capsys):
        """Test start command with locked level."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start', '2']):
            quest.main()

        captured = capsys.readouterr()
        assert "locked" in captured.out.lower()

    def test_start_invalid_level_number(self, temp_progress_file, capsys):
        """Test start command with invalid level number."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start', '99']):
            quest.main()

        captured = capsys.readouterr()
        assert "Error" in captured.out or "does not exist" in captured.out

    def test_start_non_numeric_level(self, temp_progress_file, capsys):
        """Test start command with non-numeric level."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start', 'abc']):
            quest.main()

        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_start_missing_level_arg(self, temp_progress_file, capsys):
        """Test start command without level argument."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start']):
            quest.main()

        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_check_missing_level_arg(self, temp_progress_file, capsys):
        """Test check command without level argument."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'check']):
            quest.main()

        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_hint_missing_level_arg(self, temp_progress_file, capsys):
        """Test hint command without level argument."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'hint']):
            quest.main()

        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_reset_command(self, temp_progress_file, capsys):
        """Test reset command."""
        import quest
        import json

        # Create some progress first
        with open(temp_progress_file, 'w') as f:
            json.dump({"completed_levels": [1], "hints_used": {}}, f)

        with patch.object(sys, 'argv', ['quest.py', 'reset']):
            quest.main()

        captured = capsys.readouterr()
        assert "reset" in captured.out.lower()
        assert not temp_progress_file.exists()

    def test_unknown_command(self, temp_progress_file, capsys):
        """Test unknown command shows usage."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'unknowncommand']):
            quest.main()

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        assert "Usage" in captured.out


class TestStartWithUnlockedLevels:
    """Test start command with various unlock states."""

    def test_start_level2_after_level1_complete(self, progress_with_level1_complete, capsys):
        """Test that level 2 is accessible after completing level 1."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start', '2']):
            quest.main()

        captured = capsys.readouterr()
        assert "locked" not in captured.out.lower()
        assert "LEVEL 2" in captured.out

    def test_start_level4_with_levels123_complete(self, progress_with_multiple_levels, capsys):
        """Test that level 4 is accessible after completing levels 1-3."""
        import quest
        with patch.object(sys, 'argv', ['quest.py', 'start', '4']):
            quest.main()

        captured = capsys.readouterr()
        assert "locked" not in captured.out.lower()
        assert "LEVEL 4" in captured.out

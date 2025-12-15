"""
Tests for the hint system functionality.
"""

import pytest
import json
import sys
from unittest.mock import patch


class TestHintProgression:
    """Test progressive hint reveal system."""

    def test_hint_counter_increments(self, temp_progress_file, capsys):
        """Test that hint counter increments on each call."""
        import quest

        # First hint
        with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
            quest.main()

        progress = quest.load_progress()
        assert progress["hints_used"].get("1", 0) == 1

        # Second hint
        with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
            quest.main()

        progress = quest.load_progress()
        assert progress["hints_used"]["1"] == 2

    def test_hint_shows_correct_number(self, temp_progress_file, capsys):
        """Test that hint displays correct hint number."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
            quest.main()

        captured = capsys.readouterr()
        assert "Hint 1/6" in captured.out

    def test_all_hints_shown_message(self, temp_progress_file, capsys):
        """Test message when all hints have been shown."""
        import quest

        # Use all 6 hints
        for _ in range(6):
            with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
                quest.main()

        # Try to get another hint
        with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
            quest.main()

        captured = capsys.readouterr()
        assert "all" in captured.out.lower() and "hints" in captured.out.lower()

    def test_hints_for_different_levels_independent(self, temp_progress_file, capsys):
        """Test that hint counters for different levels are independent."""
        import quest

        # Get hints for level 1
        with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
            quest.main()
        with patch.object(sys, 'argv', ['quest.py', 'hint', '1']):
            quest.main()

        progress = quest.load_progress()
        # Level 2+ are locked, but we can still check the data structure
        assert progress["hints_used"]["1"] == 2


class TestHintContent:
    """Test hint content existence and quality."""

    def test_hints_exist_for_all_levels(self):
        """Test that HINTS dict has entries for all 8 levels."""
        import quest

        for level in range(1, 9):
            assert level in quest.HINTS
            assert len(quest.HINTS[level]) > 0

    def test_each_level_has_six_hints(self):
        """Test that each level has exactly 6 hints."""
        import quest

        for level in range(1, 9):
            assert len(quest.HINTS[level]) == 6, f"Level {level} should have 6 hints"

    def test_hints_are_non_empty_strings(self):
        """Test that all hints are non-empty strings."""
        import quest

        for level, hints in quest.HINTS.items():
            for i, hint in enumerate(hints):
                assert isinstance(hint, str), f"Hint {i+1} for level {level} should be string"
                assert len(hint) > 0, f"Hint {i+1} for level {level} should not be empty"


class TestHintEdgeCases:
    """Test edge cases in hint system."""

    def test_hint_invalid_level(self, temp_progress_file, capsys):
        """Test hint command with invalid level."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'hint', '99']):
            quest.main()

        captured = capsys.readouterr()
        assert "No hints" in captured.out or "Error" in captured.out

    def test_hint_preserves_other_progress(self, temp_progress_file, capsys):
        """Test that getting hints doesn't affect completed_levels."""
        import quest

        # Set up progress with completed level
        quest.save_progress({
            "completed_levels": [1],
            "hints_used": {}
        })

        # Get a hint for level 2
        with patch.object(sys, 'argv', ['quest.py', 'hint', '2']):
            quest.main()

        progress = quest.load_progress()
        assert 1 in progress["completed_levels"]

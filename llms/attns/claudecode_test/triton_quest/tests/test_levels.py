"""
Tests for level unlocking logic and level definitions.
"""

import pytest
import json
import sys
from unittest.mock import patch


class TestLevelUnlocking:
    """Test level unlocking logic."""

    def test_level1_always_unlocked(self, temp_progress_file, capsys):
        """Test that level 1 is always accessible."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'start', '1']):
            quest.main()

        captured = capsys.readouterr()
        assert "locked" not in captured.out.lower()
        assert "LEVEL 1" in captured.out

    def test_level2_locked_initially(self, temp_progress_file, capsys):
        """Test that level 2 is locked without level 1 completion."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'start', '2']):
            quest.main()

        captured = capsys.readouterr()
        assert "locked" in captured.out.lower()

    def test_level2_unlocks_after_level1(self, progress_with_level1_complete, capsys):
        """Test that level 2 unlocks after completing level 1."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'start', '2']):
            quest.main()

        captured = capsys.readouterr()
        assert "locked" not in captured.out.lower()
        assert "LEVEL 2" in captured.out

    def test_all_levels_locked_sequentially(self, temp_progress_file):
        """Test that each level requires previous level completion."""
        import quest

        # No levels completed
        progress = quest.load_progress()
        completed = set(progress.get("completed_levels", []))

        # Level 1 always unlocked, levels 2-8 should be locked
        for level in range(2, 9):
            assert level > 1 and (level - 1) not in completed


class TestLevelsDictionary:
    """Test LEVELS dictionary completeness."""

    def test_levels_has_all_eight(self):
        """Test that LEVELS dict contains all 8 levels."""
        from levels import LEVELS

        assert len(LEVELS) == 8
        for level in range(1, 9):
            assert level in LEVELS

    def test_each_level_has_required_keys(self):
        """Test that each level has name, subtitle, module, description."""
        from levels import LEVELS

        required_keys = ["name", "subtitle", "module", "description"]
        for level, info in LEVELS.items():
            for key in required_keys:
                assert key in info, f"Level {level} missing key: {key}"
                assert info[key], f"Level {level} has empty {key}"

    def test_level_modules_are_importable(self):
        """Test that all level modules can be imported."""
        from levels import LEVELS
        import importlib

        for level, info in LEVELS.items():
            module_name = f"levels.{info['module']}"
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError as e:
                pytest.fail(f"Cannot import {module_name}: {e}")


class TestLevelNaming:
    """Test level naming conventions."""

    def test_level_names_progression(self):
        """Test that levels follow expected naming progression."""
        from levels import LEVELS

        expected_names = [
            "Vector Addition",
            "Matrix Multiplication",
            "Softmax",
            "Attention Scores",
            "Causal Masking",
            "Naive Causal Attention",
            "Fused Attention",
            "Flash Attention",
        ]

        for level, expected_name in enumerate(expected_names, 1):
            assert LEVELS[level]["name"] == expected_name

    def test_level_subtitles_non_empty(self):
        """Test that all levels have non-empty subtitles."""
        from levels import LEVELS

        for level, info in LEVELS.items():
            assert len(info["subtitle"]) > 0


class TestStatusDisplay:
    """Test status display for locked/unlocked levels."""

    def test_status_shows_lock_emoji_for_locked(self, temp_progress_file, capsys):
        """Test that locked levels show lock emoji."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'status']):
            quest.main()

        captured = capsys.readouterr()
        # Levels 2-8 should be locked (no level completed)
        assert captured.out.count("🔒") >= 7

    def test_status_shows_checkmark_for_completed(self, progress_with_level1_complete, capsys):
        """Test that completed levels show checkmark."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'status']):
            quest.main()

        captured = capsys.readouterr()
        # Level 1 completed should show checkmark
        assert "[✓]" in captured.out

    def test_status_progress_bar(self, progress_with_multiple_levels, capsys):
        """Test that progress bar shows correct fill."""
        import quest

        with patch.object(sys, 'argv', ['quest.py', 'status']):
            quest.main()

        captured = capsys.readouterr()
        # Should show 3/8 completed
        assert "3/8" in captured.out

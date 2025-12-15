"""
Tests for progress tracking functionality.
"""

import pytest
import json


class TestLoadProgress:
    """Test load_progress() function."""

    def test_load_progress_no_file_returns_defaults(self, temp_progress_file):
        """Test that loading with no file returns default structure."""
        import quest

        # Ensure file doesn't exist
        if temp_progress_file.exists():
            temp_progress_file.unlink()

        progress = quest.load_progress()

        assert "completed_levels" in progress
        assert "hints_used" in progress
        assert progress["completed_levels"] == []
        assert progress["hints_used"] == {}

    def test_load_progress_existing_file(self, temp_progress_file):
        """Test loading from an existing progress file."""
        import quest

        expected_progress = {
            "completed_levels": [1, 2],
            "hints_used": {"1": 3, "2": 1}
        }
        with open(temp_progress_file, 'w') as f:
            json.dump(expected_progress, f)

        progress = quest.load_progress()

        assert progress["completed_levels"] == [1, 2]
        assert progress["hints_used"]["1"] == 3


class TestSaveProgress:
    """Test save_progress() function."""

    def test_save_progress_creates_file(self, temp_progress_file):
        """Test that save_progress creates the file."""
        import quest

        progress = {
            "completed_levels": [1],
            "hints_used": {"1": 2}
        }
        quest.save_progress(progress)

        assert temp_progress_file.exists()

    def test_save_progress_valid_json(self, temp_progress_file):
        """Test that saved progress is valid JSON."""
        import quest

        progress = {
            "completed_levels": [1, 2, 3],
            "hints_used": {"1": 6, "2": 4}
        }
        quest.save_progress(progress)

        with open(temp_progress_file) as f:
            loaded = json.load(f)

        assert loaded == progress

    def test_save_progress_overwrites(self, temp_progress_file):
        """Test that save_progress overwrites existing data."""
        import quest

        # Save initial progress
        quest.save_progress({"completed_levels": [1], "hints_used": {}})

        # Save new progress
        quest.save_progress({"completed_levels": [1, 2], "hints_used": {"1": 1}})

        progress = quest.load_progress()
        assert progress["completed_levels"] == [1, 2]


class TestResetProgress:
    """Test reset_progress() function."""

    def test_reset_progress_removes_file(self, temp_progress_file):
        """Test that reset removes the progress file."""
        import quest

        # Create a progress file
        quest.save_progress({"completed_levels": [1], "hints_used": {}})
        assert temp_progress_file.exists()

        quest.reset_progress()
        assert not temp_progress_file.exists()

    def test_reset_progress_no_file_no_error(self, temp_progress_file):
        """Test that reset doesn't error when no file exists."""
        import quest

        # Ensure file doesn't exist
        if temp_progress_file.exists():
            temp_progress_file.unlink()

        # Should not raise
        quest.reset_progress()


class TestProgressIntegrity:
    """Test progress data structure integrity."""

    def test_completed_levels_is_list(self, temp_progress_file):
        """Test completed_levels is always a list."""
        import quest

        progress = quest.load_progress()
        assert isinstance(progress["completed_levels"], list)

    def test_hints_used_is_dict(self, temp_progress_file):
        """Test hints_used is always a dict."""
        import quest

        progress = quest.load_progress()
        assert isinstance(progress["hints_used"], dict)

    def test_round_trip_preserves_data(self, temp_progress_file):
        """Test that save and load preserves all data."""
        import quest

        original = {
            "completed_levels": [1, 2, 3, 4, 5],
            "hints_used": {"1": 6, "2": 6, "3": 6, "4": 6, "5": 6}
        }
        quest.save_progress(original)
        loaded = quest.load_progress()

        assert loaded == original

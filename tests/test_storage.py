"""Tests for sopilot.storage — ensure_directories."""

from __future__ import annotations

from conftest import make_test_settings
from sopilot.storage import ensure_directories


class TestEnsureDirectories:
    def test_creates_all_directories(self, tmp_path):
        settings = make_test_settings(data_dir=tmp_path / "test_data")
        ensure_directories(settings)
        assert settings.data_dir.is_dir()
        assert settings.raw_dir.is_dir()
        assert settings.embeddings_dir.is_dir()
        assert settings.reports_dir.is_dir()
        assert settings.index_dir.is_dir()
        assert settings.models_dir.is_dir()

    def test_idempotent(self, tmp_path):
        settings = make_test_settings(data_dir=tmp_path / "test_data")
        ensure_directories(settings)
        ensure_directories(settings)  # should not raise
        assert settings.data_dir.is_dir()

    def test_pre_existing_partial(self, tmp_path):
        """Some dirs exist, others don't."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        (data_dir / "raw").mkdir()
        settings = make_test_settings(data_dir=data_dir)
        ensure_directories(settings)
        assert settings.embeddings_dir.is_dir()
        assert settings.reports_dir.is_dir()
        assert settings.index_dir.is_dir()
        assert settings.models_dir.is_dir()

    def test_nested_parent_creation(self, tmp_path):
        """Deeply nested path should work (parents=True)."""
        deep = tmp_path / "a" / "b" / "c" / "data"
        settings = make_test_settings(data_dir=deep)
        ensure_directories(settings)
        assert deep.is_dir()

    def test_directory_count(self, tmp_path):
        """Exactly 6 directories should be created."""
        data_dir = tmp_path / "test_data"
        settings = make_test_settings(data_dir=data_dir)
        ensure_directories(settings)
        expected_dirs = {
            settings.data_dir,
            settings.raw_dir,
            settings.embeddings_dir,
            settings.reports_dir,
            settings.index_dir,
            settings.models_dir,
        }
        for d in expected_dirs:
            assert d.is_dir(), f"Expected {d} to be a directory"

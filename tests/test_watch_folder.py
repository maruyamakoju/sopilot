from __future__ import annotations

from pathlib import Path, PurePosixPath

from sopilot.watch_folder import _derive_task_role, _is_hidden_path


class TestIsHiddenPath:
    def test_hidden_file(self) -> None:
        root = Path("/inbox")
        assert _is_hidden_path(Path("/inbox/.hidden/video.mp4"), root) is True

    def test_hidden_nested(self) -> None:
        root = Path("/inbox")
        assert _is_hidden_path(Path("/inbox/task/.processed/video.mp4"), root) is True

    def test_not_hidden(self) -> None:
        root = Path("/inbox")
        assert _is_hidden_path(Path("/inbox/task/gold/video.mp4"), root) is False

    def test_root_level_file(self) -> None:
        root = Path("/inbox")
        assert _is_hidden_path(Path("/inbox/video.mp4"), root) is False


class TestDeriveTaskRole:
    def test_single_file_uses_defaults(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/video.mp4"),
            default_task_id="default_task",
            default_role="trainee",
        )
        assert task_id == "default_task"
        assert role == "trainee"

    def test_task_and_role_from_directory(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/filter_swap/gold/video.mp4"),
            default_task_id="default_task",
            default_role="trainee",
        )
        assert task_id == "filter_swap"
        assert role == "gold"

    def test_task_only_from_directory(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/filter_swap/video.mp4"),
            default_task_id="default_task",
            default_role="trainee",
        )
        assert task_id == "filter_swap"
        assert role == "trainee"

    def test_audit_role(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/task1/audit/clip.mp4"),
            default_task_id="default",
            default_role="trainee",
        )
        assert task_id == "task1"
        assert role == "audit"

    def test_invalid_role_defaults_to_trainee(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/task1/invalid_role/video.mp4"),
            default_task_id="default",
            default_role="invalid_role",
        )
        assert role == "trainee"

    def test_deep_nesting_with_role_in_third_position(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/task1/subdir/gold/video.mp4"),
            default_task_id="default",
            default_role="trainee",
        )
        assert task_id == "task1"
        assert role == "gold"

    def test_strips_whitespace_from_task_id(self) -> None:
        root = Path("/inbox")
        task_id, role = _derive_task_role(
            root=root,
            source_path=Path("/inbox/my_task/gold/video.mp4"),
            default_task_id="  default  ",
            default_role="trainee",
        )
        assert task_id == "my_task"

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from sopilot.utils import normalize_rows, now_tag, safe_filename, write_json


def test_normalize_rows_unit_length() -> None:
    mat = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normed = normalize_rows(mat)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_normalize_rows_zero_vector_stays_bounded() -> None:
    mat = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normed = normalize_rows(mat)
    assert np.all(np.isfinite(normed))
    assert float(np.linalg.norm(normed)) < 1.0


def test_normalize_rows_preserves_direction() -> None:
    mat = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    normed = normalize_rows(mat)
    np.testing.assert_allclose(normed[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normed[1], [0.0, 1.0], atol=1e-6)


def test_safe_filename_removes_special_chars() -> None:
    assert safe_filename("hello world!@#.mp4") == "hello_world___.mp4"


def test_safe_filename_truncates_long_names() -> None:
    long_name = "a" * 200 + ".mp4"
    result = safe_filename(long_name)
    assert len(result) <= 120


def test_safe_filename_uses_fallback_for_empty() -> None:
    assert safe_filename("") == "video.mp4"
    assert safe_filename("", fallback="upload.bin") == "upload.bin"


def test_safe_filename_preserves_valid_chars() -> None:
    assert safe_filename("my_video-2024.mp4") == "my_video-2024.mp4"


def test_now_tag_format() -> None:
    tag = now_tag()
    assert tag.endswith("Z")
    assert "T" in tag
    assert len(tag) == 16  # YYYYMMDDTHHMMSSz


def test_write_json_creates_file_and_parents() -> None:
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "sub" / "deep" / "data.json"
        write_json(target, {"key": "value", "num": 42})
        assert target.exists()
        import json

        content = json.loads(target.read_text(encoding="utf-8"))
        assert content["key"] == "value"
        assert content["num"] == 42


def test_write_json_overwrites_existing() -> None:
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "data.json"
        write_json(target, {"version": 1})
        write_json(target, {"version": 2})
        import json

        content = json.loads(target.read_text(encoding="utf-8"))
        assert content["version"] == 2

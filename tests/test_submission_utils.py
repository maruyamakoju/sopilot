from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from artifacts import verify_submission_bundle
from artifacts.submission_utils import (
    extract_sha256,
    latest_subdir,
    read_json_dict,
    read_sha256,
    resolve_path,
    sha256_file,
    zip_entries,
)


class SubmissionUtilsTests(unittest.TestCase):
    def test_resolve_path_handles_relative_absolute_and_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = Path(tmp_dir).resolve()
            rel = resolve_path(ws, "a/b.txt")
            self.assertEqual(rel, (ws / "a/b.txt").resolve())

            absolute = (ws / "x.txt").resolve()
            self.assertEqual(resolve_path(ws, str(absolute)), absolute)
            self.assertIsNone(resolve_path(ws, None))

    def test_sha256_and_sha_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            file_path = root / "sample.txt"
            file_path.write_text("abc", encoding="utf-8")
            digest = sha256_file(file_path)
            self.assertEqual(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

            sha_file = root / "sample.sha256"
            sha_file.write_text(f"{digest}  sample.txt\n", encoding="utf-8")
            self.assertEqual(read_sha256(sha_file), digest)
            self.assertEqual(extract_sha256(f"sha={digest}"), digest)

    def test_read_json_dict_rejects_non_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            good = root / "good.json"
            good.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(read_json_dict(good), {"ok": True})

            bad = root / "bad.json"
            bad.write_text("not-json", encoding="utf-8")
            self.assertIsNone(read_json_dict(bad))

            as_list = root / "list.json"
            as_list.write_text(json.dumps([1, 2]), encoding="utf-8")
            self.assertIsNone(read_json_dict(as_list))

    def test_zip_entries_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            zip_path = root / "x.zip"
            with zipfile.ZipFile(zip_path, mode="w") as zf:
                zf.writestr("b.txt", "b")
                zf.writestr("a.txt", "a")
            self.assertEqual(zip_entries(zip_path), ["a.txt", "b.txt"])

    def test_latest_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.assertIsNone(latest_subdir(root / "missing"))
            (root / "20260218_010000").mkdir()
            (root / "20260218_020000").mkdir()
            latest = latest_subdir(root)
            self.assertEqual(latest, (root / "20260218_020000"))

    def test_markdown_path_token_filter_ignores_commands(self) -> None:
        text = "\n".join(
            [
                "- run `python scripts/run_backend_ablation.py --summarize-only`",
                "- packet `artifacts/SUBMISSION_PACKET_20260216.md`",
                "- flag `--workspace`",
                "- hash `461fd416f92e73f1752c3b7cb05a3332aba780536b08c9cbf3c9aee96dce4287`",
            ]
        )
        tokens = verify_submission_bundle._path_like_tokens_from_markdown(text)
        self.assertEqual(tokens, ["artifacts/SUBMISSION_PACKET_20260216.md"])


if __name__ == "__main__":
    unittest.main()

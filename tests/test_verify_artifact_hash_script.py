from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_artifact_hash import _verify_one
from sopilot.eval.integrity import attach_payload_hash


class VerifyArtifactHashScriptTests(unittest.TestCase):
    def test_verify_one_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.json"
            payload = attach_payload_hash({"version": "split_manifest_v1", "split_job_ids": {"dev": [], "test": [], "challenge": []}})
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            row = _verify_one(path)
            self.assertTrue(row["verified"])

    def test_verify_one_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.json"
            payload = attach_payload_hash({"version": "split_manifest_v1", "split_job_ids": {"dev": [], "test": [], "challenge": []}})
            payload["split_job_ids"]["dev"] = [1]
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            row = _verify_one(path)
            self.assertFalse(row["verified"])


if __name__ == "__main__":
    unittest.main()

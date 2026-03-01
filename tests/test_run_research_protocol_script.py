from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_research_protocol import _cmd_text, _run_step


class RunResearchProtocolScriptTests(unittest.TestCase):
    def test_cmd_text_quotes_spaces(self) -> None:
        text = _cmd_text(["python", "scripts/run.py", "C:\\tmp folder\\x.json"])
        self.assertIn('"C:\\tmp folder\\x.json"', text)

    def test_run_step_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir = Path(tmp_dir) / "logs"
            row = _run_step(
                name="dry_step",
                cmd=["python", "--version"],
                logs_dir=logs_dir,
                dry_run=True,
            )
            self.assertEqual(row["return_code"], 0)
            self.assertTrue((logs_dir / "dry_step.stdout.txt").exists())
            self.assertTrue((logs_dir / "dry_step.stderr.txt").exists())


if __name__ == "__main__":
    unittest.main()

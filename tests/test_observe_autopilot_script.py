from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.observe_autopilot import snapshot_once


class ObserveAutopilotScriptTests(unittest.TestCase):
    def test_snapshot_once_writes_status_and_copies_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            autopilot_dir = data_dir / "autopilot"
            observations_dir = autopilot_dir / "observations"
            autopilot_dir.mkdir(parents=True, exist_ok=True)

            (autopilot_dir / "heartbeat.json").write_text(
                json.dumps({"run_id": "run-1", "state": "cycle_done", "ts": 123.0}),
                encoding="utf-8",
            )
            (autopilot_dir / "latest_cycle.json").write_text(
                json.dumps({"run_id": "run-1", "cycle": 1, "ok": True}),
                encoding="utf-8",
            )
            (data_dir / "gate_report.json").write_text(
                json.dumps({"gates": {"overall_pass": True}}),
                encoding="utf-8",
            )

            payload = snapshot_once(data_dir=data_dir, observations_dir=observations_dir, stale_seconds=1800.0)

            status_path = Path(payload["status_path"])
            self.assertTrue(status_path.exists())
            self.assertIn("gate_report.json", payload["copied"])
            latest_observation = observations_dir / "latest_observation.json"
            self.assertTrue(latest_observation.exists())


if __name__ == "__main__":
    unittest.main()


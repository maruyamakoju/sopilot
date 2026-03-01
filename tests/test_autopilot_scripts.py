import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.autopilot_common import process_exists, runner_is_active, write_json
from scripts.autopilot_status import collect_status
from scripts.poc_autopilot_runner import sync_runner_metadata
from scripts.start_autopilot_detached import main as start_main
from scripts.start_observer_detached import main as start_observer_main
from scripts.stop_autopilot import main as stop_main
from scripts.stop_autopilot import request_stop, resolve_pid_file_path


class TestAutopilotScripts(unittest.TestCase):
    def test_write_json_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "x" / "state.json"
            write_json(target, {"k": "v"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"k": "v"})
            leftovers = list((root / "x").glob(".state.json.*.tmp"))
            self.assertEqual(leftovers, [])

    def test_process_exists_current_pid(self) -> None:
        self.assertTrue(process_exists(os.getpid()))

    def test_runner_is_active_false_when_stopped(self) -> None:
        payload = {"pid": os.getpid(), "stopped": True}
        self.assertFalse(runner_is_active(payload))

    def test_collect_status_runner_state_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            autopilot = root / "autopilot"
            autopilot.mkdir(parents=True, exist_ok=True)
            (autopilot / "runner.json").write_text(
                json.dumps({"pid": 99999999, "started_at": "2026-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )
            (root / "gate_report.json").write_text(
                json.dumps({"gates": {"overall_pass": True}, "num_completed_jobs": 10}),
                encoding="utf-8",
            )
            payload = collect_status(root)
            self.assertFalse(payload["runner_active"])
            self.assertEqual(payload["runner_state"], "not_running")

    def test_collect_status_runner_state_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            autopilot = root / "autopilot"
            autopilot.mkdir(parents=True, exist_ok=True)
            (autopilot / "runner.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "stopped": True,
                    }
                ),
                encoding="utf-8",
            )
            payload = collect_status(root)
            self.assertFalse(payload["runner_active"])
            self.assertEqual(payload["runner_state"], "stopped")

    def test_collect_status_run_id_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            autopilot = root / "autopilot"
            autopilot.mkdir(parents=True, exist_ok=True)
            run_id = "run-abc"
            (autopilot / "runner.json").write_text(
                json.dumps({"pid": 99999999, "run_id": run_id}),
                encoding="utf-8",
            )
            (autopilot / "heartbeat.json").write_text(
                json.dumps({"run_id": run_id, "state": "cycle_done"}),
                encoding="utf-8",
            )
            payload = collect_status(root)
            self.assertEqual(payload["runner_run_id"], run_id)
            self.assertEqual(payload["heartbeat_run_id"], run_id)
            self.assertTrue(payload["run_id_match"])

    def test_collect_status_heartbeat_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            autopilot = root / "autopilot"
            autopilot.mkdir(parents=True, exist_ok=True)
            (autopilot / "heartbeat.json").write_text(
                json.dumps({"run_id": "run-abc", "state": "running_step", "ts": 1.0}),
                encoding="utf-8",
            )
            payload = collect_status(root, stale_seconds=1.0)
            self.assertTrue(payload["heartbeat_stale"])
            self.assertEqual(payload["lifecycle_state"], "stale")

    def test_collect_status_finished_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            autopilot = root / "autopilot"
            autopilot.mkdir(parents=True, exist_ok=True)
            run_id = "run-finished"
            (autopilot / "runner.json").write_text(
                json.dumps({"pid": 99999999, "run_id": run_id}),
                encoding="utf-8",
            )
            (autopilot / "final_summary.json").write_text(
                json.dumps({"run_id": run_id, "cycles": 3}),
                encoding="utf-8",
            )
            payload = collect_status(root, stale_seconds=1.0)
            self.assertEqual(payload["lifecycle_state"], "finished")
            self.assertEqual(payload["recommended_action"], "collect_artifacts")

    def test_start_blocks_if_runner_already_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            autopilot = root / "autopilot"
            autopilot.mkdir(parents=True, exist_ok=True)
            (autopilot / "runner.json").write_text(
                json.dumps({"pid": os.getpid(), "started_at": "2026-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )
            argv = [
                "start_autopilot_detached.py",
                "--task-id",
                "task-a",
                "--base-dir",
                "poc_videos",
                "--trainee-dir",
                "poc_videos/trainee",
                "--trainee-bad-dir",
                "poc_videos/trainee_bad",
                "--data-dir",
                str(root),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with self.assertRaises(SystemExit) as ctx:
                        start_main()
            self.assertIn("already running", str(ctx.exception))

    def test_stop_rejects_invalid_pid_file_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "runner.json"
            pid_file.write_text("{}", encoding="utf-8")
            argv = [
                "stop_autopilot.py",
                "--pid-file",
                str(pid_file),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with self.assertRaises(SystemExit) as ctx:
                        stop_main()
            self.assertIn("invalid pid-file json", str(ctx.exception))

    def test_stop_requires_exactly_one_pid_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pid_file = root / "autopilot" / "runner.json"
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(json.dumps({"pid": 99999999, "run_id": "abc"}), encoding="utf-8")
            argv = [
                "stop_autopilot.py",
                "--pid-file",
                str(pid_file),
                "--data-dir",
                str(root),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with self.assertRaises(SystemExit) as ctx:
                        stop_main()
            self.assertIn("exactly one", str(ctx.exception))

    def test_resolve_pid_file_path_from_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir).resolve()
            result = resolve_pid_file_path(None, str(root))
            self.assertEqual(result, root / "autopilot" / "runner.json")

    def test_request_stop_force_fallback(self) -> None:
        with mock.patch("scripts.stop_autopilot.process_exists", side_effect=[True, True, True, False]):
            with mock.patch("scripts.stop_autopilot._soft_terminate") as soft:
                with mock.patch("scripts.stop_autopilot._force_terminate") as force:
                    stopped, error, method = request_stop(1234, force=False, grace_seconds=0.0)
        soft.assert_called_once_with(1234)
        force.assert_called_once_with(1234)
        self.assertTrue(stopped)
        self.assertIsNone(error)
        self.assertEqual(method, "force_fallback")

    def test_stop_rejects_when_require_run_id_without_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "runner.json"
            pid_file.write_text(json.dumps({"pid": 99999999, "run_id": "abc"}), encoding="utf-8")
            argv = [
                "stop_autopilot.py",
                "--pid-file",
                str(pid_file),
                "--require-run-id",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with self.assertRaises(SystemExit) as ctx:
                        stop_main()
            self.assertIn("--require-run-id", str(ctx.exception))

    def test_stop_rejects_when_run_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "runner.json"
            pid_file.write_text(json.dumps({"pid": 99999999, "run_id": "abc"}), encoding="utf-8")
            argv = [
                "stop_autopilot.py",
                "--pid-file",
                str(pid_file),
                "--run-id",
                "other",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with self.assertRaises(SystemExit) as ctx:
                        stop_main()
            self.assertIn("run_id mismatch", str(ctx.exception))

    def test_stop_accepts_matching_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "runner.json"
            pid_file.write_text(json.dumps({"pid": 99999999, "run_id": "abc"}), encoding="utf-8")
            argv = [
                "stop_autopilot.py",
                "--pid-file",
                str(pid_file),
                "--run-id",
                "abc",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with mock.patch("scripts.stop_autopilot.process_exists", return_value=False):
                        with mock.patch("scripts.stop_autopilot.write_json") as write_json_mock:
                            with redirect_stdout(io.StringIO()):
                                stop_main()
            write_json_mock.assert_called_once()

    def test_start_writes_runner_cmd_with_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            argv = [
                "start_autopilot_detached.py",
                "--task-id",
                "task-a",
                "--task-name",
                "Task A",
                "--base-dir",
                "poc_videos",
                "--trainee-dir",
                "poc_videos/trainee",
                "--trainee-bad-dir",
                "poc_videos/trainee_bad",
                "--data-dir",
                str(root),
                "--run-id",
                "fixed-run-id",
            ]
            fake_proc = mock.Mock()
            fake_proc.pid = 12345
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with mock.patch("scripts.start_autopilot_detached.subprocess.Popen", return_value=fake_proc) as popen:
                        with redirect_stdout(io.StringIO()):
                            start_main()

            popen.assert_called_once()
            runner_path = root / "autopilot" / "runner.json"
            payload = json.loads(runner_path.read_text(encoding="utf-8"))
            cmd = payload["cmd"]
            self.assertEqual(payload["run_id"], "fixed-run-id")
            self.assertEqual(cmd[0], sys.executable)
            self.assertTrue(Path(cmd[1]).name == "poc_autopilot_runner.py")
            self.assertIn("--run-id", cmd)
            self.assertEqual(cmd[cmd.index("--run-id") + 1], "fixed-run-id")
            self.assertIn("--step-timeout-minutes", cmd)
            self.assertEqual(cmd[cmd.index("--step-timeout-minutes") + 1], "30.0")

    def test_start_writes_runner_cmd_with_custom_step_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            argv = [
                "start_autopilot_detached.py",
                "--task-id",
                "task-a",
                "--base-dir",
                "poc_videos",
                "--trainee-dir",
                "poc_videos/trainee",
                "--trainee-bad-dir",
                "poc_videos/trainee_bad",
                "--data-dir",
                str(root),
                "--step-timeout-minutes",
                "12.5",
            ]
            fake_proc = mock.Mock()
            fake_proc.pid = 5555
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with mock.patch("scripts.start_autopilot_detached.subprocess.Popen", return_value=fake_proc):
                        with redirect_stdout(io.StringIO()):
                            start_main()

            payload = json.loads((root / "autopilot" / "runner.json").read_text(encoding="utf-8"))
            cmd = payload["cmd"]
            self.assertIn("--step-timeout-minutes", cmd)
            self.assertEqual(cmd[cmd.index("--step-timeout-minutes") + 1], "12.5")

    def test_start_observer_writes_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            argv = [
                "start_observer_detached.py",
                "--data-dir",
                str(root),
                "--interval-seconds",
                "60",
                "--max-iterations",
                "3",
            ]
            fake_proc = mock.Mock()
            fake_proc.pid = 7777
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
                    with mock.patch(
                        "scripts.start_observer_detached.subprocess.Popen", return_value=fake_proc
                    ) as popen:
                        with redirect_stdout(io.StringIO()):
                            start_observer_main()

            popen.assert_called_once()
            pid_path = root / "autopilot" / "observer.json"
            self.assertTrue(pid_path.exists())
            payload = json.loads(pid_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pid"], 7777)
            self.assertTrue(Path(payload["cmd"][1]).name == "observe_autopilot.py")

    def test_sync_runner_metadata_replaces_launcher_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runner_path = root / "autopilot" / "runner.json"
            runner_path.parent.mkdir(parents=True, exist_ok=True)
            runner_path.write_text(
                json.dumps(
                    {
                        "pid": 11111,
                        "run_id": "old-run",
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "cmd": ["python", "old_runner.py"],
                    }
                ),
                encoding="utf-8",
            )

            payload = sync_runner_metadata(
                runner_path,
                run_id="new-run",
                data_dir=root,
                log_file=root / "autopilot" / "runner.log",
                current_pid=22222,
            )
            on_disk = json.loads(runner_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pid"], 22222)
            self.assertEqual(on_disk["pid"], 22222)
            self.assertEqual(on_disk["launcher_pid"], 11111)
            self.assertEqual(on_disk["run_id"], "new-run")
            self.assertEqual(on_disk["started_at"], "2026-01-01T00:00:00+00:00")
            self.assertEqual(on_disk["pid_source"], "runner")


if __name__ == "__main__":
    unittest.main()

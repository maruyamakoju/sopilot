from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def now_ts() -> float:
    return time.time()


def autopilot_dir(data_dir: Path) -> Path:
    return data_dir / "autopilot"


def default_runner_pid_file(data_dir: Path) -> Path:
    return autopilot_dir(data_dir) / "runner.json"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write to reduce risk of partial files on interruption.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) can be unreliable for detached Windows processes.
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            # Access denied still indicates the process exists.
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but we may not have permission to signal it.
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False
    except Exception:
        return False


def runner_pid(payload: dict | None) -> int | None:
    if not payload:
        return None
    try:
        pid = int(payload.get("pid"))
    except Exception:
        return None
    return pid if pid > 0 else None


def runner_run_id(payload: dict | None) -> str | None:
    if not payload:
        return None
    value = payload.get("run_id")
    if value is None:
        return None
    run_id = str(value).strip()
    return run_id or None


def runner_is_stopped(payload: dict | None) -> bool:
    if not payload:
        return False
    return bool(payload.get("stopped"))


def runner_is_active(payload: dict | None) -> bool:
    pid = runner_pid(payload)
    if pid is None:
        return False
    if runner_is_stopped(payload):
        return False
    return process_exists(pid)


def heartbeat_age_seconds(payload: dict | None, *, now_epoch: float | None = None) -> float | None:
    if not payload:
        return None
    value = payload.get("ts")
    try:
        ts = float(value)
    except Exception:
        return None
    now_value = float(now_epoch if now_epoch is not None else now_ts())
    if now_value < ts:
        return 0.0
    return now_value - ts

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore SOPilot data from backup bundle.")
    parser.add_argument("--backup", required=True, help="Backup zip path")
    parser.add_argument("--data-dir", default="data", help="Target SOPilot data directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    backup_path = Path(args.backup).resolve()
    if not backup_path.exists():
        raise SystemExit(f"Backup not found: {backup_path}")

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        with zipfile.ZipFile(backup_path, "r") as zf:
            zf.extractall(tmp)

        src_db = tmp / "sopilot.db"
        if src_db.exists():
            dst_db = data_dir / "sopilot.db"
            if dst_db.exists() and not args.force:
                raise SystemExit(f"Database exists: {dst_db}. Use --force to overwrite.")
            shutil.copy2(src_db, dst_db)

        src_raw = tmp / "raw"
        if src_raw.exists():
            for path in src_raw.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(src_raw)
                dst = raw_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and not args.force:
                    raise SystemExit(f"File exists: {dst}. Use --force to overwrite.")
                shutil.copy2(path, dst)

    print(f"restored to {data_dir}")


if __name__ == "__main__":
    main()


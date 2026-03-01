from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SOPilot on-prem backup bundle.")
    parser.add_argument("--data-dir", default="data", help="SOPilot data directory")
    parser.add_argument("--out-dir", default="backups", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = out_dir / f"sopilot-backup-{ts}.zip"
    manifest = {
        "created_at_utc": ts,
        "data_dir": str(data_dir),
        "files": [],
    }

    with zipfile.ZipFile(backup_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        db_path = data_dir / "sopilot.db"
        if db_path.exists():
            zf.write(db_path, arcname="sopilot.db")
            manifest["files"].append("sopilot.db")
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            for path in raw_dir.rglob("*"):
                if path.is_file():
                    arc = Path("raw") / path.relative_to(raw_dir)
                    zf.write(path, arcname=str(arc))
                    manifest["files"].append(str(arc))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(str(backup_path))


if __name__ == "__main__":
    main()


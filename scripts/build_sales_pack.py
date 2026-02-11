from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil


def _slug(s: str) -> str:
    x = s.strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_") or "company"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a send-ready SOPilot sales pack")
    parser.add_argument("--company", required=True, help="Company name for pack folder")
    parser.add_argument("--source-offer", default="docs/paid_poc_offer_example_plant_a.md")
    parser.add_argument("--demo-artifacts-dir", default="demo_artifacts")
    parser.add_argument("--out-root", default="sales_pack")
    parser.add_argument(
        "--include-all-artifacts",
        action="store_true",
        help="Include all score/audit artifacts from demo directory (default: latest set only)",
    )
    args = parser.parse_args()

    today = datetime.now().strftime("%Y%m%d")
    company_slug = _slug(args.company)
    pack_dir = Path(args.out_root) / f"{today}_{company_slug}"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        ("docs/sopilot_concept_onepager.md", "01_sopilot_concept_onepager.md"),
        ("docs/paid_poc_fixed_package.md", "02_paid_poc_fixed_package.md"),
        (args.source_offer, "03_paid_poc_offer_for_customer.md"),
        ("docs/infosec_onepager.md", "04_infosec_onepager.md"),
        ("docs/field_capture_guide.md", "05_field_capture_guide.md"),
        ("docs/task_definition_sheet_template.md", "06_task_definition_sheet_template.md"),
        ("docs/data_handling_spec.md", "07_data_handling_spec.md"),
        ("docs/demo_recording_script_3min.md", "08_demo_recording_script_3min.md"),
        ("sales/outreach_email_templates.md", "09_outreach_email_templates.md"),
    ]
    for src_raw, dst_name in docs:
        _copy_if_exists(Path(src_raw), pack_dir / dst_name)

    artifacts_dir = Path(args.demo_artifacts_dir)
    if artifacts_dir.exists():
        score_jsons = sorted(
            artifacts_dir.glob("score_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if args.include_all_artifacts:
            selected_score_jsons = score_jsons
        else:
            selected_score_jsons = score_jsons[:1]

        for p in selected_score_jsons:
            _copy_if_exists(p, pack_dir / p.name)
            pdf = p.with_suffix(".pdf")
            _copy_if_exists(pdf, pack_dir / pdf.name)

        audit_exports = sorted(
            artifacts_dir.glob("audit_export_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if args.include_all_artifacts:
            selected_audit_exports = audit_exports
        else:
            selected_audit_exports = audit_exports[:1]
        for p in selected_audit_exports:
            _copy_if_exists(p, pack_dir / p.name)

        _copy_if_exists(artifacts_dir / "queue_metrics_latest.json", pack_dir / "queue_metrics_latest.json")
        _copy_if_exists(artifacts_dir / "audit_trail_latest.json", pack_dir / "audit_trail_latest.json")
        _copy_if_exists(artifacts_dir / "release_gate_report.json", pack_dir / "release_gate_report.json")

    readme = pack_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                f"SOPilot Sales Pack for: {args.company}",
                "",
                "Contents:",
                "- One-page concept",
                "- Fixed paid PoC package",
                "- Customer offer draft",
                "- InfoSec one-pager",
                "- Capture guide / task sheet / data handling",
                "- Demo script (3 min)",
                "- Outreach templates",
                "- Latest demo artifacts (PDF/JSON/audit)",
                "- Signed audit export and queue metrics",
                "",
                "Next action:",
                "1) Edit 03_paid_poc_offer_for_customer.md with real company details",
                "2) Attach 3-min demo recording",
                "3) Send to training + QA/audit + maintenance contacts",
            ]
        ),
        encoding="utf-8",
    )

    print(str(pack_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

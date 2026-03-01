"""PDF audit report generation for score jobs.

Uses fpdf2 to generate a structured A4 PDF report with the same
data as the HTML report. Produces a self-contained PDF with
Japanese text support.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from sopilot import __version__

# Font directory for bundled fonts
_FONT_DIR = Path(__file__).resolve().parent / "fonts"

# ── Decision colour map (RGB tuples) ─────────────────────────────────────
_DEC_RGB: dict[str, tuple[int, int, int]] = {
    "pass": (16, 185, 129),
    "fail": (239, 68, 68),
    "needs_review": (59, 130, 246),
    "retrain": (245, 158, 11),
}
_DEC_JP: dict[str, str] = {
    "pass": "合格", "fail": "不合格",
    "needs_review": "要確認", "retrain": "再研修",
}
_SEV_RGB: dict[str, tuple[int, int, int]] = {
    "critical": (239, 68, 68),
    "quality": (245, 158, 11),
    "efficiency": (59, 130, 246),
}
_SEV_JP: dict[str, str] = {"critical": "重大", "quality": "品質", "efficiency": "効率"}
_TYPE_JP: dict[str, str] = {
    "missing_step": "手順省略", "step_deviation": "品質逸脱",
    "order_swap": "手順入替", "over_time": "時間超過",
}


def _ensure_font(pdf: Any) -> None:
    """Register a CJK-capable font if available, otherwise use built-in.

    Safe to call multiple times — skips registration if already loaded.
    """
    # If we already registered a font, just re-set it
    if hasattr(pdf, "_sopilot_font"):
        pdf.set_font(pdf._sopilot_font, "", 10)
        return

    try:
        from fpdf import FPDF  # noqa: F401

        # Try bundled NotoSansJP font
        font_path = _FONT_DIR / "NotoSansJP-Regular.ttf"
        font_bold_path = _FONT_DIR / "NotoSansJP-Bold.ttf"

        if font_path.exists():
            pdf.add_font("NotoJP", "", str(font_path))
            if font_bold_path.exists():
                pdf.add_font("NotoJP", "B", str(font_bold_path))
            else:
                pdf.add_font("NotoJP", "B", str(font_path))
            pdf.set_font("NotoJP", "", 10)
            pdf._sopilot_font = "NotoJP"
            return

        # Fallback: try system fonts
        import platform
        if platform.system() == "Windows":
            for name in ["msgothic.ttc", "meiryo.ttc", "YuGothM.ttc"]:
                sys_font = Path("C:/Windows/Fonts") / name
                if sys_font.exists():
                    pdf.add_font("SysJP", "", str(sys_font))
                    pdf.add_font("SysJP", "B", str(sys_font))
                    pdf.set_font("SysJP", "", 10)
                    pdf._sopilot_font = "SysJP"
                    return

        # Linux: try common CJK fonts
        for p in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        ]:
            if Path(p).exists():
                pdf.add_font("SysJP", "", p)
                pdf.add_font("SysJP", "B", p)
                pdf.set_font("SysJP", "", 10)
                pdf._sopilot_font = "SysJP"
                return
    except Exception:
        pass

    # Ultimate fallback: Helvetica (no Japanese support)
    pdf.set_font("Helvetica", "", 10)
    pdf._sopilot_font = "Helvetica"


def build_report_pdf(job_id: int, job: dict[str, Any]) -> bytes:
    """Generate a PDF audit report and return raw bytes."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError(
            "fpdf2 is required for PDF report generation. "
            "Install it with: pip install fpdf2"
        ) from exc

    result = job.get("result") or {}
    review_raw = job.get("review") or {}
    score_raw = result.get("score")
    score_s = f"{score_raw:.1f}" if score_raw is not None else "—"
    summary = result.get("summary") or {}
    metrics = result.get("metrics") or {}
    deviations_raw: list[dict[str, Any]] = result.get("deviations") or []
    decision = summary.get("decision", "—")
    sev_counts = summary.get("severity_counts") or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dec_rgb = _DEC_RGB.get(decision, (148, 163, 184))
    decision_jp = _DEC_JP.get(decision, decision)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _ensure_font(pdf)
    font_family = pdf.font_family

    # ── Helper functions ──────────────────────────────────────────────

    def set_bold(size: float = 10) -> None:
        pdf.set_font(font_family, "B", size)

    def set_normal(size: float = 10) -> None:
        pdf.set_font(font_family, "", size)

    def color(r: int, g: int, b: int) -> None:
        pdf.set_text_color(r, g, b)

    def gray() -> None:
        color(100, 116, 139)

    def black() -> None:
        color(30, 41, 59)

    def section_title(title: str) -> None:
        pdf.ln(6)
        set_bold(12)
        color(71, 85, 105)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(226, 232, 240)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

    # ── Header ────────────────────────────────────────────────────────
    # Logo + title
    set_bold(16)
    color(13, 148, 136)
    pdf.cell(30, 10, "SOPilot", new_x="END")
    set_normal(9)
    gray()
    pdf.cell(20, 10, f"v{__version__}", new_x="END")
    pdf.ln(10)

    set_bold(18)
    black()
    pdf.cell(0, 10, "SOP 遵守評価報告書", new_x="LMARGIN", new_y="NEXT")

    set_normal(9)
    gray()
    pdf.cell(0, 5, f"Job #{job_id}  |  出力日時: {now}", new_x="LMARGIN", new_y="NEXT")

    # Accent line
    pdf.set_draw_color(13, 148, 136)
    pdf.set_line_width(0.8)
    pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.w - pdf.r_margin, pdf.get_y() + 2)
    pdf.ln(6)

    # ── Score + Decision (right-aligned big number) ───────────────────
    y_start = pdf.get_y()
    set_bold(42)
    color(*dec_rgb)
    # Right-align score
    score_w = pdf.get_string_width(score_s)
    pdf.set_xy(pdf.w - pdf.r_margin - score_w - 5, y_start)
    pdf.cell(score_w + 5, 16, score_s, align="R")

    set_normal(9)
    gray()
    pdf.set_xy(pdf.w - pdf.r_margin - 40, y_start + 16)
    pdf.cell(40, 5, "総合スコア", align="R")

    # Decision pill
    set_bold(12)
    color(*dec_rgb)
    pill_w = pdf.get_string_width(decision_jp) + 16
    pdf.set_xy(pdf.w - pdf.r_margin - pill_w, y_start + 22)
    pdf.set_fill_color(dec_rgb[0], dec_rgb[1], dec_rgb[2])
    pdf.set_text_color(255, 255, 255)
    pdf.cell(pill_w, 8, decision_jp, align="C", fill=True)
    black()

    pdf.set_y(y_start + 34)

    # ── Job Info ──────────────────────────────────────────────────────
    section_title("評価基本情報")

    info_items = [
        ("タスク", str(result.get("task_id", "—") or "—")),
        ("Gold 動画 ID", f"#{result.get('gold_video_id', '—')}"),
        ("研修生動画 ID", f"#{result.get('trainee_video_id', '—')}"),
        ("判定理由", str(summary.get("decision_reason", "") or "")),
    ]

    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    for i, (label, value) in enumerate(info_items):
        x = pdf.l_margin + (i % 2) * col_w
        y = pdf.get_y()
        if i % 2 == 0 and i > 0:
            pdf.ln(10)
            y = pdf.get_y()

        pdf.set_xy(x, y)
        set_normal(8)
        gray()
        pdf.cell(col_w - 4, 4, label, new_x="LMARGIN")

        pdf.set_xy(x, y + 4)
        set_bold(10)
        black()
        # Truncate long values
        val_display = value[:60] + "..." if len(value) > 60 else value
        pdf.cell(col_w - 4, 5, val_display, new_x="LMARGIN")

    pdf.ln(12)

    # ── Metrics Table ─────────────────────────────────────────────────
    section_title("評価メトリクス")

    miss = int(metrics.get("miss_steps", 0) or 0)
    swap = int(metrics.get("swap_steps", 0) or 0)
    dev = int(metrics.get("deviation_steps", 0) or 0)
    ot = float(metrics.get("over_time_ratio", 0.0) or 0.0)
    dtw_v = metrics.get("dtw_normalized_cost")

    metric_rows = [
        ("総合スコア", score_s, f"合格 >= {summary.get('pass_score', '—')} pt", dec_rgb),
        ("手順遺漏数 (Miss)", str(miss), "0 が理想", (239, 68, 68) if miss > 0 else (16, 185, 129)),
        ("手順入替数 (Swap)", str(swap), "0 が理想", (245, 158, 11) if swap > 0 else (16, 185, 129)),
        ("品質逸脱数 (Dev)", str(dev), "0 が理想", (245, 158, 11) if dev > 0 else (16, 185, 129)),
        ("時間超過率", f"{ot:.3f}", "<= 0.200", (59, 130, 246) if ot > 0.2 else (16, 185, 129)),
        ("DTW 正規化コスト", f"{dtw_v:.4f}" if dtw_v is not None else "—", "参考値", (30, 41, 59)),
    ]

    # Table header
    table_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_widths = [table_w * 0.35, table_w * 0.20, table_w * 0.25, table_w * 0.20]

    pdf.set_fill_color(248, 250, 252)
    set_bold(8)
    gray()
    headers = ["指標", "値", "基準", "判定"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    set_normal(9)
    for label, value, standard, val_rgb in metric_rows:
        black()
        pdf.cell(col_widths[0], 7, label, border=1)
        color(*val_rgb)
        set_bold(9)
        pdf.cell(col_widths[1], 7, value, border=1, align="C")
        set_normal(8)
        gray()
        pdf.cell(col_widths[2], 7, standard, border=1, align="C")
        if val_rgb == (16, 185, 129):
            color(16, 185, 129)
            pdf.cell(col_widths[3], 7, "OK", border=1, align="C")
        elif val_rgb == (30, 41, 59):
            gray()
            pdf.cell(col_widths[3], 7, "—", border=1, align="C")
        else:
            color(*val_rgb)
            pdf.cell(col_widths[3], 7, "注意", border=1, align="C")
        set_normal(9)
        pdf.ln()

    # ── Severity Summary ──────────────────────────────────────────────
    pdf.ln(4)
    crit_n = int(sev_counts.get("critical", 0) or 0)
    qual_n = int(sev_counts.get("quality", 0) or 0)
    eff_n = int(sev_counts.get("efficiency", 0) or 0)

    sev_box_w = table_w / 3
    sev_items = [
        (crit_n, "Critical", (239, 68, 68) if crit_n > 0 else (16, 185, 129)),
        (qual_n, "Quality", (245, 158, 11) if qual_n > 0 else (16, 185, 129)),
        (eff_n, "Efficiency", (59, 130, 246) if eff_n > 0 else (16, 185, 129)),
    ]

    y_sev = pdf.get_y()
    for i, (count, label, rgb) in enumerate(sev_items):
        x = pdf.l_margin + i * sev_box_w
        pdf.set_xy(x, y_sev)
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(x, y_sev, sev_box_w - 2, 16, style="DF")
        set_bold(18)
        color(*rgb)
        pdf.set_xy(x, y_sev + 1)
        pdf.cell(sev_box_w - 2, 8, str(count), align="C")
        set_normal(7)
        gray()
        pdf.set_xy(x, y_sev + 9)
        pdf.cell(sev_box_w - 2, 5, label, align="C")

    pdf.set_y(y_sev + 20)

    # ── Deviations ────────────────────────────────────────────────────
    section_title(f"逸脱詳細 ({len(deviations_raw)} 件)")

    if not deviations_raw:
        set_bold(10)
        color(16, 185, 129)
        pdf.set_fill_color(240, 253, 244)
        pdf.cell(0, 10, "逸脱なし — 全手順正常", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    else:
        # Deviation table header
        dev_cols = [12, table_w * 0.12, table_w * 0.18, table_w * 0.40, table_w * 0.22]
        # Adjust last column to fill remaining space
        dev_cols[4] = table_w - sum(dev_cols[:4])

        pdf.set_fill_color(248, 250, 252)
        set_bold(7)
        gray()
        dev_headers = ["#", "重大度", "タイプ", "詳細", "タイムコード"]
        for i, h in enumerate(dev_headers):
            pdf.cell(dev_cols[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()

        set_normal(8)
        for idx, dv in enumerate(deviations_raw):
            sev = dv.get("severity", "quality")
            sev_rgb = _SEV_RGB.get(sev, (148, 163, 184))
            sev_jp = _SEV_JP.get(sev, sev)
            type_jp = _TYPE_JP.get(dv.get("type", ""), dv.get("type", ""))
            detail = str(dv.get("detail", "") or "")
            if len(detail) > 50:
                detail = detail[:47] + "..."

            gt = dv.get("gold_timecode") or []
            tt = dv.get("trainee_timecode") or []
            tc = ""
            if len(gt) == 2:
                tc += f"G:{gt[0]}s-{gt[1]}s"
            if len(tt) == 2:
                tc += f" T:{tt[0]}s-{tt[1]}s"

            # Check if we need a new page
            if pdf.get_y() > 260:
                pdf.add_page()
                _ensure_font(pdf)

            gray()
            pdf.cell(dev_cols[0], 6, str(idx + 1), border=1, align="C")
            color(*sev_rgb)
            set_bold(8)
            pdf.cell(dev_cols[1], 6, sev_jp, border=1, align="C")
            set_normal(8)
            black()
            pdf.cell(dev_cols[2], 6, type_jp, border=1)
            set_normal(7)
            pdf.cell(dev_cols[3], 6, detail, border=1)
            set_normal(7)
            gray()
            pdf.cell(dev_cols[4], 6, tc, border=1)
            pdf.ln()

    # ── Review ────────────────────────────────────────────────────────
    if review_raw:
        section_title("評価者記録")

        verdict_jp = _DEC_JP.get(review_raw.get("verdict", ""), review_raw.get("verdict", "—"))
        note = str(review_raw.get("note", "") or "—")
        updated = str(review_raw.get("updated_at", "") or "—")

        review_items = [
            ("評価者判定", verdict_jp),
            ("コメント", note[:80] + "..." if len(note) > 80 else note),
            ("記録日時", updated),
        ]

        for label, value in review_items:
            set_normal(8)
            gray()
            pdf.cell(30, 6, label + ":")
            set_bold(9)
            black()
            pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    # ── Signature Block ───────────────────────────────────────────────
    if pdf.get_y() > 230:
        pdf.add_page()
        _ensure_font(pdf)

    section_title("確認・承認")

    sig_w = (table_w - 10) / 2
    y_sig = pdf.get_y() + 4

    for i, title in enumerate(["評価担当者 氏名", "承認者 氏名"]):
        x = pdf.l_margin + i * (sig_w + 10)
        set_normal(9)
        gray()
        pdf.set_xy(x, y_sig)
        pdf.cell(sig_w, 5, title)
        pdf.set_draw_color(30, 41, 59)
        pdf.line(x, y_sig + 20, x + sig_w, y_sig + 20)
        set_normal(8)
        pdf.set_xy(x, y_sig + 21)
        pdf.cell(sig_w, 4, "署名・日付")

    pdf.set_y(y_sig + 30)

    # ── Footer ────────────────────────────────────────────────────────
    pdf.set_draw_color(226, 232, 240)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    set_normal(7)
    gray()
    pdf.cell(0, 4, f"SOPilot v{__version__}  |  On-Prem SOP Evaluation System  |  出力: {now}", align="C")

    # ── Output ────────────────────────────────────────────────────────
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

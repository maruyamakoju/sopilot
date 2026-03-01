"""Executive-level analytics PDF report generation for SOPilot.

Uses fpdf2 to generate a structured multi-page A4 PDF report with
aggregate analytics data for management/team review. Follows the same
patterns as report_pdf.py.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from sopilot import __version__

# Font directory for bundled fonts (same location as report_pdf.py)
_FONT_DIR = Path(__file__).resolve().parent / "fonts"

# ── Decision colour map (RGB tuples) ─────────────────────────────────────────
_DEC_RGB: dict[str, tuple[int, int, int]] = {
    "pass": (16, 185, 129),
    "fail": (239, 68, 68),
    "needs_review": (59, 130, 246),
    "retrain": (245, 158, 11),
}
_DEC_JP: dict[str, str] = {
    "pass": "合格",
    "fail": "不合格",
    "needs_review": "要確認",
    "retrain": "再研修",
}

# Teal accent used throughout
_TEAL = (13, 148, 136)
_NAVY = (30, 41, 59)
_GRAY = (100, 116, 139)
_LIGHT_GRAY = (226, 232, 240)
_BG_LIGHT = (248, 250, 252)


def _ensure_font(pdf: Any) -> None:
    """Register a CJK-capable font if available, otherwise use built-in.

    Safe to call multiple times — skips re-registration if already loaded.
    Mirrors the implementation in report_pdf.py.
    """
    if hasattr(pdf, "_sopilot_font"):
        pdf.set_font(pdf._sopilot_font, "", 10)
        return

    try:
        from fpdf import FPDF  # noqa: F401

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


def _safe_int(v: Any, default: int = 0) -> int:
    """Safely cast value to int."""
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely cast value to float."""
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def generate_analytics_pdf(
    analytics_data: dict,
    *,
    title: str = "SOP\u8a55\u4fa1 \u30a8\u30b0\u30bc\u30af\u30c6\u30a3\u30d6\u30ec\u30dd\u30fc\u30c8",
) -> bytes:
    """Generate a management-level PDF analytics report.

    Args:
        analytics_data: dict containing aggregate scoring analytics.
        title: PDF title string.

    Returns:
        PDF content as bytes.
    """
    try:
        return _build_analytics_pdf(analytics_data, title=title)
    except Exception as exc:
        # Return a minimal error PDF so the caller always gets bytes back
        return _build_error_pdf(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Internal builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_analytics_pdf(analytics_data: dict, *, title: str) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError(
            "fpdf2 is required for PDF report generation. "
            "Install it with: pip install fpdf2"
        ) from exc

    # ── Extract data with safe defaults ──────────────────────────────────────
    completed_jobs = _safe_int(analytics_data.get("completed_jobs"))
    pass_count = _safe_int(analytics_data.get("pass_count"))
    fail_count = _safe_int(analytics_data.get("fail_count"))
    needs_review_count = _safe_int(analytics_data.get("needs_review_count"))
    retrain_count = _safe_int(analytics_data.get("retrain_count"))
    avg_score_raw = analytics_data.get("avg_score")
    min_score_raw = analytics_data.get("min_score")
    max_score_raw = analytics_data.get("max_score")
    avg_score = _safe_float(avg_score_raw) if avg_score_raw is not None else None
    min_score = _safe_float(min_score_raw) if min_score_raw is not None else None
    max_score = _safe_float(max_score_raw) if max_score_raw is not None else None

    by_operator: list[dict] = analytics_data.get("by_operator") or []
    by_site: list[dict] = analytics_data.get("by_site") or []
    score_distribution: list[dict] = analytics_data.get("score_distribution") or []
    recent_trend: list[dict] = analytics_data.get("recent_trend") or []
    reviewer_agreement: dict = analytics_data.get("reviewer_agreement") or {}

    compliance_rate_raw = analytics_data.get("compliance_rate")
    compliance_rate = _safe_float(compliance_rate_raw) if compliance_rate_raw is not None else None

    pass_rate = pass_count / completed_jobs if completed_jobs > 0 else 0.0
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    now_jp = now_dt.strftime("%Y年%m月%d日 %H時%M分")

    # ── PDF setup ─────────────────────────────────────────────────────────────
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)

    _ensure_font(pdf)
    font_family = pdf.font_family

    # ── Helper closures ───────────────────────────────────────────────────────
    def set_bold(size: float = 10) -> None:
        pdf.set_font(font_family, "B", size)

    def set_normal(size: float = 10) -> None:
        pdf.set_font(font_family, "", size)

    def color(r: int, g: int, b: int) -> None:
        pdf.set_text_color(r, g, b)

    def teal() -> None:
        color(*_TEAL)

    def gray() -> None:
        color(*_GRAY)

    def black() -> None:
        color(*_NAVY)

    def draw_header_bar() -> None:
        """Draw the teal accent line at top of each page."""
        pdf.set_fill_color(*_TEAL)
        pdf.rect(0, 0, pdf.w, 6, style="F")

    def draw_page_header(page_title: str) -> None:
        """Draw consistent header on each page after the first."""
        draw_header_bar()
        pdf.set_y(10)
        set_bold(9)
        teal()
        pdf.cell(40, 5, "SOPilot", new_x="END")
        set_normal(8)
        gray()
        pdf.cell(0, 5, f"v{__version__}  |  {title}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        set_bold(14)
        black()
        pdf.cell(0, 8, page_title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*_TEAL)
        pdf.set_line_width(0.6)
        pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
        pdf.ln(5)

    def draw_page_footer() -> None:
        """Draw page number footer."""
        y_foot = pdf.h - 12
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.line(pdf.l_margin, y_foot, pdf.w - pdf.r_margin, y_foot)
        pdf.set_y(y_foot + 2)
        set_normal(7)
        gray()
        pdf.cell(
            0, 4,
            f"SOPilot v{__version__}  |  SOP評価 エグゼクティブレポート  |  出力: {now_str}"
            f"  |  ページ {pdf.page}",
            align="C",
        )

    def section_title(sec_title: str) -> None:
        pdf.ln(5)
        set_bold(11)
        color(71, 85, 105)
        pdf.cell(0, 7, sec_title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

    table_w = pdf.w - pdf.l_margin - pdf.r_margin

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Executive Summary
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    draw_header_bar()

    pdf.set_y(10)
    set_bold(9)
    teal()
    pdf.cell(40, 5, "SOPilot", new_x="END")
    set_normal(8)
    gray()
    pdf.cell(0, 5, f"v{__version__}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Main title
    set_bold(20)
    black()
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

    # Subtitle with date
    set_normal(9)
    gray()
    pdf.cell(0, 5, f"レポート生成日: {now_jp}", new_x="LMARGIN", new_y="NEXT")

    # Accent line
    pdf.set_draw_color(*_TEAL)
    pdf.set_line_width(0.8)
    pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.w - pdf.r_margin, pdf.get_y() + 2)
    pdf.ln(8)

    # ── KPI boxes (4 in a row) ────────────────────────────────────────────────
    kpi_w = (table_w - 6) / 4  # 3 gaps of 2mm each
    kpi_h = 26
    y_kpi = pdf.get_y()

    pass_rate_pct = f"{pass_rate * 100:.1f}%"
    avg_score_s = f"{avg_score:.1f}" if avg_score is not None else "—"
    fail_s = str(fail_count)

    kpis = [
        ("採点完了件数", str(completed_jobs), _TEAL),
        ("合格率", pass_rate_pct, (16, 185, 129) if pass_rate >= 0.7 else (239, 68, 68)),
        ("平均スコア", avg_score_s, (16, 185, 129) if avg_score is not None and avg_score >= 80 else (245, 158, 11) if avg_score is not None else _GRAY),
        ("不合格件数", fail_s, (239, 68, 68) if fail_count > 0 else (16, 185, 129)),
    ]

    for i, (label, value, rgb) in enumerate(kpis):
        x = pdf.l_margin + i * (kpi_w + 2)
        # Box background
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.rect(x, y_kpi, kpi_w, kpi_h, style="DF")
        # Colored top accent bar
        pdf.set_fill_color(*rgb)
        pdf.rect(x, y_kpi, kpi_w, 2, style="F")
        # Value (large)
        set_bold(16)
        color(*rgb)
        pdf.set_xy(x, y_kpi + 4)
        pdf.cell(kpi_w, 10, value, align="C")
        # Label (small)
        set_normal(7)
        gray()
        pdf.set_xy(x, y_kpi + 15)
        pdf.cell(kpi_w, 5, label, align="C")

    pdf.set_y(y_kpi + kpi_h + 6)

    # ── Score distribution (horizontal bar chart) ─────────────────────────────
    section_title("スコア分布")

    # Build distribution lookup
    dist_map: dict[str, int] = {}
    for d in score_distribution:
        bucket = str(d.get("bucket", ""))
        cnt = _safe_int(d.get("count"))
        dist_map[bucket] = cnt

    dist_items = [
        ("90-100", dist_map.get("90-100", 0), (16, 185, 129)),
        ("80-89",  dist_map.get("80-89",  0), (13, 148, 136)),
        ("70-79",  dist_map.get("70-79",  0), (245, 158, 11)),
        ("0-69",   dist_map.get("0-69",   0), (239, 68, 68)),
    ]

    total_dist = sum(c for _, c, _ in dist_items) or 1
    bar_area_w = table_w - 35  # label column width = 35mm
    bar_h = 7
    bar_gap = 3

    for bucket_label, cnt, rgb in dist_items:
        bar_len = (cnt / total_dist) * bar_area_w
        y_bar = pdf.get_y()

        # Bucket label
        set_normal(8)
        gray()
        pdf.set_xy(pdf.l_margin, y_bar)
        pdf.cell(20, bar_h, bucket_label, align="R")

        # Background track
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.rect(pdf.l_margin + 22, y_bar + 1, bar_area_w, bar_h - 2, style="F")

        # Colored fill
        if bar_len > 0:
            pdf.set_fill_color(*rgb)
            pdf.rect(pdf.l_margin + 22, y_bar + 1, bar_len, bar_h - 2, style="F")

        # Count label
        set_bold(8)
        color(*rgb)
        pdf.set_xy(pdf.l_margin + 22 + bar_area_w + 2, y_bar)
        pdf.cell(12, bar_h, str(cnt), align="L")

        pdf.set_y(y_bar + bar_h + bar_gap)

    # ── Decision breakdown table ──────────────────────────────────────────────
    section_title("判定内訳")

    decisions = [
        ("合格",   pass_count,          (16, 185, 129)),
        ("要確認", needs_review_count,  (59, 130, 246)),
        ("再研修", retrain_count,       (245, 158, 11)),
        ("不合格", fail_count,          (239, 68, 68)),
    ]

    col_w4 = [table_w * 0.30, table_w * 0.25, table_w * 0.25, table_w * 0.20]

    # Table header
    pdf.set_fill_color(*_BG_LIGHT)
    set_bold(8)
    gray()
    headers4 = ["判定", "件数", "割合", ""]
    for i, h in enumerate(headers4):
        pdf.cell(col_w4[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    set_normal(9)
    for dec_label, cnt, rgb in decisions:
        pct = cnt / completed_jobs * 100 if completed_jobs > 0 else 0.0
        black()
        pdf.cell(col_w4[0], 6, dec_label, border=1)
        color(*rgb)
        set_bold(9)
        pdf.cell(col_w4[1], 6, str(cnt), border=1, align="C")
        set_normal(8)
        gray()
        pdf.cell(col_w4[2], 6, f"{pct:.1f}%", border=1, align="C")
        # Mini bar
        y_row = pdf.get_y()
        x_row = pdf.get_x()
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.rect(x_row, y_row, col_w4[3], 6, style="F")
        bar_pct_w = (pct / 100) * col_w4[3]
        if bar_pct_w > 0:
            pdf.set_fill_color(*rgb)
            pdf.rect(x_row, y_row, bar_pct_w, 6, style="F")
        pdf.cell(col_w4[3], 6, "", border=1)
        set_normal(9)
        pdf.ln()

    draw_page_footer()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Operator Rankings
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    draw_page_header("担当者別パフォーマンス")

    if not by_operator:
        set_normal(11)
        gray()
        pdf.cell(0, 10, "データなし", align="C", new_x="LMARGIN", new_y="NEXT")
    else:
        # Column widths: 順位, 担当者ID, 件数, 平均スコア, 合格率, 最新判定
        col_ops = [
            table_w * 0.08,
            table_w * 0.28,
            table_w * 0.12,
            table_w * 0.16,
            table_w * 0.16,
            table_w * 0.20,
        ]
        op_headers = ["順位", "担当者ID", "件数", "平均スコア", "合格率", "最新判定"]

        pdf.set_fill_color(*_BG_LIGHT)
        set_bold(8)
        gray()
        for i, h in enumerate(op_headers):
            pdf.cell(col_ops[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()

        for rank, op in enumerate(by_operator, start=1):
            if pdf.get_y() > 260:
                pdf.add_page()
                _ensure_font(pdf)
                draw_page_header("担当者別パフォーマンス (続き)")

            op_id = str(op.get("operator_id", "unknown") or "unknown")
            op_jobs = _safe_int(op.get("job_count"))
            op_avg = _safe_float(op.get("avg_score"))
            op_pass = _safe_int(op.get("pass_count"))
            _safe_int(op.get("fail_count"))

            op_pass_rate = op_pass / op_jobs * 100 if op_jobs > 0 else 0.0

            # Color code avg_score
            if op_avg >= 90:
                score_rgb = (16, 185, 129)
            elif op_avg >= 80:
                score_rgb = (245, 158, 11)
            else:
                score_rgb = (239, 68, 68)

            # Latest decision approximation based on pass/fail ratio
            if op_pass_rate >= 90:
                latest_jp = "合格"
                latest_rgb = (16, 185, 129)
            elif op_pass_rate >= 70:
                latest_jp = "要確認"
                latest_rgb = (59, 130, 246)
            elif op_pass_rate >= 50:
                latest_jp = "再研修"
                latest_rgb = (245, 158, 11)
            else:
                latest_jp = "不合格"
                latest_rgb = (239, 68, 68)

            # Truncate long operator IDs
            op_id_display = op_id[:20] + "..." if len(op_id) > 20 else op_id

            gray()
            set_normal(8)
            pdf.cell(col_ops[0], 6, str(rank), border=1, align="C")
            black()
            pdf.cell(col_ops[1], 6, op_id_display, border=1)
            gray()
            pdf.cell(col_ops[2], 6, str(op_jobs), border=1, align="C")
            color(*score_rgb)
            set_bold(8)
            pdf.cell(col_ops[3], 6, f"{op_avg:.1f}", border=1, align="C")
            set_normal(8)
            gray()
            pdf.cell(col_ops[4], 6, f"{op_pass_rate:.1f}%", border=1, align="C")
            color(*latest_rgb)
            set_bold(8)
            pdf.cell(col_ops[5], 6, latest_jp, border=1, align="C")
            set_normal(8)
            pdf.ln()

    # ── Score min/max summary ─────────────────────────────────────────────────
    section_title("スコア統計サマリー")

    stat_items = [
        ("最高スコア", f"{max_score:.1f}" if max_score is not None else "—", (16, 185, 129)),
        ("平均スコア", f"{avg_score:.1f}" if avg_score is not None else "—", _TEAL),
        ("最低スコア", f"{min_score:.1f}" if min_score is not None else "—", (239, 68, 68)),
    ]

    stat_box_w = (table_w - 4) / 3
    y_stat = pdf.get_y()

    for i, (label, val, rgb) in enumerate(stat_items):
        x = pdf.l_margin + i * (stat_box_w + 2)
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.rect(x, y_stat, stat_box_w, 18, style="DF")
        pdf.set_fill_color(*rgb)
        pdf.rect(x, y_stat, stat_box_w, 2, style="F")
        set_bold(14)
        color(*rgb)
        pdf.set_xy(x, y_stat + 3)
        pdf.cell(stat_box_w, 8, val, align="C")
        set_normal(7)
        gray()
        pdf.set_xy(x, y_stat + 12)
        pdf.cell(stat_box_w, 4, label, align="C")

    pdf.set_y(y_stat + 22)

    if compliance_rate is not None:
        section_title("コンプライアンス率")
        set_bold(11)
        color(*(_TEAL if compliance_rate >= 0.7 else (239, 68, 68)))
        pdf.cell(0, 8, f"{compliance_rate * 100:.1f}%", new_x="LMARGIN", new_y="NEXT")

    draw_page_footer()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Site Analysis & Score Trend
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    draw_page_header("拠点別分析")

    if not by_site:
        set_normal(11)
        gray()
        pdf.cell(0, 10, "データなし", align="C", new_x="LMARGIN", new_y="NEXT")
    else:
        col_site = [
            table_w * 0.08,
            table_w * 0.28,
            table_w * 0.12,
            table_w * 0.16,
            table_w * 0.16,
            table_w * 0.20,
        ]
        site_headers = ["順位", "拠点ID", "件数", "平均スコア", "合格率", "状態"]

        pdf.set_fill_color(*_BG_LIGHT)
        set_bold(8)
        gray()
        for i, h in enumerate(site_headers):
            pdf.cell(col_site[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()

        for rank, site in enumerate(by_site, start=1):
            if pdf.get_y() > 230:
                pdf.add_page()
                _ensure_font(pdf)
                draw_page_header("拠点別分析 (続き)")

            site_id = str(site.get("site_id", "unknown") or "unknown")
            site_jobs = _safe_int(site.get("job_count"))
            site_avg = _safe_float(site.get("avg_score"))
            site_pass = _safe_int(site.get("pass_count"))

            site_pass_rate = site_pass / site_jobs * 100 if site_jobs > 0 else 0.0

            if site_avg >= 90:
                score_rgb = (16, 185, 129)
            elif site_avg >= 80:
                score_rgb = (245, 158, 11)
            else:
                score_rgb = (239, 68, 68)

            if site_pass_rate >= 90:
                status_jp = "良好"
                status_rgb = (16, 185, 129)
            elif site_pass_rate >= 70:
                status_jp = "注意"
                status_rgb = (245, 158, 11)
            else:
                status_jp = "要改善"
                status_rgb = (239, 68, 68)

            site_id_display = site_id[:20] + "..." if len(site_id) > 20 else site_id

            gray()
            set_normal(8)
            pdf.cell(col_site[0], 6, str(rank), border=1, align="C")
            black()
            pdf.cell(col_site[1], 6, site_id_display, border=1)
            gray()
            pdf.cell(col_site[2], 6, str(site_jobs), border=1, align="C")
            color(*score_rgb)
            set_bold(8)
            pdf.cell(col_site[3], 6, f"{site_avg:.1f}", border=1, align="C")
            set_normal(8)
            gray()
            pdf.cell(col_site[4], 6, f"{site_pass_rate:.1f}%", border=1, align="C")
            color(*status_rgb)
            set_bold(8)
            pdf.cell(col_site[5], 6, status_jp, border=1, align="C")
            set_normal(8)
            pdf.ln()

    # ── Recent Score Trend ────────────────────────────────────────────────────
    section_title("スコアトレンド (直近30件)")

    if len(recent_trend) < 2:
        set_normal(9)
        gray()
        pdf.cell(0, 8, "データ不足 (2件以上必要)", new_x="LMARGIN", new_y="NEXT")
    else:
        # Draw a simple line chart manually using FPDF drawing primitives
        chart_h = 50.0   # mm
        chart_w = table_w
        x0 = pdf.l_margin
        y0 = pdf.get_y()
        y_bottom = y0 + chart_h

        # Axis labels Y
        set_normal(6)
        gray()
        for score_mark, label in [(100, "100"), (90, "90"), (80, "80"), (70, "70"), (0, "0")]:
            y_mark = y_bottom - (score_mark / 100.0) * chart_h
            pdf.set_xy(x0 - 8, y_mark - 2)
            pdf.cell(7, 4, label, align="R")
            # Grid line (light)
            pdf.set_draw_color(*_LIGHT_GRAY)
            pdf.set_line_width(0.1)
            pdf.line(x0, y_mark, x0 + chart_w, y_mark)

        # Pass threshold line at 90 in green
        y_thresh = y_bottom - (90.0 / 100.0) * chart_h
        pdf.set_draw_color(16, 185, 129)
        pdf.set_line_width(0.4)
        pdf.line(x0, y_thresh, x0 + chart_w, y_thresh)

        # Draw label for threshold
        set_normal(6)
        color(16, 185, 129)
        pdf.set_xy(x0 + chart_w + 1, y_thresh - 2)
        pdf.cell(10, 4, "合格線", align="L")

        # Plot data points
        trend_reversed = list(reversed(recent_trend))  # oldest first
        n = len(trend_reversed)
        step_x = chart_w / (n - 1) if n > 1 else chart_w

        points: list[tuple[float, float, str]] = []
        for idx, item in enumerate(trend_reversed):
            score_val = _safe_float(item.get("score"))
            score_val = max(0.0, min(100.0, score_val))
            decision = str(item.get("decision", ""))
            px = x0 + idx * step_x
            py = y_bottom - (score_val / 100.0) * chart_h
            points.append((px, py, decision))

        # Draw connecting lines
        pdf.set_draw_color(*_GRAY)
        pdf.set_line_width(0.5)
        for i in range(len(points) - 1):
            pdf.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

        # Draw data points (small filled circles)
        for px, py, dec in points:
            pt_rgb = _DEC_RGB.get(dec, _GRAY)
            pdf.set_fill_color(*pt_rgb)
            pdf.set_draw_color(*pt_rgb)
            radius = 1.2
            pdf.ellipse(px - radius, py - radius, radius * 2, radius * 2, style="F")

        # Chart border
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.set_line_width(0.3)
        pdf.rect(x0, y0, chart_w, chart_h, style="D")

        pdf.set_y(y_bottom + 6)

        # Legend
        legend_items = [
            ("合格", (16, 185, 129)),
            ("不合格", (239, 68, 68)),
            ("要確認", (59, 130, 246)),
            ("再研修", (245, 158, 11)),
        ]
        set_normal(7)
        gray()
        pdf.cell(15, 4, "凡例: ", new_x="END")
        for leg_label, leg_rgb in legend_items:
            pdf.set_fill_color(*leg_rgb)
            pdf.rect(pdf.get_x(), pdf.get_y() + 1, 3, 3, style="F")
            pdf.set_x(pdf.get_x() + 4)
            color(*leg_rgb)
            pdf.cell(15, 4, leg_label, new_x="END")
        pdf.ln(6)

    draw_page_footer()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — Reviewer Agreement & Recommendations
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    draw_page_header("品質管理")

    # ── Reviewer agreement ────────────────────────────────────────────────────
    section_title("レビュアー一致率")

    reviewed_count = _safe_int(reviewer_agreement.get("reviewed_count"))
    agree_count = _safe_int(reviewer_agreement.get("agree_count"))
    agreement_rate_raw = reviewer_agreement.get("agreement_rate")
    agreement_rate = _safe_float(agreement_rate_raw) if agreement_rate_raw is not None else None
    by_verdict: dict = reviewer_agreement.get("by_verdict") or {}

    if reviewed_count == 0:
        set_normal(10)
        gray()
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.cell(0, 10, "レビューデータなし", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    else:
        # Agreement rate big number
        agree_pct = (agreement_rate * 100) if agreement_rate is not None else 0.0
        agree_rgb = (16, 185, 129) if agree_pct >= 80 else (245, 158, 11) if agree_pct >= 60 else (239, 68, 68)

        y_agree = pdf.get_y()
        agree_box_w = table_w / 2 - 5

        # Left box: agreement rate
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.rect(pdf.l_margin, y_agree, agree_box_w, 22, style="DF")
        pdf.set_fill_color(*agree_rgb)
        pdf.rect(pdf.l_margin, y_agree, agree_box_w, 2, style="F")
        set_bold(18)
        color(*agree_rgb)
        pdf.set_xy(pdf.l_margin, y_agree + 3)
        pdf.cell(agree_box_w, 10, f"{agree_pct:.1f}%", align="C")
        set_normal(7)
        gray()
        pdf.set_xy(pdf.l_margin, y_agree + 14)
        pdf.cell(agree_box_w, 5, "レビュアー一致率", align="C")

        # Right box: counts
        x_right = pdf.l_margin + agree_box_w + 10
        pdf.set_fill_color(*_BG_LIGHT)
        pdf.set_draw_color(*_LIGHT_GRAY)
        pdf.rect(x_right, y_agree, agree_box_w, 22, style="DF")
        set_bold(14)
        teal()
        pdf.set_xy(x_right, y_agree + 3)
        pdf.cell(agree_box_w, 10, f"{agree_count} / {reviewed_count}", align="C")
        set_normal(7)
        gray()
        pdf.set_xy(x_right, y_agree + 14)
        pdf.cell(agree_box_w, 5, "一致件数 / レビュー総数", align="C")

        pdf.set_y(y_agree + 26)

        # Verdict breakdown
        if by_verdict:
            section_title("レビュアー判定内訳")
            verdict_data = [
                ("合格",   _safe_int(by_verdict.get("pass")),         (16, 185, 129)),
                ("不合格", _safe_int(by_verdict.get("fail")),         (239, 68, 68)),
                ("再研修", _safe_int(by_verdict.get("retrain")),      (245, 158, 11)),
                ("要確認", _safe_int(by_verdict.get("needs_review")), (59, 130, 246)),
            ]
            col_v = [table_w * 0.30, table_w * 0.35, table_w * 0.35]

            pdf.set_fill_color(*_BG_LIGHT)
            set_bold(8)
            gray()
            for h in ["レビュアー判定", "件数", "割合"]:
                pdf.cell(col_v[0] if h == "レビュアー判定" else col_v[1], 7, h, border=1, fill=True, align="C")
            pdf.ln()

            for verd_label, verd_cnt, verd_rgb in verdict_data:
                verd_pct = verd_cnt / reviewed_count * 100 if reviewed_count > 0 else 0.0
                black()
                set_normal(8)
                pdf.cell(col_v[0], 6, verd_label, border=1)
                color(*verd_rgb)
                set_bold(8)
                pdf.cell(col_v[1], 6, str(verd_cnt), border=1, align="C")
                set_normal(8)
                gray()
                pdf.cell(col_v[2], 6, f"{verd_pct:.1f}%", border=1, align="C")
                pdf.ln()

    # ── Auto-generated improvement notes ──────────────────────────────────────
    section_title("改善提案・コメント")

    notes: list[tuple[str, tuple[int, int, int]]] = []

    if pass_rate < 0.7:
        notes.append((
            "合格率が70%を下回っています。研修プログラムの見直しを推奨します。",
            (239, 68, 68),
        ))

    if avg_score is not None and avg_score < 80:
        notes.append((
            f"平均スコアが80点を下回っています (現在: {avg_score:.1f}点)。",
            (245, 158, 11),
        ))

    # Check for operators with compliance < 50%
    low_compliance_ops = [
        op for op in by_operator
        if _safe_int(op.get("job_count")) > 0
        and _safe_int(op.get("pass_count")) / _safe_int(op.get("job_count")) < 0.5
    ]
    if low_compliance_ops:
        notes.append((
            f"低合格率の担当者が{len(low_compliance_ops)}名存在します。個別研修を推奨します。",
            (245, 158, 11),
        ))

    if not notes:
        notes.append((
            "全体的に良好なパフォーマンスを維持しています。",
            (16, 185, 129),
        ))

    for note_text, note_rgb in notes:
        y_note = pdf.get_y()
        # Bullet indicator bar
        pdf.set_fill_color(*note_rgb)
        pdf.rect(pdf.l_margin, y_note + 1, 3, 8, style="F")
        # Note text
        pdf.set_fill_color(
            min(255, note_rgb[0] + 220),
            min(255, note_rgb[1] + 220),
            min(255, note_rgb[2] + 220),
        )
        color(*note_rgb)
        set_bold(9)
        pdf.set_xy(pdf.l_margin + 5, y_note)
        pdf.cell(table_w - 5, 10, note_text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # ── Report metadata ───────────────────────────────────────────────────────
    if pdf.get_y() > 230:
        pdf.add_page()
        _ensure_font(pdf)
        draw_page_header("レポートメタデータ")

    section_title("レポート情報")

    meta_items = [
        ("レポート種別", "エグゼクティブ アナリティクス レポート"),
        ("生成日時", now_str),
        ("SOPilot バージョン", f"v{__version__}"),
        ("集計完了ジョブ数", str(completed_jobs)),
        ("分析対象担当者数", str(len(by_operator))),
        ("分析対象拠点数", str(len(by_site))),
    ]

    col_meta = [table_w * 0.40, table_w * 0.60]
    for label, val in meta_items:
        set_normal(8)
        gray()
        pdf.cell(col_meta[0], 6, label + ":", border="B")
        set_bold(8)
        black()
        pdf.cell(col_meta[1], 6, val, border="B", new_x="LMARGIN", new_y="NEXT")

    draw_page_footer()

    # ── Output ────────────────────────────────────────────────────────────────
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _build_error_pdf(error_message: str) -> bytes:
    """Return a minimal single-page PDF indicating report generation failed."""
    try:
        from fpdf import FPDF
    except ImportError:
        # Last resort: return empty bytes
        return b""

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    _ensure_font(pdf)
    font_family = pdf.font_family

    pdf.set_font(font_family, "B", 16)
    pdf.set_text_color(239, 68, 68)
    pdf.cell(0, 20, "レポート生成エラー", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(font_family, "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, "PDFレポートの生成中にエラーが発生しました。", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(font_family, "", 8)
    pdf.set_text_color(30, 41, 59)
    msg = error_message[:200] + ("..." if len(error_message) > 200 else "")
    pdf.ln(6)
    pdf.cell(0, 6, msg, align="C", new_x="LMARGIN", new_y="NEXT")

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

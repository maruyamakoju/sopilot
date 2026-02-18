"""Insurance MVP - Streamlit Dashboard for PoC Demo.

Interactive dashboard for Sompo Japan PoC demonstration.
Supports video upload, pipeline execution, and results visualization.
Backend modes: Mock (no GPU), Real (Qwen2.5-VL-7B on GPU), Replay (load JSON).

Usage:
    streamlit run insurance_mvp/dashboard.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from insurance_mvp.config import CosmosBackend, PipelineConfig  # noqa: E402
from insurance_mvp.pipeline import InsurancePipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_DIR = PROJECT_ROOT / "data" / "dashcam_demo"
SEVERITY_COLORS = {
    "HIGH": "#DC2626",
    "MEDIUM": "#F59E0B",
    "LOW": "#3B82F6",
    "NONE": "#10B981",
}
SEVERITY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "NONE": "🟢"}
PRIORITY_COLORS = {
    "URGENT": "#DC2626",
    "STANDARD": "#F59E0B",
    "LOW_PRIORITY": "#10B981",
}

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

TRANSLATIONS = {
    "en": {
        "title": "Insurance Claim Assessment",
        "subtitle": "AI-Powered Dashcam Video Analysis",
        "sidebar_title": "Settings",
        "lang_label": "Language",
        "mode_label": "Input Mode",
        "mode_demo": "Demo Videos",
        "mode_upload": "Upload Video",
        "select_video": "Select Demo Video",
        "upload_prompt": "Upload a dashcam video (.mp4)",
        "run_button": "Run Assessment",
        "running": "Processing video...",
        "results_title": "Assessment Results",
        "severity": "Severity",
        "confidence": "Confidence",
        "prediction_set": "Prediction Set (90% CI)",
        "review_priority": "Review Priority",
        "fault_title": "Fault Assessment",
        "fault_ratio": "Fault Ratio",
        "scenario": "Scenario Type",
        "reasoning": "Reasoning",
        "rules": "Applicable Rules",
        "fraud_title": "Fraud Detection",
        "fraud_score": "Fraud Risk Score",
        "fraud_indicators": "Indicators",
        "fraud_reasoning": "Reasoning",
        "evidence_title": "Evidence",
        "hazards_title": "Detected Hazards",
        "causal": "Causal Analysis",
        "action": "Recommended Action",
        "processing_time": "Processing Time",
        "stage_timings": "Stage Timings",
        "report_title": "Report",
        "download_json": "Download JSON",
        "download_html": "Download HTML Report",
        "no_results": "No assessment results. Run the pipeline first.",
        "pipeline_success": "Pipeline completed successfully!",
        "pipeline_failed": "Pipeline failed",
        "ground_truth": "Ground Truth",
        "gt_match": "Match",
        "gt_mismatch": "Mismatch",
        "ego_vehicle": "Ego Vehicle",
        "other_vehicle": "Other Vehicle",
        "shared": "Shared Fault",
        "vlm_backend": "VLM Backend",
        "conformal_alpha": "Conformal Alpha",
        "overview": "Overview",
        "details": "Details",
        "comparison": "Ground Truth Comparison",
        "backend_label": "Backend",
        "backend_mock": "Mock (No GPU)",
        "backend_real": "Real (Qwen2.5-VL-7B)",
        "backend_replay": "Replay (Load JSON)",
        "replay_select": "Select Result File",
        "replay_upload": "Or upload JSON file",
        "replay_loaded": "Results loaded from file",
        "replay_no_files": "No result files found in reports/",
        "gpu_not_available": "GPU not available. Install PyTorch with CUDA support.",
        "gpu_info": "GPU",
        "real_warning": "Real VLM inference requires GPU (14GB+ VRAM). ~2-3 min per video.",
        "replay_source": "Source",
    },
    "ja": {
        "title": "保険金請求査定",
        "subtitle": "AI搭載ドラレコ映像解析",
        "sidebar_title": "設定",
        "lang_label": "言語",
        "mode_label": "入力モード",
        "mode_demo": "デモ動画",
        "mode_upload": "動画アップロード",
        "select_video": "デモ動画を選択",
        "upload_prompt": "ドラレコ動画をアップロード (.mp4)",
        "run_button": "査定実行",
        "running": "動画を処理中...",
        "results_title": "査定結果",
        "severity": "重大度",
        "confidence": "確信度",
        "prediction_set": "予測セット (90% CI)",
        "review_priority": "審査優先度",
        "fault_title": "過失割合",
        "fault_ratio": "過失割合",
        "scenario": "シナリオタイプ",
        "reasoning": "判断根拠",
        "rules": "適用ルール",
        "fraud_title": "不正検知",
        "fraud_score": "不正リスクスコア",
        "fraud_indicators": "検出指標",
        "fraud_reasoning": "判断根拠",
        "evidence_title": "証拠",
        "hazards_title": "検出された危険",
        "causal": "因果分析",
        "action": "推奨アクション",
        "processing_time": "処理時間",
        "stage_timings": "ステージ別処理時間",
        "report_title": "レポート",
        "download_json": "JSONダウンロード",
        "download_html": "HTMLレポートダウンロード",
        "no_results": "査定結果がありません。パイプラインを実行してください。",
        "pipeline_success": "パイプライン正常完了!",
        "pipeline_failed": "パイプライン失敗",
        "ground_truth": "正解データ",
        "gt_match": "一致",
        "gt_mismatch": "不一致",
        "ego_vehicle": "自車",
        "other_vehicle": "相手車",
        "shared": "双方過失",
        "vlm_backend": "VLMバックエンド",
        "conformal_alpha": "Conformal Alpha",
        "overview": "概要",
        "details": "詳細",
        "comparison": "正解データ比較",
        "backend_label": "バックエンド",
        "backend_mock": "Mock（GPU不要）",
        "backend_real": "Real（Qwen2.5-VL-7B）",
        "backend_replay": "リプレイ（JSON読込）",
        "replay_select": "結果ファイルを選択",
        "replay_upload": "またはJSONファイルをアップロード",
        "replay_loaded": "ファイルから結果を読み込みました",
        "replay_no_files": "reports/ に結果ファイルが見つかりません",
        "gpu_not_available": "GPUが利用できません。CUDA対応PyTorchをインストールしてください。",
        "gpu_info": "GPU",
        "real_warning": "Real VLM推論にはGPU（VRAM 14GB以上）が必要です。動画1本あたり約2〜3分。",
        "replay_source": "ソース",
    },
}

DEMO_VIDEOS = {
    "collision": {
        "en": "Rear-end Collision (HIGH severity)",
        "ja": "追突事故 (重大度: HIGH)",
        "gt_severity": "HIGH",
        "gt_fault": 100.0,
        "gt_fraud": 0.0,
        "gt_scenario": "rear_end",
    },
    "near_miss": {
        "en": "Pedestrian Near-Miss (MEDIUM severity)",
        "ja": "歩行者ニアミス (重大度: MEDIUM)",
        "gt_severity": "MEDIUM",
        "gt_fault": 0.0,
        "gt_fraud": 0.0,
        "gt_scenario": "pedestrian_avoidance",
    },
    "normal": {
        "en": "Normal Driving (NONE severity)",
        "ja": "通常走行 (重大度: NONE)",
        "gt_severity": "NONE",
        "gt_fault": 0.0,
        "gt_fraud": 0.0,
        "gt_scenario": "normal_driving",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def t(key: str) -> str:
    """Get translation for current language."""
    lang = st.session_state.get("lang", "ja")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def severity_badge(severity: str) -> str:
    """Create HTML badge for severity level."""
    color = SEVERITY_COLORS.get(severity, "#6B7280")
    return (
        f'<span style="background-color:{color};color:white;padding:4px 12px;'
        f'border-radius:9999px;font-weight:bold;font-size:0.9em;">{severity}</span>'
    )


def gauge_metric(value: float, max_val: float, label: str, color: str = "#3B82F6") -> str:
    """Create a simple gauge-style metric HTML."""
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return f"""
    <div style="margin:8px 0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-weight:600;">{label}</span>
            <span style="font-weight:700;color:{color};">{value:.1f}{"%" if max_val == 100 else ""}</span>
        </div>
        <div style="background:#E5E7EB;border-radius:9999px;height:12px;overflow:hidden;">
            <div style="background:{color};height:100%;width:{pct}%;border-radius:9999px;
                        transition:width 0.5s ease;"></div>
        </div>
    </div>
    """


def check_gpu() -> dict:
    """Check GPU availability for real VLM inference."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {"available": True, "name": name, "vram_gb": round(vram, 1)}
        return {"available": False, "name": None, "vram_gb": 0}
    except ImportError:
        return {"available": False, "name": None, "vram_gb": 0}


def run_pipeline(video_path: str, video_id: str, backend: str = "mock") -> dict:
    """Run the insurance pipeline on a video.

    Args:
        video_path: Path to video file
        video_id: Video identifier
        backend: "mock" or "real"
    """
    output_dir = tempfile.mkdtemp(prefix="insurance_dashboard_")

    config = PipelineConfig(
        output_dir=output_dir,
        log_level="WARNING",
        parallel_workers=1,
        enable_conformal=True,
        enable_transcription=False,
        continue_on_error=True,
    )

    if backend == "real":
        config.cosmos.backend = CosmosBackend.QWEN25VL
    else:
        config.cosmos.backend = CosmosBackend.MOCK

    pipeline = InsurancePipeline(config)

    if backend == "mock":
        pipeline.signal_fuser = None
        pipeline.cosmos_client = None

    result = pipeline.process_video(str(video_path), video_id=video_id)

    return {
        "video_id": result.video_id,
        "success": result.success,
        "error_message": result.error_message,
        "assessments": result.assessments or [],
        "processing_time_sec": result.processing_time_sec,
        "stage_timings": result.stage_timings or {},
        "output_json_path": result.output_json_path,
        "output_html_path": result.output_html_path,
        "backend": backend,
    }


def load_replay_json(json_path: str | Path) -> dict | None:
    """Load benchmark/replay results from JSON file.

    Supports two formats:
    - Benchmark format: {"videos": {"collision": {"actual": {...}}}, "summary": {...}}
    - Pipeline format: [{"severity": ..., "confidence": ...}]
    """
    from insurance_mvp.cosmos.schema import ClaimAssessment, FaultAssessment, FraudRisk

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Format 1: Benchmark results (real_benchmark.json)
    if isinstance(data, dict) and "videos" in data:
        all_assessments = []
        for vid_name, vid_data in data["videos"].items():
            if not vid_data.get("success", False):
                continue
            actual = vid_data.get("actual", {})
            try:
                assessment = ClaimAssessment(
                    severity=actual.get("severity", "LOW"),
                    confidence=actual.get("confidence", 0.0),
                    prediction_set=set(actual.get("prediction_set", ["LOW"])),
                    review_priority=actual.get("review_priority", "STANDARD"),
                    fault_assessment=FaultAssessment(
                        fault_ratio=actual.get("fault_ratio", 50.0),
                        reasoning=actual.get("fault_reasoning", "From replay"),
                        scenario_type=actual.get("scenario_type", "unknown"),
                    ),
                    fraud_risk=FraudRisk(
                        risk_score=actual.get("fraud_score", 0.0),
                        indicators=actual.get("fraud_indicators", []),
                        reasoning="From replay",
                    ),
                    hazards=[],
                    evidence=[],
                    causal_reasoning=actual.get("causal_reasoning", "Loaded from replay"),
                    recommended_action=actual.get("recommended_action", "REVIEW"),
                    video_id=vid_name,
                    processing_time_sec=vid_data.get("inference_time_sec", 0.0),
                )
                all_assessments.append(assessment)
            except Exception:
                continue

        if not all_assessments:
            return None

        return {
            "video_id": "replay",
            "success": True,
            "error_message": None,
            "assessments": all_assessments,
            "processing_time_sec": data.get("summary", {}).get("total_inference_time_sec", 0.0),
            "stage_timings": {},
            "output_json_path": None,
            "output_html_path": None,
            "backend": f"replay ({data.get('backend', 'unknown')})",
            "replay_source": str(json_path),
            "replay_metadata": {
                "model": data.get("model", "unknown"),
                "timestamp": data.get("timestamp", "unknown"),
                "model_load_time": data.get("model_load_time_sec", 0),
                "summary": data.get("summary", {}),
            },
        }

    return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Insurance Claim Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2em; }
    .main-header p { color: #BFDBFE; margin: 4px 0 0; }
    .metric-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card h3 { margin: 0 0 12px; color: #374151; font-size: 1.1em; }
    .severity-high { border-left: 4px solid #DC2626; }
    .severity-medium { border-left: 4px solid #F59E0B; }
    .severity-low { border-left: 4px solid #3B82F6; }
    .severity-none { border-left: 4px solid #10B981; }
    .gt-match { color: #10B981; font-weight: bold; }
    .gt-mismatch { color: #DC2626; font-weight: bold; }
    .stage-timing { display: flex; justify-content: space-between; padding: 4px 0; }
    div[data-testid="stMetricValue"] { font-size: 1.8em; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "lang" not in st.session_state:
    st.session_state.lang = "ja"
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "backend" not in st.session_state:
    st.session_state.backend = "mock"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### {t('sidebar_title')}")

    lang = st.radio(t("lang_label"), ["ja", "en"], index=0, format_func=lambda x: "日本語" if x == "ja" else "English")
    st.session_state.lang = lang

    st.divider()

    # Backend selector
    backend_options = {
        "mock": t("backend_mock"),
        "real": t("backend_real"),
        "replay": t("backend_replay"),
    }
    backend = st.radio(
        t("backend_label"),
        list(backend_options.keys()),
        format_func=lambda x: backend_options[x],
    )
    st.session_state.backend = backend

    # GPU info for real backend
    if backend == "real":
        gpu_info = check_gpu()
        if gpu_info["available"]:
            st.success(f"🟢 {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
        else:
            st.error(t("gpu_not_available"))
        st.caption(t("real_warning"))

    st.divider()

    # Mode selector (not needed for replay)
    if backend != "replay":
        mode = st.radio(t("mode_label"), ["demo", "upload"], format_func=lambda x: t(f"mode_{x}"))
    else:
        mode = "replay"

    st.divider()

    st.markdown(f"**{t('vlm_backend')}**: {backend_options[backend]}")
    st.markdown(f"**{t('conformal_alpha')}**: 0.10 (90% CI)")

    st.divider()
    st.caption("Insurance MVP v0.1.0")
    st.caption("Powered by Qwen2.5-VL + Conformal Prediction")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<div class="main-header">
    <h1>🚗 {t("title")}</h1>
    <p>{t("subtitle")}</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Video selection
# ---------------------------------------------------------------------------

video_path = None
video_id = None
gt_data = None

if mode == "replay":
    # ------------------------------------------------------------------
    # Replay mode: load JSON results from previous GPU runs
    # ------------------------------------------------------------------
    st.markdown("### " + t("backend_replay"))

    # Find existing result files (demo_assets first, then reports/)
    demo_assets_dir = PROJECT_ROOT / "insurance_mvp" / "demo_assets"
    reports_dir = PROJECT_ROOT / "reports"
    result_files = []
    if demo_assets_dir.exists():
        result_files.extend(sorted(demo_assets_dir.glob("*.json"), reverse=True))
    if reports_dir.exists():
        result_files.extend(sorted(reports_dir.glob("*.json"), reverse=True))

    replay_source = None

    if result_files:
        file_options = {str(f): f.name for f in result_files}
        selected_file = st.selectbox(
            t("replay_select"), list(file_options.keys()), format_func=lambda x: file_options[x]
        )
        if selected_file:
            replay_source = selected_file

    # Also allow upload
    uploaded_json = st.file_uploader(t("replay_upload"), type=["json"])
    if uploaded_json:
        tmp_json = Path(tempfile.mkdtemp()) / uploaded_json.name
        tmp_json.write_bytes(uploaded_json.read())
        replay_source = str(tmp_json)

    if replay_source:
        replay_data = load_replay_json(replay_source)
        if replay_data:
            st.session_state.results = replay_data
            st.success(f"{t('replay_loaded')}: {Path(replay_source).name}")

            # Show replay metadata
            meta = replay_data.get("replay_metadata", {})
            if meta:
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.metric("Model", meta.get("model", "unknown"))
                with meta_col2:
                    summary = meta.get("summary", {})
                    st.metric("Severity Accuracy", f"{summary.get('severity_accuracy', 0)}%")
                with meta_col3:
                    st.metric("Inference Time", f"{summary.get('mean_inference_time_sec', 0):.1f}s/video")

            # Show per-video results selector for replay
            if replay_data.get("assessments") and len(replay_data["assessments"]) > 1:
                video_names = [a.video_id for a in replay_data["assessments"]]
                selected_replay = st.selectbox("Select video result", video_names)
                if selected_replay:
                    # Filter to selected video
                    sel_assessment = [a for a in replay_data["assessments"] if a.video_id == selected_replay]
                    if sel_assessment:
                        replay_data_filtered = dict(replay_data)
                        replay_data_filtered["assessments"] = sel_assessment
                        replay_data_filtered["video_id"] = selected_replay
                        st.session_state.results = replay_data_filtered
                        # Set GT data if it's a demo video
                        if selected_replay in DEMO_VIDEOS:
                            gt_data = DEMO_VIDEOS[selected_replay]
                            video_id = selected_replay
        else:
            st.warning("Could not parse JSON file. Expected benchmark or pipeline format.")

elif mode == "demo":
    demo_options = {k: v[lang] for k, v in DEMO_VIDEOS.items()}
    selected = st.selectbox(t("select_video"), list(demo_options.keys()), format_func=lambda k: demo_options[k])

    if selected:
        video_file = DEMO_DIR / f"{selected}.mp4"
        if video_file.exists():
            video_path = str(video_file)
            video_id = selected
            gt_data = DEMO_VIDEOS[selected]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(str(video_file))
            with col2:
                st.markdown(f"#### {t('ground_truth')}")
                gt_sev = gt_data["gt_severity"]
                st.markdown(f"**{t('severity')}**: {SEVERITY_EMOJI.get(gt_sev, '')} {gt_sev}")
                st.markdown(f"**{t('fault_ratio')}**: {gt_data['gt_fault']:.0f}%")
                st.markdown(f"**{t('scenario')}**: {gt_data['gt_scenario']}")
                st.markdown(f"**{t('fraud_score')}**: {gt_data['gt_fraud']:.1f}")
        else:
            st.warning(f"Demo video not found: {video_file}")
else:
    uploaded = st.file_uploader(t("upload_prompt"), type=["mp4", "avi", "mov"])
    if uploaded:
        tmp = Path(tempfile.mkdtemp()) / uploaded.name
        tmp.write_bytes(uploaded.read())
        video_path = str(tmp)
        video_id = tmp.stem
        st.video(str(tmp))

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if video_path and mode != "replay":
    # Check if real backend is available
    can_run = True
    if backend == "real":
        gpu_info = check_gpu()
        if not gpu_info["available"]:
            st.error(t("gpu_not_available"))
            can_run = False

    if can_run and st.button(t("run_button"), type="primary", use_container_width=True):
        with st.spinner(t("running")):
            start = time.time()
            results = run_pipeline(video_path, video_id, backend=backend)
            elapsed = time.time() - start

        st.session_state.results = results
        st.session_state.selected_video = video_id

        if results["success"]:
            st.success(f"{t('pipeline_success')} ({elapsed:.1f}s) [{backend_options[backend]}]")
        else:
            st.error(f"{t('pipeline_failed')}: {results.get('error_message', 'unknown')}")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

results = st.session_state.results
if results and results.get("assessments"):
    assessments = results["assessments"]
    primary = assessments[0] if assessments else None

    if primary is None:
        st.info(t("no_results"))
    else:
        st.markdown(f"## {t('results_title')}")

        tab_overview, tab_details, tab_comparison = st.tabs([t("overview"), t("details"), t("comparison")])

        # ---------------------------------------------------------------
        # Tab 1: Overview
        # ---------------------------------------------------------------
        with tab_overview:
            col1, col2, col3, col4 = st.columns(4)

            sev = primary.severity
            with col1:
                st.markdown(
                    f"""<div class="metric-card severity-{sev.lower()}">
                    <h3>{t("severity")}</h3>
                    <div style="font-size:2.2em;font-weight:800;color:{SEVERITY_COLORS.get(sev, "#6B7280")};">
                        {SEVERITY_EMOJI.get(sev, "")} {sev}
                    </div>
                    <div style="color:#6B7280;margin-top:4px;">
                        {t("confidence")}: {primary.confidence:.0%}
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            with col2:
                fault = primary.fault_assessment.fault_ratio
                fault_color = "#DC2626" if fault >= 75 else "#F59E0B" if fault >= 25 else "#10B981"
                at_fault = t("ego_vehicle") if fault >= 75 else t("other_vehicle") if fault <= 25 else t("shared")
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>{t("fault_title")}</h3>
                    <div style="font-size:2.2em;font-weight:800;color:{fault_color};">
                        {fault:.0f}%
                    </div>
                    <div style="color:#6B7280;margin-top:4px;">{at_fault}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            with col3:
                fraud = primary.fraud_risk.risk_score
                fraud_color = "#DC2626" if fraud >= 0.65 else "#F59E0B" if fraud >= 0.4 else "#10B981"
                fraud_label = "HIGH" if fraud >= 0.65 else "MEDIUM" if fraud >= 0.4 else "LOW"
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>{t("fraud_title")}</h3>
                    <div style="font-size:2.2em;font-weight:800;color:{fraud_color};">
                        {fraud:.2f}
                    </div>
                    <div style="color:#6B7280;margin-top:4px;">{fraud_label} Risk</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            with col4:
                priority = primary.review_priority
                p_color = PRIORITY_COLORS.get(priority, "#6B7280")
                p_label_map = {
                    "URGENT": "緊急" if lang == "ja" else "URGENT",
                    "STANDARD": "標準" if lang == "ja" else "STANDARD",
                    "LOW_PRIORITY": "低" if lang == "ja" else "LOW",
                }
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>{t("review_priority")}</h3>
                    <div style="font-size:2.2em;font-weight:800;color:{p_color};">
                        {p_label_map.get(priority, priority)}
                    </div>
                    <div style="color:#6B7280;margin-top:4px;">{t("action")}: {primary.recommended_action}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            # Prediction set
            st.markdown("---")
            pred_set = primary.prediction_set
            pred_badges = " ".join(
                f'<span style="background:{SEVERITY_COLORS.get(s, "#6B7280")};color:white;'
                f'padding:3px 10px;border-radius:9999px;margin:2px;font-size:0.85em;">{s}</span>'
                for s in sorted(
                    pred_set,
                    key=lambda x: ["NONE", "LOW", "MEDIUM", "HIGH"].index(x)
                    if x in ["NONE", "LOW", "MEDIUM", "HIGH"]
                    else 99,
                )
            )
            st.markdown(
                f"**{t('prediction_set')}**: {pred_badges}",
                unsafe_allow_html=True,
            )

            # Causal reasoning
            st.markdown("---")
            st.markdown(f"#### {t('causal')}")
            st.info(primary.causal_reasoning)

            # Processing time
            st.markdown("---")
            timing_col1, timing_col2 = st.columns(2)
            with timing_col1:
                st.metric(t("processing_time"), f"{results['processing_time_sec']:.2f}s")
            with timing_col2:
                if results.get("stage_timings"):
                    st.markdown(f"**{t('stage_timings')}**")
                    for stage, dur in results["stage_timings"].items():
                        st.markdown(f"- {stage}: `{dur:.3f}s`")

        # ---------------------------------------------------------------
        # Tab 2: Details
        # ---------------------------------------------------------------
        with tab_details:
            # Fault details
            st.markdown(f"### {t('fault_title')}")
            fa = primary.fault_assessment
            st.markdown(
                gauge_metric(fa.fault_ratio, 100, t("fault_ratio"), fault_color),
                unsafe_allow_html=True,
            )
            st.markdown(f"**{t('scenario')}**: `{fa.scenario_type}`")
            if fa.traffic_signal:
                st.markdown(f"**Traffic Signal**: `{fa.traffic_signal}`")
            if fa.right_of_way:
                st.markdown(f"**Right of Way**: `{fa.right_of_way}`")
            st.markdown(f"**{t('reasoning')}**")
            st.text(fa.reasoning)
            if fa.applicable_rules:
                st.markdown(f"**{t('rules')}**")
                for rule in fa.applicable_rules:
                    st.markdown(f"- {rule}")

            st.divider()

            # Fraud details
            st.markdown(f"### {t('fraud_title')}")
            fr = primary.fraud_risk
            st.markdown(
                gauge_metric(fr.risk_score * 100, 100, t("fraud_score"), fraud_color),
                unsafe_allow_html=True,
            )
            if fr.indicators:
                st.markdown(f"**{t('fraud_indicators')}**")
                for ind in fr.indicators:
                    st.markdown(f"- ⚠️ {ind}")
            else:
                st.success("No fraud indicators detected" if lang == "en" else "不正指標は検出されませんでした")
            st.markdown(f"**{t('fraud_reasoning')}**")
            st.text(fr.reasoning)

            st.divider()

            # Evidence
            if primary.evidence:
                st.markdown(f"### {t('evidence_title')}")
                for ev in primary.evidence:
                    st.markdown(f"- **{ev.timestamp_sec:.1f}s**: {ev.description}")

            # Hazards
            if primary.hazards:
                st.markdown(f"### {t('hazards_title')}")
                for hz in primary.hazards:
                    st.markdown(
                        f"- **{hz.type}** at {hz.timestamp_sec:.1f}s - Actors: {', '.join(hz.actors)} ({hz.spatial_relation})"
                    )

            st.divider()

            # Raw JSON
            st.markdown("### Raw Assessment Data")
            with st.expander("JSON"):
                st.json(primary.model_dump(mode="json", exclude={"timestamp"}))

        # ---------------------------------------------------------------
        # Tab 3: Ground Truth Comparison
        # ---------------------------------------------------------------
        with tab_comparison:
            if gt_data:
                st.markdown(f"### {t('comparison')}")

                comp_col1, comp_col2, comp_col3 = st.columns(3)

                with comp_col1:
                    st.markdown("**Field**")
                    st.markdown(t("severity"))
                    st.markdown(t("fault_ratio"))
                    st.markdown(t("fraud_score"))
                    st.markdown(t("scenario"))

                with comp_col2:
                    st.markdown(f"**{t('ground_truth')}**")
                    st.markdown(f"{SEVERITY_EMOJI.get(gt_data['gt_severity'], '')} {gt_data['gt_severity']}")
                    st.markdown(f"{gt_data['gt_fault']:.0f}%")
                    st.markdown(f"{gt_data['gt_fraud']:.1f}")
                    st.markdown(f"`{gt_data['gt_scenario']}`")

                with comp_col3:
                    st.markdown("**AI Assessment**")

                    # Severity: exact match, borderline (distance=1), or mismatch
                    sev_match = primary.severity == gt_data["gt_severity"]
                    sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
                    sev_dist = abs(sev_order.get(primary.severity, -1) - sev_order.get(gt_data["gt_severity"], -1))
                    sev_in_pred_set = gt_data["gt_severity"] in primary.prediction_set

                    if sev_match:
                        sev_icon = "✅"
                    elif sev_dist == 1:
                        sev_icon = "🟡"  # Borderline
                    else:
                        sev_icon = "❌"
                    sev_note = f" {primary.severity} {sev_icon}"
                    if not sev_match and sev_dist == 1:
                        borderline_label = "Borderline" if lang == "en" else "境界値"
                        sev_note += f' <span style="color:#F59E0B;font-size:0.85em;">({borderline_label})</span>'
                    if not sev_match and sev_in_pred_set:
                        covered_label = "Covered by 90% CI" if lang == "en" else "90% CIでカバー"
                        sev_note += f' <span style="color:#3B82F6;font-size:0.85em;">({covered_label})</span>'
                    st.markdown(
                        f"{SEVERITY_EMOJI.get(primary.severity, '')}{sev_note}",
                        unsafe_allow_html=True,
                    )

                    fault_match = abs(primary.fault_assessment.fault_ratio - gt_data["gt_fault"]) < 20
                    st.markdown(f"{primary.fault_assessment.fault_ratio:.0f}% {'✅' if fault_match else '❌'}")
                    fraud_match = abs(primary.fraud_risk.risk_score - gt_data["gt_fraud"]) < 0.3
                    st.markdown(f"{primary.fraud_risk.risk_score:.1f} {'✅' if fraud_match else '❌'}")
                    st.markdown(f"`{primary.fault_assessment.scenario_type}`")

                # Accuracy summary (borderline counts as partial pass)
                st.divider()
                sev_score = 1.0 if sev_match else (0.5 if sev_dist == 1 else 0.0)
                checks_score = sev_score + (1.0 if fault_match else 0.0) + (1.0 if fraud_match else 0.0)
                total = 3
                display_score = f"{checks_score:.1f}" if checks_score != int(checks_score) else f"{int(checks_score)}"
                acc_color = "#10B981" if checks_score >= 2.5 else "#F59E0B" if checks_score >= 1.5 else "#DC2626"
                acc_label = "Accuracy Score" if lang == "en" else "精度スコア"
                st.markdown(
                    f"""<div style="text-align:center;padding:16px;background:#F9FAFB;border-radius:12px;">
                    <div style="font-size:2em;font-weight:800;color:{acc_color};">{display_score}/{total}</div>
                    <div style="color:#6B7280;">{acc_label}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

                # Conformal coverage note
                if not sev_match and sev_in_pred_set:
                    coverage_note = (
                        "The ground truth severity is within the model's 90% confidence interval (prediction set). "
                        "This indicates the model's uncertainty quantification is well-calibrated."
                        if lang == "en"
                        else "正解の重大度はモデルの90%信頼区間（予測セット）内に含まれています。"
                        "これはモデルの不確実性定量化が適切に校正されていることを示します。"
                    )
                    st.info(coverage_note)
            else:
                st.info(
                    "Ground truth comparison is only available for demo videos."
                    if lang == "en"
                    else "正解データ比較はデモ動画のみ利用可能です。"
                )

        # ---------------------------------------------------------------
        # Download buttons
        # ---------------------------------------------------------------
        st.markdown("---")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            json_data = json.dumps(
                [a.model_dump(mode="json", exclude={"timestamp"}) for a in assessments],
                indent=2,
                ensure_ascii=False,
            )
            st.download_button(
                t("download_json"),
                data=json_data,
                file_name=f"assessment_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with dl_col2:
            if results.get("output_html_path") and Path(results["output_html_path"]).exists():
                html_content = Path(results["output_html_path"]).read_text(encoding="utf-8")
                st.download_button(
                    t("download_html"),
                    data=html_content,
                    file_name=f"report_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True,
                )

elif not results:
    st.markdown(
        f"""<div style="text-align:center;padding:60px 20px;color:#9CA3AF;">
        <div style="font-size:4em;margin-bottom:16px;">🎥</div>
        <p style="font-size:1.2em;">{t("no_results")}</p>
    </div>""",
        unsafe_allow_html=True,
    )

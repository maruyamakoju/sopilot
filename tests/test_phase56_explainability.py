"""Phase 56: XAI Explainability Engine — Test Suite (65 tests)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from sopilot.perception.explainability import (
    ExplainabilityEngine,
    RegionScore,
    SaliencyResult,
    ViolationExplanation,
    _build_jet_lut,
    _jet,
    _lookup_template,
)
from sopilot.perception.types import BBox, PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Det:
    bbox: BBox
    label: str
    confidence: float = 0.9


@dataclass
class _Viol:
    rule: str
    confidence: float = 1.0


def _make_engine(**kwargs) -> ExplainabilityEngine:
    defaults = {"output_size": (80, 60)}
    defaults.update(kwargs)
    return ExplainabilityEngine(**defaults)


def _blank_jpeg(w: int = 80, h: int = 60) -> bytes:
    from PIL import Image
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _dets():
    return [
        _Det(BBox(0.1, 0.1, 0.4, 0.5), "person", 0.9),
        _Det(BBox(0.5, 0.5, 0.9, 0.9), "forklift", 0.8),
    ]


def _viols():
    return [_Viol("ヘルメット未着用", 0.95)]


# ---------------------------------------------------------------------------
# Tests: Jet colormap helper
# ---------------------------------------------------------------------------


class TestJetColormap:
    def test_lut_shape(self):
        lut = _build_jet_lut()
        assert lut.shape == (256, 3)

    def test_lut_dtype(self):
        lut = _build_jet_lut()
        assert lut.dtype == np.uint8

    def test_jet_output_shape(self):
        gray = np.linspace(0, 1, 80 * 60).reshape(60, 80).astype(np.float32)
        out = _jet(gray)
        assert out.shape == (60, 80, 3)

    def test_jet_zero_maps_to_dark(self):
        gray = np.zeros((1, 1), dtype=np.float32)
        out = _jet(gray)
        # Jet at 0 should be dark blue — blue channel > red
        assert int(out[0, 0, 2]) > int(out[0, 0, 0])

    def test_jet_one_maps_to_red(self):
        gray = np.ones((1, 1), dtype=np.float32)
        out = _jet(gray)
        # Jet at 1 should be red — red channel > blue
        assert int(out[0, 0, 0]) > int(out[0, 0, 2])


# ---------------------------------------------------------------------------
# Tests: Template lookup
# ---------------------------------------------------------------------------


class TestTemplateLookup:
    def test_helmet_japanese(self):
        ja, en = _lookup_template("ヘルメット未着用")
        assert "ヘルメット" in ja
        assert "Helmet" in en or "helmet" in en.lower()

    def test_restricted_area(self):
        ja, en = _lookup_template("立入禁止エリア侵入")
        assert "立入禁止" in ja

    def test_forklift(self):
        ja, en = _lookup_template("フォークリフト接近")
        assert "フォークリフト" in ja

    def test_ppe_lowercase(self):
        ja, en = _lookup_template("ppe_violation")
        assert "PPE" in ja or "PPE" in en

    def test_unknown_rule_default(self):
        ja, en = _lookup_template("some_unknown_rule_xyz")
        assert len(ja) > 0
        assert len(en) > 0

    def test_safety_keyword(self):
        ja, en = _lookup_template("安全規則違反")
        assert "安全" in ja


# ---------------------------------------------------------------------------
# Tests: ExplainabilityEngine construction
# ---------------------------------------------------------------------------


class TestEngineConstruction:
    def test_default_init(self):
        e = ExplainabilityEngine()
        assert e._blend_alpha > 0
        assert e._output_size[0] > 0

    def test_custom_output_size(self):
        e = ExplainabilityEngine(output_size=(160, 120))
        assert e._output_size == (160, 120)

    def test_stats_initial(self):
        e = _make_engine()
        s = e.get_stats()
        assert s["enabled"] is True
        assert s["frames_explained"] == 0

    def test_reset_clears_stats(self):
        e = _make_engine()
        e.compute_saliency(None, _dets(), _viols())
        e.reset()
        assert e.get_stats()["frames_explained"] == 0

    def test_output_size_in_stats(self):
        e = _make_engine(output_size=(100, 75))
        assert e.get_stats()["output_size"] == (100, 75)


# ---------------------------------------------------------------------------
# Tests: compute_saliency — no frame
# ---------------------------------------------------------------------------


class TestComputeSaliencyNoFrame:
    def test_returns_saliency_result(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert isinstance(r, SaliencyResult)

    def test_frame_id_set(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols(), frame_id="test_frame")
        assert r.frame_id == "test_frame"

    def test_peak_saliency_positive_with_detections(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert r.peak_saliency > 0.0

    def test_peak_saliency_zero_no_detections(self):
        e = _make_engine()
        r = e.compute_saliency(None, [], [])
        assert r.peak_saliency == 0.0

    def test_region_scores_count(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert len(r.region_scores) == len(_dets())

    def test_violation_explanations_count(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert len(r.violation_explanations) == 1

    def test_explanation_ja_not_empty(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert len(r.explanation_ja) > 0

    def test_explanation_en_not_empty(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert len(r.explanation_en) > 0

    def test_processing_ms_positive(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert r.processing_ms >= 0.0

    def test_increments_frame_count(self):
        e = _make_engine()
        e.compute_saliency(None, _dets(), _viols())
        e.compute_saliency(None, _dets(), _viols())
        assert e.get_stats()["frames_explained"] == 2


# ---------------------------------------------------------------------------
# Tests: compute_saliency — with JPEG frame
# ---------------------------------------------------------------------------


class TestComputeSaliencyWithFrame:
    def test_heatmap_b64_not_empty(self):
        e = _make_engine()
        jpeg = _blank_jpeg()
        r = e.compute_saliency(jpeg, _dets(), _viols())
        assert len(r.heatmap_jpeg_b64) > 0

    def test_heatmap_is_valid_jpeg(self):
        from PIL import Image
        e = _make_engine()
        jpeg = _blank_jpeg()
        r = e.compute_saliency(jpeg, _dets(), _viols())
        raw = base64.b64decode(r.heatmap_jpeg_b64)
        img = Image.open(io.BytesIO(raw))
        assert img.format == "JPEG"

    def test_heatmap_dimensions(self):
        from PIL import Image
        e = _make_engine(output_size=(80, 60))
        jpeg = _blank_jpeg(80, 60)
        r = e.compute_saliency(jpeg, _dets(), _viols())
        raw = base64.b64decode(r.heatmap_jpeg_b64)
        img = Image.open(io.BytesIO(raw))
        assert img.width == 80
        assert img.height == 60

    def test_no_frame_returns_heatmap_only(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        # heatmap_jpeg_b64 may be empty or contain heatmap-only
        assert isinstance(r.heatmap_jpeg_b64, str)

    def test_with_attention_map_blended(self):
        e = _make_engine()
        jpeg = _blank_jpeg()
        attn = np.random.rand(14, 14).astype(np.float32)
        r = e.compute_saliency(jpeg, _dets(), _viols(), attention_map=attn)
        assert r.peak_saliency >= 0.0


# ---------------------------------------------------------------------------
# Tests: Proxy saliency
# ---------------------------------------------------------------------------


class TestProxySaliency:
    def test_violation_bbox_higher_saliency_than_normal(self):
        e = _make_engine(output_size=(100, 100))
        # Violation detection at top-left
        viol_det = _Det(BBox(0.0, 0.0, 0.2, 0.2), "ヘルメット", 0.9)
        # Normal detection at bottom-right
        normal_det = _Det(BBox(0.8, 0.8, 1.0, 1.0), "person", 0.9)
        viol = _Viol("ヘルメット")

        H, W = 100, 100
        sal = e._build_proxy_saliency([viol_det, normal_det], [viol], H, W)
        # Top-left region should have higher saliency
        tl = sal[0:20, 0:20].mean()
        br = sal[80:100, 80:100].mean()
        assert tl > br

    def test_empty_detections_zero_saliency(self):
        e = _make_engine()
        sal = e._build_proxy_saliency([], [], 60, 80)
        assert sal.max() == 0.0

    def test_confidence_scales_saliency(self):
        e = _make_engine(output_size=(100, 100))
        det_high = _Det(BBox(0.45, 0.45, 0.55, 0.55), "obj", 1.0)
        det_low = _Det(BBox(0.45, 0.45, 0.55, 0.55), "obj", 0.1)
        sal_high = e._build_proxy_saliency([det_high], [], 100, 100)
        sal_low = e._build_proxy_saliency([det_low], [], 100, 100)
        assert sal_high.max() > sal_low.max()

    def test_saliency_non_negative(self):
        e = _make_engine()
        sal = e._build_proxy_saliency(_dets(), _viols(), 60, 80)
        assert sal.min() >= 0.0


# ---------------------------------------------------------------------------
# Tests: RegionScore
# ---------------------------------------------------------------------------


class TestRegionScores:
    def test_region_scores_sorted_by_score_desc(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        scores = [rs.score for rs in r.region_scores]
        assert scores == sorted(scores, reverse=True)

    def test_violation_flagged_correctly(self):
        e = _make_engine()
        det_viol = _Det(BBox(0.0, 0.0, 0.5, 0.5), "ヘルメット", 0.9)
        det_norm = _Det(BBox(0.5, 0.5, 1.0, 1.0), "person", 0.9)
        viol = _Viol("ヘルメット")
        r = e.compute_saliency(None, [det_viol, det_norm], [viol])
        viol_regions = [rs for rs in r.region_scores if rs.is_violation]
        assert len(viol_regions) >= 1

    def test_region_bbox_stored(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), [])
        for rs in r.region_scores:
            assert 0.0 <= rs.x1 <= 1.0
            assert rs.x1 < rs.x2


# ---------------------------------------------------------------------------
# Tests: ViolationExplanation
# ---------------------------------------------------------------------------


class TestViolationExplanation:
    def test_explain_violations_returns_list(self):
        e = _make_engine()
        expls = e.explain_violations(_viols())
        assert isinstance(expls, list)
        assert len(expls) == 1

    def test_explanation_has_rule(self):
        e = _make_engine()
        expls = e.explain_violations([_Viol("ヘルメット未着用")])
        assert expls[0].rule == "ヘルメット未着用"

    def test_explanation_confidence(self):
        e = _make_engine()
        expls = e.explain_violations([_Viol("ヘルメット", 0.87)])
        assert abs(expls[0].confidence - 0.87) < 0.01

    def test_empty_violations_returns_empty(self):
        e = _make_engine()
        assert e.explain_violations([]) == []

    def test_multiple_violations(self):
        e = _make_engine()
        viols = [_Viol("ヘルメット"), _Viol("立入禁止"), _Viol("フォークリフト")]
        expls = e.explain_violations(viols)
        assert len(expls) == 3


# ---------------------------------------------------------------------------
# Tests: frame explanation
# ---------------------------------------------------------------------------


class TestFrameExplanation:
    def test_no_violations_message(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), [])
        assert "検出されません" in r.explanation_ja or "no safety violation" in r.explanation_en.lower()

    def test_violations_mentioned_in_explanation(self):
        e = _make_engine()
        r = e.compute_saliency(None, _dets(), _viols())
        assert "1件" in r.explanation_ja or "違反" in r.explanation_ja


# ---------------------------------------------------------------------------
# Tests: stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_avg_processing_ms_updates(self):
        e = _make_engine()
        e.compute_saliency(None, _dets(), _viols())
        s = e.get_stats()
        assert s["avg_processing_ms"] >= 0.0

    def test_violation_count_accumulates(self):
        e = _make_engine()
        e.compute_saliency(None, _dets(), [_Viol("A"), _Viol("B")])
        e.compute_saliency(None, _dets(), [_Viol("C")])
        s = e.get_stats()
        assert s["total_violations_explained"] == 3

    def test_last_peak_saliency_updated(self):
        e = _make_engine()
        e.compute_saliency(None, _dets(), _viols())
        s = e.get_stats()
        assert s["last_peak_saliency"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: engine.py integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_explainability_attr_exists(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert hasattr(engine, "_explainability_engine")
        assert engine._explainability_engine is None

    def test_last_saliency_attr_exists(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert hasattr(engine, "_last_saliency")
        assert engine._last_saliency is None

    def test_get_explainability_state_none_when_disabled(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig()
        engine = PerceptionEngine(config)
        assert engine.get_explainability_state() is None

    def test_enabled_via_config(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(explainability_enabled=True)
        engine = build_perception_engine(config)
        assert engine._explainability_engine is not None

    def test_get_state_when_enabled(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(explainability_enabled=True)
        engine = build_perception_engine(config)
        s = engine.get_explainability_state()
        assert s is not None
        assert s["enabled"] is True


# ---------------------------------------------------------------------------
# Tests: API endpoints
# ---------------------------------------------------------------------------


class TestExplainabilityAPI:
    def _make_client(self, enabled: bool = True):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sopilot.vigil.perception_router import build_perception_router

        xai = ExplainabilityEngine(output_size=(80, 60)) if enabled else None

        engine_mock = MagicMock()
        engine_mock._explainability_engine = xai
        engine_mock._last_saliency = None
        engine_mock.get_explainability_state = (
            (lambda: xai.get_stats()) if xai else (lambda: None)
        )

        class _FakePipeline:
            _vlm = MagicMock()
            _vlm._engine = engine_mock

        app = FastAPI()
        app.state.vigil_pipeline = _FakePipeline()
        app.include_router(build_perception_router())
        return TestClient(app)

    def test_stats_disabled(self):
        client = self._make_client(enabled=False)
        r = client.get("/vigil/perception/explainability/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_stats_enabled(self):
        client = self._make_client()
        r = client.get("/vigil/perception/explainability/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is True

    def test_latest_no_frame(self):
        client = self._make_client()
        r = client.get("/vigil/perception/explainability/latest")
        assert r.status_code == 200
        assert r.json()["available"] is False

    def test_saliency_post_empty(self):
        client = self._make_client()
        r = client.post("/vigil/perception/explainability/saliency", json={
            "detections": [], "violations": []
        })
        assert r.status_code == 200
        assert r.json()["enabled"] is True

    def test_saliency_post_with_detections(self):
        client = self._make_client()
        r = client.post("/vigil/perception/explainability/saliency", json={
            "detections": [
                {"bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.5},
                 "label": "person", "confidence": 0.9}
            ],
            "violations": [{"rule": "ヘルメット", "confidence": 0.95}],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is True
        assert "explanation_ja" in d

    def test_explain_endpoint(self):
        client = self._make_client()
        r = client.post("/vigil/perception/explainability/explain", json={
            "violations": [{"rule": "ヘルメット未着用", "confidence": 0.9}]
        })
        assert r.status_code == 200
        d = r.json()
        assert len(d["explanations"]) == 1
        assert "explanation_ja" in d["explanations"][0]

    def test_reset_endpoint(self):
        client = self._make_client()
        r = client.request("DELETE", "/vigil/perception/explainability/reset")
        assert r.status_code == 200
        assert r.json()["success"] is True

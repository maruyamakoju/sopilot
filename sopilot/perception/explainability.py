"""Phase 56: XAI — Explainability Engine.

検出・違反の「なぜ?」に視覚的・言語的に答えるモジュール。

主要機能:
  - **プロキシサリエンシーマップ**: 検出 bbox + confidence から 2D ガウスブロブを生成。
    違反 bbox は強調 (weight×3)。GPU 不要・純 NumPy 実装。
  - **ヒートマップオーバーレイ**: jet カラーマップ → 元フレーム JPEG へ RGBA ブレンド。
    PIL のみ使用 (cv2 不要)。
  - **違反説明生成**: テンプレートベースの日本語・英語 NL 説明。
  - **最新フレームキャッシュ**: engine.py Stage 6t で `_last_saliency` を更新。
  - **DINOv2 注意マップ統合**: 利用可能な場合は attention_weights を混合。

Architecture::

    ExplainabilityEngine
      ├── compute_saliency(frame_jpeg, detections, violations, attention_map?)
      │       → SaliencyResult(heatmap_jpeg, region_scores, explanation_ja, explanation_en)
      ├── explain_violations(violations, detections)
      │       → list[ViolationExplanation]
      ├── get_stats()                         — 処理フレーム数・平均ピーク強度等
      └── reset()

Design decisions:
  - PIL のみで JPEG 入出力 + ブレンド → cv2 依存なし
  - サリエンシーは float32 [0,1] グリッド (同フレーム解像度)
  - ブレンド alpha=0.45 (ヒートマップ) — 元画像を損なわない
  - スレッドセーフ: RLock でキャッシュ保護
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLEND_ALPHA = 0.45        # heatmap overlay opacity
_GAUSSIAN_SIGMA_FRAC = 0.12  # Gaussian sigma as fraction of image shorter side
_VIOLATION_WEIGHT = 3.0    # violation bbox weight multiplier
_DETECTION_WEIGHT = 1.0    # regular detection weight

# Jet colormap LUT (256 entries, RGB)
_JET_LUT: np.ndarray | None = None


def _build_jet_lut() -> np.ndarray:
    """Build a 256×3 uint8 jet colormap lookup table."""
    t = np.linspace(0.0, 1.0, 256)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    lut = np.stack([r, g, b], axis=-1)
    return (lut * 255).astype(np.uint8)


def _jet(gray: np.ndarray) -> np.ndarray:
    """Apply jet colormap to a float [0,1] 2D array → uint8 H×W×3."""
    global _JET_LUT
    if _JET_LUT is None:
        _JET_LUT = _build_jet_lut()
    idx = np.clip((gray * 255).astype(np.int32), 0, 255)
    return _JET_LUT[idx]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RegionScore:
    """Saliency score for a spatial region."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float         # [0, 1]
    label: str
    is_violation: bool


@dataclass
class ViolationExplanation:
    """Natural language explanation for a single violation."""
    rule: str
    confidence: float
    explanation_ja: str
    explanation_en: str
    region: RegionScore | None = None


@dataclass
class SaliencyResult:
    """Output of ExplainabilityEngine.compute_saliency()."""
    heatmap_jpeg_b64: str          # base64-encoded JPEG
    region_scores: list[RegionScore]
    violation_explanations: list[ViolationExplanation]
    peak_saliency: float           # max saliency value [0,1]
    explanation_ja: str            # overall frame explanation (JP)
    explanation_en: str            # overall frame explanation (EN)
    frame_id: str
    processing_ms: float


# ---------------------------------------------------------------------------
# Japanese / English explanation templates
# ---------------------------------------------------------------------------

_JP_TEMPLATES: dict[str, str] = {
    "ヘルメット":    "ヘルメット未着用が検出されました。頭部保護が必要なエリアでの違反です。",
    "helmet":       "ヘルメット未着用が検出されました。頭部保護が必要なエリアでの違反です。",
    "立入禁止":      "立入禁止エリアへの侵入が検出されました。即時退去が必要です。",
    "フォークリフト": "フォークリフトと作業員の接近が検出されました。衝突リスクが高い状態です。",
    "安全":         "安全規則違反が検出されました。現場管理者に報告してください。",
    "ppe":          "PPE 未装着が検出されました。装備を確認してください。",
}
_EN_TEMPLATES: dict[str, str] = {
    "ヘルメット":    "Helmet violation detected: worker in hazard zone without head protection.",
    "helmet":       "Helmet violation detected: worker in hazard zone without head protection.",
    "立入禁止":      "Restricted-area intrusion detected: immediate evacuation required.",
    "フォークリフト": "Forklift proximity detected: high collision risk, stop operations.",
    "安全":         "Safety rule violation detected: report to site manager immediately.",
    "ppe":          "PPE non-compliance detected: check required protective equipment.",
}
_DEFAULT_JP = "安全違反が検出されました。現場を確認してください。"
_DEFAULT_EN = "Safety violation detected. Please inspect the site."


def _lookup_template(rule: str) -> tuple[str, str]:
    rule_lower = rule.lower()
    for key, tmpl in _JP_TEMPLATES.items():
        if key.lower() in rule_lower:
            return tmpl, _EN_TEMPLATES[key]
    return _DEFAULT_JP, _DEFAULT_EN


# ---------------------------------------------------------------------------
# ExplainabilityEngine
# ---------------------------------------------------------------------------


class ExplainabilityEngine:
    """GPU-free XAI engine: proxy saliency maps + NL violation explanations.

    Saliency computation:
        For each detected bbox (x1,y1,x2,y2) in normalized coords:
          1. Place a 2D Gaussian centered at bbox centroid.
          2. Sigma = _GAUSSIAN_SIGMA_FRAC × min(H, W).
          3. Weight = confidence × (VIOLATION_WEIGHT if violated else DETECTION_WEIGHT).
        Sum all Gaussian contributions → normalize to [0,1].

    Optional: if attention_map (H×W float32) is provided from DINOv2,
    blend 40% attention + 60% proxy saliency.
    """

    def __init__(
        self,
        blend_alpha: float = _BLEND_ALPHA,
        gaussian_sigma_frac: float = _GAUSSIAN_SIGMA_FRAC,
        output_size: tuple[int, int] = (320, 240),  # W, H of heatmap output
    ) -> None:
        self._blend_alpha = blend_alpha
        self._gaussian_sigma_frac = gaussian_sigma_frac
        self._output_size = output_size  # (W, H)

        self._lock = threading.RLock()

        # Stats
        self._frames_explained: int = 0
        self._total_violations_explained: int = 0
        self._total_processing_ms: float = 0.0
        self._last_peak_saliency: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def compute_saliency(
        self,
        frame_jpeg: bytes | None,
        detections: list[Any],
        violations: list[Any],
        attention_map: np.ndarray | None = None,
        frame_id: str = "",
    ) -> SaliencyResult:
        """Compute saliency map and generate explanations for a frame.

        Args:
            frame_jpeg:    Raw JPEG bytes of the source frame (may be None).
            detections:    List of Detection objects (must have .bbox and .label).
            violations:    List of Violation objects (must have .rule).
            attention_map: Optional H×W float32 attention weights from DINOv2.
            frame_id:      Identifier for logging / caching.

        Returns:
            SaliencyResult with base64-encoded JPEG heatmap overlay.
        """
        t0 = time.perf_counter()

        W, H = self._output_size

        # 1. Build proxy saliency
        saliency = self._build_proxy_saliency(detections, violations, H, W)

        # 2. Optionally blend with DINOv2 attention map
        if attention_map is not None:
            try:
                att = np.array(attention_map, dtype=np.float32)
                if att.ndim == 2 and att.size > 0:
                    from PIL import Image as _PILImage
                    att_resized = np.array(
                        _PILImage.fromarray(att).resize((W, H), _PILImage.BILINEAR)
                    )
                    att_norm = att_resized / (att_resized.max() + 1e-8)
                    saliency = 0.6 * saliency + 0.4 * att_norm
            except Exception:
                pass  # fallback to proxy saliency only

        # Normalize
        peak = float(saliency.max())
        if peak > 1e-8:
            saliency = saliency / peak
        self._last_peak_saliency = peak

        # 3. Build heatmap JPEG
        heatmap_jpeg_b64 = self._render_heatmap(frame_jpeg, saliency, W, H)

        # 4. Region scores
        region_scores = self._compute_region_scores(detections, violations, saliency, H, W)

        # 5. Violation explanations
        viol_explanations = self.explain_violations(violations, detections, region_scores)

        # 6. Overall frame explanation
        expl_ja, expl_en = self._frame_explanation(violations, detections, peak)

        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._frames_explained += 1
            self._total_violations_explained += len(violations)
            self._total_processing_ms += ms

        import base64
        return SaliencyResult(
            heatmap_jpeg_b64=heatmap_jpeg_b64,
            region_scores=region_scores,
            violation_explanations=viol_explanations,
            peak_saliency=round(float(saliency.max()), 4),
            explanation_ja=expl_ja,
            explanation_en=expl_en,
            frame_id=frame_id or f"frame_{self._frames_explained}",
            processing_ms=round(ms, 2),
        )

    def explain_violations(
        self,
        violations: list[Any],
        detections: list[Any] | None = None,
        region_scores: list[RegionScore] | None = None,
    ) -> list[ViolationExplanation]:
        """Generate NL explanations for a list of violations."""
        explanations = []
        for v in violations:
            rule = getattr(v, "rule", str(v))
            confidence = getattr(v, "confidence", 1.0)
            jp, en = _lookup_template(rule)

            # Find matching region score if available
            region = None
            if region_scores:
                for rs in region_scores:
                    if rs.is_violation:
                        region = rs
                        break

            explanations.append(ViolationExplanation(
                rule=rule,
                confidence=round(float(confidence), 3),
                explanation_ja=jp,
                explanation_en=en,
                region=region,
            ))
        return explanations

    def get_stats(self) -> dict:
        """Return engine statistics."""
        with self._lock:
            avg_ms = (
                self._total_processing_ms / self._frames_explained
                if self._frames_explained > 0
                else 0.0
            )
            return {
                "enabled": True,
                "frames_explained": self._frames_explained,
                "total_violations_explained": self._total_violations_explained,
                "avg_processing_ms": round(avg_ms, 2),
                "last_peak_saliency": round(self._last_peak_saliency, 4),
                "output_size": self._output_size,
                "blend_alpha": self._blend_alpha,
            }

    def reset(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._frames_explained = 0
            self._total_violations_explained = 0
            self._total_processing_ms = 0.0
            self._last_peak_saliency = 0.0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_proxy_saliency(
        self,
        detections: list[Any],
        violations: list[Any],
        H: int,
        W: int,
    ) -> np.ndarray:
        """Build H×W float32 saliency map from detection bboxes."""
        saliency = np.zeros((H, W), dtype=np.float32)
        if not detections and not violations:
            return saliency

        sigma = self._gaussian_sigma_frac * min(H, W)

        # Collect violation rules for quick lookup
        violation_rules: set[str] = set()
        for v in violations:
            r = getattr(v, "rule", "")
            if r:
                violation_rules.add(r)

        # Build coordinate grids for vectorized Gaussian
        ys, xs = np.mgrid[0:H, 0:W]

        for det in detections:
            bbox = getattr(det, "bbox", None)
            if bbox is None:
                continue
            cx = ((getattr(bbox, "x1", 0) + getattr(bbox, "x2", 1)) / 2.0) * W
            cy = ((getattr(bbox, "y1", 0) + getattr(bbox, "y2", 1)) / 2.0) * H
            label = getattr(det, "label", "")
            confidence = float(getattr(det, "confidence", 1.0))
            is_viol = label in violation_rules or any(
                label.lower() in r.lower() for r in violation_rules
            )
            weight = confidence * (_VIOLATION_WEIGHT if is_viol else _DETECTION_WEIGHT)

            dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
            blob = weight * np.exp(-dist_sq / (2.0 * sigma ** 2 + 1e-8))
            saliency += blob

        return saliency

    def _render_heatmap(
        self,
        frame_jpeg: bytes | None,
        saliency: np.ndarray,
        W: int,
        H: int,
    ) -> str:
        """Blend saliency heatmap over frame → base64 JPEG string."""
        import base64
        try:
            from PIL import Image

            # 1. Jet colormap on saliency
            jet_rgb = _jet(saliency)           # H×W×3 uint8
            heat_img = Image.fromarray(jet_rgb, "RGB")

            if frame_jpeg:
                # 2. Decode source frame
                src = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
                src = src.resize((W, H), Image.BILINEAR)

                # 3. Blend: out = alpha * heat + (1-alpha) * src
                alpha = int(self._blend_alpha * 255)
                heat_rgba = heat_img.convert("RGBA")
                src_rgba = src.convert("RGBA")
                # Apply saliency-adaptive alpha: low saliency → more transparent
                sal_alpha = (saliency * self._blend_alpha * 255).astype(np.uint8)
                alpha_ch = Image.fromarray(sal_alpha, "L")
                heat_rgba.putalpha(alpha_ch)
                blended = Image.alpha_composite(src_rgba, heat_rgba).convert("RGB")
            else:
                blended = heat_img

            buf = io.BytesIO()
            blended.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode()

        except Exception:
            logger.debug("Heatmap render failed", exc_info=True)
            return ""

    def _compute_region_scores(
        self,
        detections: list[Any],
        violations: list[Any],
        saliency: np.ndarray,
        H: int,
        W: int,
    ) -> list[RegionScore]:
        """Extract per-bbox average saliency scores."""
        violation_rules: set[str] = set()
        for v in violations:
            r = getattr(v, "rule", "")
            if r:
                violation_rules.add(r)

        scores = []
        for det in detections:
            bbox = getattr(det, "bbox", None)
            if bbox is None:
                continue
            x1 = int(getattr(bbox, "x1", 0) * W)
            y1 = int(getattr(bbox, "y1", 0) * H)
            x2 = int(getattr(bbox, "x2", 1) * W)
            y2 = int(getattr(bbox, "y2", 1) * H)
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            label = getattr(det, "label", "")
            is_viol = label in violation_rules or any(
                label.lower() in r.lower() for r in violation_rules
            )
            roi = saliency[y1:y2, x1:x2]
            score = float(roi.mean()) if roi.size > 0 else 0.0
            scores.append(RegionScore(
                x1=getattr(bbox, "x1", 0),
                y1=getattr(bbox, "y1", 0),
                x2=getattr(bbox, "x2", 1),
                y2=getattr(bbox, "y2", 1),
                score=round(score, 4),
                label=label,
                is_violation=is_viol,
            ))

        # Sort by score descending
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores

    def _frame_explanation(
        self,
        violations: list[Any],
        detections: list[Any],
        peak_saliency: float,
    ) -> tuple[str, str]:
        """Build overall frame-level NL explanation."""
        n_det = len(detections)
        n_viol = len(violations)

        if n_viol == 0:
            ja = f"このフレームでは{n_det}件の検出がありましたが、安全違反は検出されませんでした。"
            en = f"{n_det} detection(s) in this frame; no safety violations detected."
            return ja, en

        rules = [getattr(v, "rule", "違反") for v in violations[:3]]
        rules_str = "、".join(rules)
        intensity = "高強度" if peak_saliency > 0.7 else "中強度" if peak_saliency > 0.3 else "低強度"
        ja = (
            f"{n_viol}件の違反が検出されました: {rules_str}。"
            f"サリエンシー強度: {intensity} (peak={peak_saliency:.2f})。"
            f"総検出数: {n_det}件。"
        )
        en = (
            f"{n_viol} violation(s) detected: {rules_str}. "
            f"Saliency intensity: {intensity} (peak={peak_saliency:.2f}). "
            f"Total detections: {n_det}."
        )
        return ja, en

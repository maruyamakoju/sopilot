"""Video quality validation for SOP compliance scoring.

Performs pre-scoring quality checks on uploaded videos to ensure
reliable scoring results.  Implements the software component of the
Capture Kit standardization concept.

Checks:
    - Brightness: Mean pixel intensity across sampled frames
    - Sharpness:  Laplacian variance (focus quality)
    - Stability:  Frame-to-frame pixel difference (camera shake)
    - Duration:   Minimum video length for meaningful scoring
    - Resolution: Minimum frame dimensions

The checker is designed to reject videos that would produce unreliable
scores, and to provide actionable Japanese-language feedback so the
operator can fix the issue and re-record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Default thresholds ─────────────────────────────────────────────
_MIN_BRIGHTNESS = 40.0
_MAX_BRIGHTNESS = 220.0
_MIN_SHARPNESS = 50.0
_MAX_FRAME_DIFF = 80.0
_MIN_DURATION_SEC = 3.0
_MIN_RESOLUTION = 64
_SAMPLE_COUNT = 20


# ── Data classes ───────────────────────────────────────────────────

@dataclass(frozen=True)
class QualityCheck:
    """Result of a single quality dimension check."""

    name: str
    passed: bool
    value: float
    threshold: float
    message_ja: str
    message_en: str


@dataclass(frozen=True)
class VideoQualityReport:
    """Comprehensive video quality assessment."""

    overall_pass: bool
    checks: list[QualityCheck]
    recommendations_ja: list[str]
    recommendations_en: list[str]
    frame_count_sampled: int
    duration_sec: float = 0.0
    resolution: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        return {
            "overall_pass": self.overall_pass,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": round(c.value, 2),
                    "threshold": c.threshold,
                    "message_ja": c.message_ja,
                    "message_en": c.message_en,
                }
                for c in self.checks
            ],
            "recommendations_ja": self.recommendations_ja,
            "recommendations_en": self.recommendations_en,
            "frame_count_sampled": self.frame_count_sampled,
            "duration_sec": round(self.duration_sec, 1),
            "resolution": list(self.resolution),
        }


# ── Checker ────────────────────────────────────────────────────────

class VideoQualityChecker:
    """Validates video quality before scoring.

    Usage::

        checker = VideoQualityChecker()
        report = checker.check("/path/to/video.mp4")
        if not report.overall_pass:
            # reject the video with actionable feedback
            print(report.recommendations_ja)
    """

    def __init__(
        self,
        *,
        min_brightness: float = _MIN_BRIGHTNESS,
        max_brightness: float = _MAX_BRIGHTNESS,
        min_sharpness: float = _MIN_SHARPNESS,
        max_frame_diff: float = _MAX_FRAME_DIFF,
        min_duration_sec: float = _MIN_DURATION_SEC,
        min_resolution: int = _MIN_RESOLUTION,
    ) -> None:
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.max_frame_diff = max_frame_diff
        self.min_duration_sec = min_duration_sec
        self.min_resolution = min_resolution

    def check(self, video_path: str | Path) -> VideoQualityReport:
        """Run all quality checks on a video file."""
        path = str(video_path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return VideoQualityReport(
                overall_pass=False,
                checks=[
                    QualityCheck(
                        name="readable",
                        passed=False,
                        value=0.0,
                        threshold=1.0,
                        message_ja="動画ファイルを開けません",
                        message_en="Cannot open video file",
                    )
                ],
                recommendations_ja=["動画ファイルが破損していないか確認してください"],
                recommendations_en=["Check if the video file is corrupted"],
                frame_count_sampled=0,
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps if fps > 0 else 0.0

            frames = self._sample_frames(cap, total_frames, _SAMPLE_COUNT)

            checks: list[QualityCheck] = []
            recs_ja: list[str] = []
            recs_en: list[str] = []

            # 1. Duration ───────────────────────────────────────────
            dur_ok = duration_sec >= self.min_duration_sec
            checks.append(
                QualityCheck(
                    name="duration",
                    passed=dur_ok,
                    value=round(duration_sec, 1),
                    threshold=self.min_duration_sec,
                    message_ja=(
                        f"動画長: {duration_sec:.1f}秒"
                        if dur_ok
                        else f"動画が短すぎます（{duration_sec:.1f}秒 < {self.min_duration_sec}秒）"
                    ),
                    message_en=(
                        f"Duration: {duration_sec:.1f}s"
                        if dur_ok
                        else f"Video too short ({duration_sec:.1f}s < {self.min_duration_sec}s)"
                    ),
                )
            )
            if not dur_ok:
                recs_ja.append("より長い動画を撮影してください")
                recs_en.append("Record a longer video")

            # 2. Resolution ─────────────────────────────────────────
            min_dim = min(width, height)
            res_ok = min_dim >= self.min_resolution
            checks.append(
                QualityCheck(
                    name="resolution",
                    passed=res_ok,
                    value=float(min_dim),
                    threshold=float(self.min_resolution),
                    message_ja=(
                        f"解像度: {width}×{height}"
                        if res_ok
                        else f"解像度が低すぎます（{width}×{height}）"
                    ),
                    message_en=(
                        f"Resolution: {width}×{height}"
                        if res_ok
                        else f"Resolution too low ({width}×{height})"
                    ),
                )
            )
            if not res_ok:
                recs_ja.append("より高い解像度で撮影してください")
                recs_en.append("Record at a higher resolution")

            if not frames:
                return VideoQualityReport(
                    overall_pass=False,
                    checks=checks,
                    recommendations_ja=["フレームを読み取れませんでした"],
                    recommendations_en=["Could not read frames from video"],
                    frame_count_sampled=0,
                    duration_sec=duration_sec,
                    resolution=(width, height),
                )

            # 3. Brightness ─────────────────────────────────────────
            brightnesses = [float(np.mean(f)) for f in frames]
            mean_brightness = float(np.mean(brightnesses))
            too_dark = mean_brightness < self.min_brightness
            too_bright = mean_brightness > self.max_brightness
            bright_ok = not too_dark and not too_bright
            if too_dark:
                msg_ja = f"照度不足（平均輝度: {mean_brightness:.0f} < {self.min_brightness:.0f}）"
                msg_en = f"Too dark (mean brightness: {mean_brightness:.0f})"
                recs_ja.append("照明を追加するか、カメラの露出を上げてください")
                recs_en.append("Add more lighting or increase camera exposure")
            elif too_bright:
                msg_ja = f"露出過多（平均輝度: {mean_brightness:.0f} > {self.max_brightness:.0f}）"
                msg_en = f"Overexposed (mean brightness: {mean_brightness:.0f})"
                recs_ja.append("照明を抑えるか、カメラの露出を下げてください")
                recs_en.append("Reduce lighting or decrease camera exposure")
            else:
                msg_ja = f"照度: 適正（平均輝度: {mean_brightness:.0f}）"
                msg_en = f"Brightness: OK (mean: {mean_brightness:.0f})"
            checks.append(
                QualityCheck(
                    name="brightness",
                    passed=bright_ok,
                    value=mean_brightness,
                    threshold=self.min_brightness if too_dark else self.max_brightness,
                    message_ja=msg_ja,
                    message_en=msg_en,
                )
            )

            # 4. Sharpness ──────────────────────────────────────────
            sharpnesses: list[float] = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpnesses.append(lap_var)
            mean_sharpness = float(np.mean(sharpnesses))
            sharp_ok = mean_sharpness >= self.min_sharpness
            checks.append(
                QualityCheck(
                    name="sharpness",
                    passed=sharp_ok,
                    value=mean_sharpness,
                    threshold=self.min_sharpness,
                    message_ja=(
                        f"鮮鋭度: 適正（{mean_sharpness:.0f}）"
                        if sharp_ok
                        else f"ピンボケ（鮮鋭度: {mean_sharpness:.0f} < {self.min_sharpness:.0f}）"
                    ),
                    message_en=(
                        f"Sharpness: OK ({mean_sharpness:.0f})"
                        if sharp_ok
                        else f"Blurry (sharpness: {mean_sharpness:.0f} < {self.min_sharpness:.0f})"
                    ),
                )
            )
            if not sharp_ok:
                recs_ja.append("カメラのピントを確認し、レンズを清掃してください")
                recs_en.append("Check camera focus and clean the lens")

            # 5. Stability ──────────────────────────────────────────
            if len(frames) >= 2:
                diffs: list[float] = []
                for i in range(1, len(frames)):
                    prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY) if len(frames[i - 1].shape) == 3 else frames[i - 1]
                    curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) if len(frames[i].shape) == 3 else frames[i]
                    # Handle size mismatch defensively
                    if prev_gray.shape != curr_gray.shape:
                        h = min(prev_gray.shape[0], curr_gray.shape[0])
                        w = min(prev_gray.shape[1], curr_gray.shape[1])
                        prev_gray = prev_gray[:h, :w]
                        curr_gray = curr_gray[:h, :w]
                    diff = float(np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))))
                    diffs.append(diff)
                mean_diff = float(np.mean(diffs))
                stable_ok = mean_diff <= self.max_frame_diff
                checks.append(
                    QualityCheck(
                        name="stability",
                        passed=stable_ok,
                        value=mean_diff,
                        threshold=self.max_frame_diff,
                        message_ja=(
                            f"安定性: 適正（フレーム差分: {mean_diff:.1f}）"
                            if stable_ok
                            else f"手ブレ検出（フレーム差分: {mean_diff:.1f} > {self.max_frame_diff:.1f}）"
                        ),
                        message_en=(
                            f"Stability: OK (frame diff: {mean_diff:.1f})"
                            if stable_ok
                            else f"Camera shake detected (frame diff: {mean_diff:.1f})"
                        ),
                    )
                )
                if not stable_ok:
                    recs_ja.append("三脚を使用するか、カメラを固定してください")
                    recs_en.append("Use a tripod or stabilize the camera")

            overall_pass = all(c.passed for c in checks)

            return VideoQualityReport(
                overall_pass=overall_pass,
                checks=checks,
                recommendations_ja=recs_ja,
                recommendations_en=recs_en,
                frame_count_sampled=len(frames),
                duration_sec=duration_sec,
                resolution=(width, height),
            )
        finally:
            cap.release()

    @staticmethod
    def _sample_frames(
        cap: cv2.VideoCapture,
        total_frames: int,
        n: int,
    ) -> list[np.ndarray]:
        """Sample *n* evenly-spaced frames from the video."""
        if total_frames <= 0:
            return []
        indices = np.linspace(0, max(total_frames - 1, 0), min(n, total_frames), dtype=int)
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        return frames

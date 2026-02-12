"""Tests for nn.explainability — Temporal attention and counterfactual explanations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sopilot.nn.explainability import (
    CounterfactualExplainer,
    TemporalAttentionVisualizer,
)
from sopilot.nn.scoring_head import METRIC_KEYS, N_METRICS, ScoringHead


class TestTemporalAttentionVisualizer:
    def test_alignment_heatmap_basic(self) -> None:
        alignment = np.random.rand(10, 15).astype(np.float32)
        result = TemporalAttentionVisualizer.alignment_heatmap(alignment)

        assert result["heatmap"].shape == (10, 15)
        assert len(result["gold_importance"]) == 10
        assert len(result["trainee_importance"]) == 15
        assert isinstance(result["peak_alignment"], list)

    def test_heatmap_normalized_to_unit(self) -> None:
        alignment = np.random.rand(5, 8).astype(np.float32) * 100
        result = TemporalAttentionVisualizer.alignment_heatmap(alignment)
        assert result["heatmap"].max() <= 1.0 + 1e-6

    def test_peak_alignment_sorted(self) -> None:
        alignment = np.random.rand(8, 8).astype(np.float32)
        result = TemporalAttentionVisualizer.alignment_heatmap(alignment)
        peaks = result["peak_alignment"]
        if len(peaks) > 1:
            strengths = [p[2] for p in peaks]
            assert strengths == sorted(strengths, reverse=True)

    def test_step_alignment_summary(self) -> None:
        alignment = np.random.rand(10, 12).astype(np.float32)
        gold_boundaries = [0, 3, 7, 10]
        trainee_boundaries = [0, 4, 8, 12]

        summaries = TemporalAttentionVisualizer.step_alignment_summary(alignment, gold_boundaries, trainee_boundaries)

        assert len(summaries) == 3  # 3 gold steps
        for s in summaries:
            assert "gold_step" in s
            assert "alignment_mass" in s
            assert "mean_strength" in s
            assert "best_trainee_step" in s


class TestCounterfactualExplainer:
    def _make_explainer(self) -> tuple[CounterfactualExplainer, torch.Tensor]:
        model = ScoringHead()
        model.eval()
        explainer = CounterfactualExplainer(model)
        metrics = torch.randn(1, N_METRICS)
        return explainer, metrics

    def test_compute_sensitivity(self) -> None:
        explainer, metrics = self._make_explainer()
        sensitivity = explainer.compute_sensitivity(metrics)
        assert len(sensitivity) == N_METRICS
        for k in METRIC_KEYS:
            assert k in sensitivity
            assert isinstance(sensitivity[k], float)

    def test_counterfactual(self) -> None:
        explainer, metrics = self._make_explainer()
        result = explainer.counterfactual(metrics, "miss", -1.0)
        assert "current_score" in result
        assert "predicted_score" in result
        assert "delta" in result
        assert result["metric_name"] == "miss"
        assert result["metric_change"] == -1.0

    def test_counterfactual_unknown_metric(self) -> None:
        explainer, metrics = self._make_explainer()
        with pytest.raises(ValueError, match="Unknown metric"):
            explainer.counterfactual(metrics, "nonexistent", 1.0)

    def test_top_actionable_improvements(self) -> None:
        explainer, metrics = self._make_explainer()
        improvements = explainer.top_actionable_improvements(metrics, n_top=3)
        assert len(improvements) <= 3
        for imp in improvements:
            assert "metric" in imp
            assert "sensitivity" in imp
            assert "direction_to_improve" in imp
            assert "potential_gain_per_unit" in imp

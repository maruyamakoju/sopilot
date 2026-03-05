"""SOPilot Closed-Loop Perception Demo (Phase 13–16).

Demonstrates the full autonomous pipeline end-to-end with synthetic data:

  検出 → 異常 → 学習 → 予兆検知 → 自律対応

Stages shown:
  Phase 13/14  — SigmaTuner (self-learning σ), ReviewQueue (active query)
  Phase 15     — EarlyWarningEngine (composite risk score)
  Phase 16     — EarlyWarningResponder (autonomous actions + JP explanation)

Usage:
    python scripts/demo_closed_loop.py
    python scripts/demo_closed_loop.py --rounds 3 --verbose
    python scripts/demo_closed_loop.py --no-color
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

# ── Terminal colours ──────────────────────────────────────────────────────────

USE_COLOR = True  # toggled by --no-color


def _c(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


RED    = lambda t: _c("91", t)
YELLOW = lambda t: _c("93", t)
GREEN  = lambda t: _c("92", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    bar = "█" * filled + "░" * (width - filled)
    if value >= 0.6:
        return RED(bar)
    if value >= 0.3:
        return YELLOW(bar)
    return GREEN(bar)


def _risk_badge(level: str) -> str:
    badges = {"HIGH": RED("[HIGH]"), "MEDIUM": YELLOW("[MED] "), "LOW": GREEN("[LOW] ")}
    return badges.get(level.upper(), DIM(f"[{level[:3]}]"))


# ── Synthetic anomaly data generator ─────────────────────────────────────────

DETECTORS = ("behavioral", "spatial", "temporal", "interaction")


@dataclass
class SyntheticScenario:
    """One simulated 'shift' of camera data."""
    name: str
    behavioral_fp_rate: float    # current FP rate
    spatial_fp_rate: float
    temporal_fp_rate: float
    interaction_fp_rate: float
    sigma_drift: dict[str, float] = field(default_factory=dict)   # σ/min
    burst_events: dict[str, int]  = field(default_factory=dict)   # events in burst window


SCENARIOS = [
    SyntheticScenario(
        name="平常稼働 (Normal Operation)",
        behavioral_fp_rate=0.05, spatial_fp_rate=0.08,
        temporal_fp_rate=0.06, interaction_fp_rate=0.04,
        sigma_drift={"behavioral": 0.02, "spatial": 0.01},
        burst_events={"behavioral": 0, "spatial": 1},
    ),
    SyntheticScenario(
        name="照明変化 (Lighting Change Incident)",
        behavioral_fp_rate=0.62, spatial_fp_rate=0.20,
        temporal_fp_rate=0.08, interaction_fp_rate=0.10,
        sigma_drift={"behavioral": 0.45, "spatial": 0.08},
        burst_events={"behavioral": 5, "spatial": 2},
    ),
    SyntheticScenario(
        name="不審行動群発 (Suspicious Activity Burst)",
        behavioral_fp_rate=0.15, spatial_fp_rate=0.65,
        temporal_fp_rate=0.55, interaction_fp_rate=0.60,
        sigma_drift={"behavioral": 0.05, "spatial": 0.35, "temporal": 0.20, "interaction": 0.30},
        burst_events={"behavioral": 2, "spatial": 7, "temporal": 6, "interaction": 8},
    ),
    SyntheticScenario(
        name="システム正常化後 (Post-Reset Recovery)",
        behavioral_fp_rate=0.08, spatial_fp_rate=0.12,
        temporal_fp_rate=0.10, interaction_fp_rate=0.06,
        sigma_drift={"behavioral": 0.03, "spatial": 0.02},
        burst_events={"behavioral": 1},
    ),
]


# ── Pipeline components (real imports) ───────────────────────────────────────

def _load_components(verbose: bool):
    """Import and instantiate all closed-loop pipeline components."""
    import sys, os
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    if verbose:
        print(DIM("  Importing EarlyWarningEngine …"))
    from sopilot.perception.early_warning import EarlyWarningEngine
    ew_engine = EarlyWarningEngine()

    if verbose:
        print(DIM("  Importing EarlyWarningResponder …"))
    from sopilot.perception.early_warning_responder import EarlyWarningResponder
    ew_responder = EarlyWarningResponder(risk_threshold=0.6, cooldown_seconds=0.0)

    if verbose:
        print(DIM("  Importing SigmaTuner …"))
    from sopilot.perception.sigma_tuner import SigmaTuner
    sigma_tuner = SigmaTuner()

    return ew_engine, ew_responder, sigma_tuner


# ── Simulation helpers ────────────────────────────────────────────────────────

def _simulate_scenario(
    scenario: SyntheticScenario,
    ew_engine,
    ew_responder,
    sigma_tuner,
    num_frames: int,
    verbose: bool,
) -> dict[str, Any]:
    """Feed synthetic frame data through all phases and return results."""

    fp_rates = {
        "behavioral":  scenario.behavioral_fp_rate,
        "spatial":     scenario.spatial_fp_rate,
        "temporal":    scenario.temporal_fp_rate,
        "interaction": scenario.interaction_fp_rate,
    }

    # ── Phase 13/14: simulate SigmaTuner feedback ─────────────────────────
    sigma_changes: list[dict] = []
    for det, drift_rate in scenario.sigma_drift.items():
        if drift_rate > 0.05:
            old_s = 2.0
            new_s = max(0.5, old_s - drift_rate * 0.5)
            sigma_tuner._detector_sigmas[det] = new_s
            sigma_changes.append({"detector": det, "old": old_s, "new": new_s})

    # ── Phase 15: feed EarlyWarning observations ───────────────────────────
    # Sigma drift: inject converged EMA velocity directly (σ/min).
    # In production the EMA builds up over many live frames; here we shortcut
    # to a stable converged value to keep the demo deterministic.
    now = time.time()
    for det, drift_rate in scenario.sigma_drift.items():
        if drift_rate <= 0:
            continue
        with ew_engine._lock:
            ew_engine._drift_velocity[det] = drift_rate   # σ/min (converged EMA)
            ew_engine._drift_last_ts[det] = now

    # Anomaly burst observations — scatter within burst window
    for det, count in scenario.burst_events.items():
        for _ in range(count):
            ts = now - random.uniform(0, ew_engine.BURST_WINDOW_SECONDS * 0.9)
            ew_engine.observe_anomaly(det, timestamp=ts)

    # Build tuner_stats in the format EarlyWarningEngine expects:
    # {"pair_stats": [{"detector": ..., "total": N, "denied": N}, ...]}
    pair_stats = [
        {"detector": det, "total": 200, "denied": int(fp_rates[det] * 200)}
        for det in DETECTORS
    ]
    tuner_stats = {"pair_stats": pair_stats}
    ew_state = ew_engine.get_state(tuner_stats=tuner_stats)

    # ── Phase 16: autonomous response ─────────────────────────────────────
    actions = ew_responder.evaluate(
        ew_state,
        sigma_tuner=sigma_tuner,
        review_queue=None,
    )

    return {
        "ew_state":     ew_state,
        "sigma_changes": sigma_changes,
        "actions":       actions,
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def _print_phase15(ew_state: dict, verbose: bool) -> None:
    print()
    print(BOLD("  Phase 15 — 予兆検知リスクスコア"))
    print(f"  Overall: {_bar(ew_state['overall_risk'])}  "
          f"{ew_state['overall_risk']:.3f}  {_risk_badge(ew_state['overall_level'])}")
    if verbose:
        for det, info in ew_state.get("detectors", {}).items():
            score = info.get("risk_score", 0.0)
            level = info.get("risk_level", "LOW")
            sdv   = info.get("sigma_drift_velocity", 0.0)
            fpr   = info.get("fp_rate", 0.0)
            abr   = info.get("anomaly_burst_rate", 0.0)
            print(f"    {det:>12}: {_bar(score, 14)} {score:.3f} {_risk_badge(level)}"
                  f"  σdrift={sdv:.2f}  FP={fpr:.0%}  burst={abr:.1f}/min")


def _print_phase16(actions, verbose: bool) -> None:
    print()
    if not actions:
        print(BOLD("  Phase 16 — 自律対応") + "  " + GREEN("✓ 対応不要 (リスク閾値以下)"))
        return

    print(BOLD(f"  Phase 16 — 自律対応  ({len(actions)} detector(s) triggered)"))
    for act in actions:
        d = act.to_dict()
        print()
        print(f"    {RED('●')} {BOLD(d['detector'])}  score={d['risk_score']:.3f}  "
              f"{_risk_badge(d['risk_level'])}")
        print(f"    {DIM('説明:')} {d['explanation_ja']}")
        print(f"    {DIM('推奨アクション:')}")
        for rec in d["recommendations"]:
            print(f"      › {rec}")


def _print_sigma_changes(changes: list[dict]) -> None:
    if not changes:
        return
    print()
    print(BOLD("  Phase 13 — SigmaTuner 自動調整"))
    for ch in changes:
        arrow = "↓" if ch["new"] < ch["old"] else "↑"
        color = YELLOW if ch["new"] < ch["old"] else GREEN
        print(f"    {ch['detector']:>12}: σ {ch['old']:.2f} {color(arrow)} {ch['new']:.2f}")


# ── Main entry point ──────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║   SOPilot  Closed-Loop Perception Demo  v3.0.0                       ║
║   検出 → 異常 → 学習 → 予兆検知 → 自律対応                         ║
║   Phase 13 SigmaTuner / Phase 15 EarlyWarning / Phase 16 Responder  ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    global USE_COLOR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds",   type=int,  default=1, help="Number of full scenario runs (default 1)")
    parser.add_argument("--frames",   type=int,  default=30, help="Synthetic frames per scenario (unused, for parity)")
    parser.add_argument("--verbose",  action="store_true", help="Show per-detector breakdown")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour output")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    print(BOLD(BANNER))

    # Load components once
    print(BOLD("Loading pipeline components …"))
    t0 = time.perf_counter()
    ew_engine, ew_responder, sigma_tuner = _load_components(args.verbose)
    print(f"  {GREEN('✓')} Components ready in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    total_triggered = 0

    for run in range(args.rounds):
        if args.rounds > 1:
            print(BOLD(f"\n{'═'*70}"))
            print(BOLD(f"  Round {run+1}/{args.rounds}"))

        for idx, scenario in enumerate(SCENARIOS, 1):
            print(BOLD(f"\n{'─'*70}"))
            print(BOLD(f"  Scenario {idx}/{len(SCENARIOS)}: {scenario.name}"))
            print(BOLD(f"{'─'*70}"))

            # Reset per-scenario to show fresh state
            ew_engine.reset()
            ew_responder.reset()

            result = _simulate_scenario(
                scenario, ew_engine, ew_responder, sigma_tuner,
                num_frames=args.frames, verbose=args.verbose,
            )

            _print_sigma_changes(result["sigma_changes"])
            _print_phase15(result["ew_state"], verbose=args.verbose)
            _print_phase16(result["actions"], verbose=args.verbose)

            total_triggered += len(result["actions"])

            # Small pause for readability
            time.sleep(0.05)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(BOLD(f"\n{'═'*70}"))
    print(BOLD("  Demo Summary"))
    print(f"  Scenarios run      : {len(SCENARIOS) * args.rounds}")
    print(f"  Total responses    : {total_triggered}")
    print(f"  Closed-loop phases : Phase 13 (SigmaTuner) → Phase 15 (予兆検知) → Phase 16 (自律対応)")
    print()
    print(GREEN("  ✓ Closed-loop demo complete."))
    print(BOLD(f"{'═'*70}\n"))


if __name__ == "__main__":
    main()

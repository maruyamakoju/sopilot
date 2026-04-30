"""Agentic OODA Loop Demo — Phases 59→62 end-to-end.

Tells the killer-narrative story of the Perception Engine in 60 seconds:

    OBSERVE  →  ORIENT  →  DECIDE  →  ACT  →  VERIFY  →  REMEMBER  →  ADAPT

Each round simulates a scene's worth of violation events, lets the agent
think, schedules a closed-loop verification, fast-forwards time, evaluates
the outcome, stores the episode, and lets the meta-learner adjust the
agent's own decision thresholds. The same situation re-encountered later
is recognised via cosine recall on a 6-dim fingerprint.

Run::

    python scripts/demo_agentic_loop.py
    python scripts/demo_agentic_loop.py --rounds 6 --no-color
"""
from __future__ import annotations

import argparse
import time

from sopilot.perception.agent_loop import AgentEvent, AgentLoop
from sopilot.perception.agent_episode_store import AgentEpisodeStore
from sopilot.perception.closed_loop_verifier import ClosedLoopVerifier


# ── Terminal colours ──────────────────────────────────────────────────────

USE_COLOR = True


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


CYAN = lambda s: _c("36", s)
YELLOW = lambda s: _c("33", s)
GREEN = lambda s: _c("32", s)
RED = lambda s: _c("31", s)
DIM = lambda s: _c("2", s)
BOLD = lambda s: _c("1", s)


# ── Scenario synthesis ────────────────────────────────────────────────────


def _violations_for_round(round_idx: int, total_rounds: int) -> list[AgentEvent]:
    """Synthesise a violation pattern that escalates then resolves.

    Round 1: 1 warning  (low risk)        — agent: MONITOR
    Round 2: 2 warnings + 1 critical      — agent: ESCALATE
    Round 3: 4 warnings (post-action drop) — verifier: SUCCESS
    Later rounds: vary so meta-learner has data to act on.
    """
    rng_seed = round_idx
    now = time.time()

    if round_idx == 0:
        return [
            AgentEvent(event_type="violation", camera_id="cam-A",
                       severity="warning", violation_rule="ヘルメット未着用",
                       timestamp=now),
        ]
    if round_idx == 1:
        return [
            AgentEvent(event_type="violation", camera_id="cam-A",
                       severity="warning", violation_rule="ヘルメット未着用",
                       timestamp=now),
            AgentEvent(event_type="violation", camera_id="cam-A",
                       severity="warning", violation_rule="ヘルメット未着用",
                       timestamp=now),
            AgentEvent(event_type="violation", camera_id="cam-A",
                       severity="critical", violation_rule="重機接近",
                       timestamp=now),
        ]
    if round_idx == 2:
        return [
            AgentEvent(event_type="violation", camera_id="cam-A",
                       severity="warning", violation_rule="ヘルメット未着用",
                       timestamp=now),
        ]
    return [
        AgentEvent(event_type="violation", camera_id="cam-A",
                   severity="warning" if rng_seed % 2 else "critical",
                   violation_rule="ヘルメット未着用",
                   timestamp=now),
    ] * (1 + rng_seed % 3)


# ── Verification stub ─────────────────────────────────────────────────────


class _FrameStub:
    """Stand-in for a per-frame result. Only ``violations`` is consulted."""

    def __init__(self, n: int) -> None:
        self.violations = [None] * n


# ── Demo runner ───────────────────────────────────────────────────────────


def run(rounds: int, verify_after_s: float) -> None:
    print(BOLD(CYAN("\n─── Agentic OODA Loop Demo (Phase 59 → 62) ───\n")))

    episode_store = AgentEpisodeStore()
    verifier = ClosedLoopVerifier(
        verify_after_s=verify_after_s,
        success_threshold=0.5,
        meta_learning_enabled=True,
        original_escalation=3,
        original_plan=5,
        episode_store=episode_store,
    )
    agent = AgentLoop(
        think_interval_s=0.0,           # demo: always "due" to think
        critical_trigger=True,
        use_rag_orient=False,           # offline demo
        use_llava_orient=False,
        # Demo thresholds — tuned so a single demo run exercises every action:
        escalation_threshold=1,         # any critical → ESCALATE (default: 3)
        plan_threshold=3,               # 3+ violations under high risk → PLAN (default: 5)
        verifier=verifier,
        episode_store=episode_store,
    )

    for round_idx in range(rounds):
        print(BOLD(f"━━━━━ Round {round_idx + 1} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))

        # OBSERVE
        events = _violations_for_round(round_idx, rounds)
        for ev in events:
            agent.inject_event(ev)
        sev_summary = ", ".join(
            f"{ev.severity}:{ev.violation_rule}" for ev in events
        )
        print(f"  {CYAN('OBSERVE')}  {len(events)} event(s)  →  {DIM(sev_summary)}")

        # ORIENT + DECIDE + ACT + LEARN
        decision = agent.think()
        if decision is None:
            print(f"  {YELLOW('SKIP')}    no decision")
            continue

        action_color = {
            "monitor":  GREEN,
            "escalate": RED,
            "plan":     YELLOW,
            "adapt":    CYAN,
            "query":    DIM,
            "respond":  RED,
        }.get(decision.action.value, DIM)

        print(f"  {CYAN('ORIENT')}   risk={decision.situation.get('risk_level')}  "
              f"dominant={decision.situation.get('dominant_violation_type')}")
        print(f"  {CYAN('DECIDE')}   {action_color(decision.action.value.upper())}  "
              f"({decision.latency_ms:.1f} ms)")
        print(f"           {DIM(decision.reasoning_ja)}")

        ep_id = decision.params.get("episode_id")
        if ep_id:
            print(f"  {CYAN('REMEMBER')} episode_id={ep_id}  outcome=pending")

        # Recalled episodes (Phase 62) — visible from round 2 onwards
        recalled = episode_store.recall(
            risk_level=decision.situation.get("risk_level", "low"),
            severity_counts=decision.situation.get("severity_counts", {}),
            camera_violations={"cam-A": 1},
            dominant_violation_type=decision.situation.get("dominant_violation_type", ""),
            top_k=2,
        )
        if recalled and round_idx > 0:
            sims = ", ".join(
                f"{r.get('decision_action', '?')}({r.get('similarity', 0):.2f})"
                for r in recalled
            )
            print(f"  {CYAN('RECALL')}   {len(recalled)} similar past episode(s): {DIM(sims)}")

        # VERIFY (fast-forward simulation: frame later shows fewer violations)
        if decision.action.value != "monitor":
            time.sleep(min(verify_after_s, 0.05) + 0.01)  # bounded sleep
            # Simulate that the action worked: violations halved
            current_violations = max(0, len(events) // 2)
            completed = verifier.check_pending(_FrameStub(current_violations), camera_id="cam-A")
            for record in completed:
                outcome_color = {
                    "success":  GREEN,
                    "partial":  YELLOW,
                    "failure":  RED,
                    "unknown":  DIM,
                }[record.outcome.value]
                print(f"  {CYAN('VERIFY')}   T+{record.latency_s:.0f}s  "
                      f"{outcome_color(record.outcome.value.upper())}  "
                      f"reduction={record.reduction_ratio:+.0%}")

        print()

    # Drain any pending verifications so the summary is fully populated.
    if verifier.get_stats()["pending_tasks"] > 0:
        time.sleep(0.05)
        verifier.check_pending(_FrameStub(0), camera_id="cam-A")

    # ── Final report ──────────────────────────────────────────────────────
    print(BOLD(CYAN("─── Summary ───")))
    print(f"  decisions:        {agent._think_count}")
    print(f"  action mix:       {dict(agent._action_counts)}")
    print(f"  episodes stored:  {episode_store.get_stats().get('total_episodes', 0)}")
    v_stats = verifier.get_stats()
    print(f"  verifications:    {v_stats['completed_records']} completed, "
          f"{v_stats['pending_tasks']} pending")
    print(f"  outcomes:         {v_stats['outcome_counts']}")
    meta = v_stats["meta"]
    if meta.get("update_counts"):
        print(f"  meta-learner:     success_rates={meta['success_rates']}")
        print(f"                    suggested escalation_threshold="
              f"{meta['suggested_escalation_threshold']} (was 3)")
        print(f"                    suggested plan_threshold="
              f"{meta['suggested_plan_threshold']} (was 5)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of OBSERVE→…→VERIFY cycles to run")
    parser.add_argument("--verify-after-s", type=float, default=0.0,
                        help="Seconds before verifier checks (0 = immediate, demo mode)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    global USE_COLOR
    USE_COLOR = not args.no_color

    run(rounds=args.rounds, verify_after_s=args.verify_after_s)


if __name__ == "__main__":
    main()

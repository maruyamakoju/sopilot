# SOPilot PoC Capture Guide (1-page)

## Target

- Fixed `task_id`: define one task only for PoC.
- Suggested initial volume: `20 Gold` + `50 Trainee`.

## Camera and scene rules

- Fix camera position (no handheld).
- Keep workbench and operated parts fully in frame.
- Prefer angle that avoids faces and name tags.
- Keep lighting stable (avoid strong backlight).
- Keep background clutter minimal.

## Recording rules

- One video = one task execution.
- Start before first SOP step and stop after last step.
- Keep clip length consistent (avoid very short fragments).
- Keep site metadata (`site_id`, `camera_id`, timestamp).

## Gold data rules

- Gold must contain only verified correct procedures.
- Exclude uncertain or mixed-quality Gold samples.
- Capture 2-3 operators if possible to avoid overfitting one style.

## Trainee data rules

- Include both good and bad examples intentionally.
- Ensure bad examples include:
  - missing step (critical)
  - wrong method (quality)
  - over-time / inefficient flow (efficiency)

## Labeling policy for PoC review

- Critical deviation: immediate NG.
- Quality deviation: correction needed.
- Efficiency deviation: coaching level.

## Completion checklist

- [ ] `task_id` fixed and agreed by trainer/QA.
- [ ] Pass/retrain thresholds agreed.
- [ ] At least 20 Gold videos uploaded.
- [ ] At least 50 Trainee videos uploaded.
- [ ] Deviation examples visible in UI side-by-side review.


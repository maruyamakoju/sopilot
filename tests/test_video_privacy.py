from __future__ import annotations

import numpy as np

from sopilot.video import apply_privacy_mask, parse_mask_rects


def test_parse_mask_rects_handles_invalid_entries() -> None:
    rects = parse_mask_rects("0:0:1:0.2;bad;0.6:0.6:0.5:0.9;0.1:0.2:0.4:0.8")
    assert rects == [(0.0, 0.0, 1.0, 0.2), (0.1, 0.2, 0.4, 0.8)]


def test_apply_privacy_mask_black_mode_masks_region() -> None:
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    out = apply_privacy_mask(
        frame,
        rects=[(0.25, 0.25, 0.75, 0.75)],
        mode="black",
        face_blur=False,
        face_cascade=None,
    )
    assert int(np.sum(out[50, 50])) == 0
    assert int(np.sum(out[10, 10])) == int(np.sum(frame[10, 10]))


def test_apply_privacy_mask_blur_mode_changes_region() -> None:
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[:, :30] = (255, 0, 0)
    frame[:, 30:] = (0, 255, 0)
    out = apply_privacy_mask(
        frame,
        rects=[(0.2, 0.2, 0.8, 0.8)],
        mode="blur",
        face_blur=False,
        face_cascade=None,
    )
    # Center region should differ after blur while untouched corner remains equal.
    assert not np.array_equal(out[30, 30], frame[30, 30])
    assert np.array_equal(out[2, 2], frame[2, 2])

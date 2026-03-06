"""Tests for ReIDGallery and ReIDExtractor — Phase 28B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sopilot.vigil.reid_extractor import ReIDExtractor, _color_histogram
from sopilot.vigil.reid_gallery import ReIDGallery, _cosine_sim


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass
class _BBox:
    x1: float = 0.1
    y1: float = 0.1
    x2: float = 0.5
    y2: float = 0.9


@dataclass
class _Det:
    label: str = "person"
    confidence: float = 0.9
    bbox: _BBox = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = _BBox()


def _rand_emb(dim=512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _zeros_emb(dim=512) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _make_frame(h=64, w=64) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ── _cosine_sim ───────────────────────────────────────────────────────────────


def test_cosine_sim_identical():
    v = _rand_emb()
    assert abs(_cosine_sim(v, v) - 1.0) < 1e-5


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert _cosine_sim(a, b) == 0.0


def test_cosine_sim_zero_vector():
    a = np.zeros(512, dtype=np.float32)
    b = _rand_emb()
    assert _cosine_sim(a, b) == 0.0


def test_cosine_sim_clipped_to_zero():
    a = _rand_emb()
    b = -a
    assert _cosine_sim(a, b) == 0.0  # clipped to [0, 1]


# ── ReIDExtractor ─────────────────────────────────────────────────────────────


def test_extractor_returns_512_dim():
    ext = ReIDExtractor()
    frame = _make_frame()
    emb = ext.extract(frame, _BBox())
    assert emb.shape == (512,)
    assert emb.dtype == np.float32


def test_extractor_normalised():
    ext = ReIDExtractor()
    frame = _make_frame()
    emb = ext.extract(frame, _BBox())
    # L1-normalized histogram (sums to ~1.0) or L2-normalised
    assert 0.0 <= emb.sum() <= 512.0  # loose bound


def test_extractor_invalid_bbox_returns_zeros():
    ext = ReIDExtractor()
    frame = _make_frame()
    bad_bbox = _BBox(x1=0.9, y1=0.9, x2=0.1, y2=0.1)  # inverted
    emb = ext.extract(frame, bad_bbox)
    assert np.all(emb == 0.0)


def test_extractor_similarity_same_crop():
    ext = ReIDExtractor()
    frame = _make_frame()
    e1 = ext.extract(frame, _BBox())
    e2 = ext.extract(frame, _BBox())
    sim = ext.similarity(e1, e2)
    assert sim > 0.99


def test_extractor_similarity_different_crops():
    ext = ReIDExtractor()
    f1 = np.zeros((64, 64, 3), dtype=np.uint8)    # black
    f2 = np.full((64, 64, 3), 255, dtype=np.uint8)  # white
    e1 = ext.extract(f1, _BBox())
    e2 = ext.extract(f2, _BBox())
    sim = ext.similarity(e1, e2)
    # Different colours → lower similarity
    assert sim < 0.99


def test_extractor_with_external_embedder():
    from unittest.mock import MagicMock

    fake_emb = _rand_emb(128)
    embedder = MagicMock()
    embedder.embed.return_value = fake_emb
    ext = ReIDExtractor(embedder=embedder)
    frame = _make_frame()
    emb = ext.extract(frame, _BBox())
    embedder.embed.assert_called_once()
    assert emb.shape == (128,)


def test_extractor_external_embedder_fallback_on_error():
    from unittest.mock import MagicMock

    embedder = MagicMock()
    embedder.embed.side_effect = RuntimeError("embedder failed")
    ext = ReIDExtractor(embedder=embedder)
    frame = _make_frame()
    emb = ext.extract(frame, _BBox())
    assert emb.shape == (512,)  # fallback histogram


# ── ReIDGallery ───────────────────────────────────────────────────────────────


def test_gallery_empty():
    g = ReIDGallery()
    assert g.total_persons == 0
    assert g.get_persons() == []


def test_gallery_update_adds_person():
    g = ReIDGallery()
    g.update("cam1", "C1", [_Det()], [_rand_emb()])
    assert g.total_persons == 1


def test_gallery_update_non_person_ignored():
    g = ReIDGallery()
    det = _Det(label="helmet")
    g.update("cam1", "C1", [det], [_rand_emb()])
    assert g.total_persons == 0


def test_gallery_update_zero_emb_ignored():
    g = ReIDGallery()
    g.update("cam1", "C1", [_Det()], [_zeros_emb()])
    assert g.total_persons == 0


def test_gallery_same_camera_same_person():
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    g.update("cam1", "C1", [_Det()], [emb])  # same emb → match
    assert g.total_persons == 1


def test_gallery_distinct_embeddings_create_distinct_persons():
    g = ReIDGallery(match_threshold=0.99)
    for i in range(3):
        emb = np.zeros(512, dtype=np.float32)
        emb[i] = 1.0
        g.update("cam1", "C1", [_Det()], [emb])
    assert g.total_persons == 3


def test_gallery_cross_camera_event_fires():
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    events = g.update("cam2", "C2", [_Det()], [emb])
    assert len(events) == 1
    ev = events[0]
    assert ev.from_session_id == "cam1"
    assert ev.to_session_id == "cam2"


def test_gallery_no_cross_camera_same_session():
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    events = g.update("cam1", "C1", [_Det()], [emb])  # same camera
    assert events == []


def test_gallery_get_person():
    g = ReIDGallery()
    g.update("cam1", "C1", [_Det()], [_rand_emb()])
    pid = g.get_persons()[0]["person_id"]
    person = g.get_person(pid)
    assert person is not None
    assert person["person_id"] == pid


def test_gallery_get_person_not_found():
    g = ReIDGallery()
    assert g.get_person("no-such") is None


def test_gallery_person_has_camera_history():
    g = ReIDGallery()
    g.update("cam1", "Camera 1", [_Det()], [_rand_emb()])
    person = g.get_persons()[0]
    assert len(person["camera_history"]) == 1
    assert person["camera_history"][0]["session_id"] == "cam1"


def test_gallery_clear_session():
    g = ReIDGallery()
    g.update("cam1", "C1", [_Det()], [_rand_emb()])
    g.update("cam2", "C2", [_Det()], [_rand_emb()])
    removed = g.clear_session("cam1")
    assert removed >= 0
    remaining = g.get_persons(session_id="cam1")
    assert remaining == []


def test_gallery_recent_events():
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    g.update("cam2", "C2", [_Det()], [emb])
    events = g.get_recent_events()
    assert len(events) >= 1
    assert "person_id" in events[0]
    assert events[0]["type"] == "cross_camera_match"


def test_gallery_subscribe_events():
    import asyncio

    g = ReIDGallery(match_threshold=0.5)
    q = g.subscribe_events()
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    g.update("cam2", "C2", [_Det()], [emb])
    # Should have one event in queue
    ev = q.get_nowait()
    assert ev["type"] == "cross_camera_match"
    g.unsubscribe_events(q)


def test_gallery_unsubscribe_noop():
    import asyncio

    g = ReIDGallery()
    q = asyncio.Queue()
    g.unsubscribe_events(q)  # should not raise


def test_gallery_multi_camera_person_track():
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    g.update("cam2", "C2", [_Det()], [emb])
    g.update("cam3", "C3", [_Det()], [emb])
    person = g.get_persons()[0]
    assert person["camera_count"] == 3
    assert set(person["cameras"]) == {"cam1", "cam2", "cam3"}

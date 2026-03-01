"""Shared helpers for route handlers."""

from __future__ import annotations

from fastapi import Request

from sopilot.services.score_queue import ScoreJobQueue
from sopilot.services.sopilot_service import SOPilotService


def get_service(request: Request) -> SOPilotService:
    """Extract the SOPilotService from application state."""
    svc: SOPilotService = request.app.state.sopilot_service
    return svc


def get_queue(request: Request) -> ScoreJobQueue:
    """Extract the ScoreJobQueue from application state."""
    q: ScoreJobQueue = request.app.state.score_queue
    return q

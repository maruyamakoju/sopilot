"""VigilPilot — Surveillance camera violation detection module.

Analyzes video footage (files or camera streams) at 1 fps using a VLM
and user-defined text rules, returning structured violation events.
"""

__all__ = ["build_vigil_router"]

from sopilot.vigil.router import build_vigil_router

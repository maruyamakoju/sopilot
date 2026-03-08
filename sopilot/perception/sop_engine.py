"""SOP Engine — Fine-grained action analysis via mask geometry and state machines.

Phase 51: Analyzes precise interactions between entities (e.g. hand-tool-part)
using SAM 2 masks and maintains a state machine for SOP step tracking.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sopilot.perception.types import (
    SceneEntity, 
    SceneGraph, 
    SOPStep, 
    SOPProgress,
    Violation,
    ViolationSeverity
)

logger = logging.getLogger(__name__)

@dataclass
class MaskGeometry:
    centroid: tuple[float, float]
    area: float
    orientation: float  # radians
    aspect_ratio: float
    major_axis_len: float

def analyze_mask_geometry(mask: np.ndarray) -> MaskGeometry | None:
    """Compute geometric properties of a binary mask."""
    import cv2
    coords = np.argwhere(mask > 0)
    if len(coords) < 10:
        return None
        
    y, x = coords[:, 0], coords[:, 1]
    centroid = (float(np.mean(x)), float(np.mean(y)))
    area = float(len(coords))
    
    # Moments for orientation
    moments = cv2.moments(mask.astype(np.uint8))
    if moments['m00'] == 0:
        return None
        
    # Orientation via PCA/Moments
    mu20 = moments['mu20'] / moments['m00']
    mu02 = moments['mu02'] / moments['m00']
    mu11 = moments['mu11'] / moments['m00']
    
    # Angle of major axis
    angle = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    
    # Eigenvalues for axis lengths
    common = math.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
    lam1 = (mu20 + mu02 + common) / 2
    lam2 = (mu20 + mu02 - common) / 2
    
    major_axis = 4 * math.sqrt(lam1) if lam1 > 0 else 0
    minor_axis = 4 * math.sqrt(lam2) if lam2 > 0 else 0
    aspect = major_axis / minor_axis if minor_axis > 0 else 0
    
    return MaskGeometry(
        centroid=centroid,
        area=area,
        orientation=angle,
        aspect_ratio=aspect,
        major_axis_len=major_axis
    )

class SOPEngine:
    """State machine for tracking Standard Operating Procedures."""
    
    def __init__(self, steps: list[SOPStep]) -> None:
        self._steps = sorted(steps, key=lambda s: s.order)
        self._current_step_idx = 0
        self._completed_steps: list[str] = []
        self._history: list[dict] = []
        self._last_state_change = time.time()
        
    def update(self, scene_graph: SceneGraph) -> tuple[SOPProgress, list[Violation]]:
        """Evaluate current scene against the SOP state machine."""
        if self._current_step_idx >= len(self._steps):
            return SOPProgress("completed", self._completed_steps), []
            
        current_step = self._steps[self._current_step_idx]
        violations = []
        
        # 1. Check for required entities
        present_labels = {e.label.lower() for e in scene_graph.entities}
        missing = [req for req in current_step.required_entities if req.lower() not in present_labels]
        
        # 2. Geometry analysis if entities are present
        interaction_detected = False
        if not missing:
            interaction_detected = self._check_interaction(scene_graph, current_step)
            
        if interaction_detected:
            # Advance state
            logger.info("SOP Step Completed: %s", current_step.name)
            self._completed_steps.append(current_step.step_id)
            self._current_step_idx += 1
            self._last_state_change = time.time()
            
        # 3. Detect sequence violations (e.g. skipping steps)
        # (Simplified logic: if future step entities are interacting but current is not)
        if self._current_step_idx < len(self._steps) - 1:
            next_step = self._steps[self._current_step_idx + 1]
            if self._check_interaction(scene_graph, next_step):
                violations.append(Violation(
                    rule_index=-1,
                    rule="SOP Sequence Error",
                    description_ja=f"手順を飛ばしています: '{current_step.name}' が未完了です。",
                    severity=ViolationSeverity.CRITICAL,
                    confidence=0.9,
                    source="sop_engine"
                ))

        prog = SOPProgress(
            current_step_id=self._steps[self._current_step_idx].step_id if self._current_step_idx < len(self._steps) else "completed",
            completed_steps=list(self._completed_steps),
            is_violation=len(violations) > 0,
            action_suggestion=f"次の手順: {current_step.name}" if not interaction_detected else "手順完了、次へ進んでください"
        )
        
        return prog, violations

    def _check_interaction(self, scene_graph: SceneGraph, step: SOPStep) -> bool:
        """Analyze if the physical conditions for a step are met using mask geometry."""
        # Find relevant entities
        targets = []
        for req in step.required_entities:
            matches = [e for e in scene_graph.entities if req.lower() in e.label.lower()]
            if not matches: return False
            targets.append(matches[0]) # Take first match
            
        # Trivially satisfied if no required entities
        if not targets:
            return True

        # If we have masks, check intersection
        if all(getattr(t, "mask", None) is not None for t in targets):
            # Mask-based intersection check
            mask_a = targets[0].mask
            for i in range(1, len(targets)):
                mask_b = targets[i].mask
                if mask_a.shape != mask_b.shape:
                    continue
                # Check for overlap (contact)
                intersection = (mask_a & mask_b).sum()
                if intersection < 50: # Threshold for contact
                    return False
                    
            # Check for movement/orientation change (Level 3 depth)
            # This is a stub for specific action logic like "twisting"
            if step.target_state == "tightened":
                # Check rotation of tool mask
                pass
                
            return True # Basic contact met for step completion in this stub
            
        return False

    def reset(self) -> None:
        self._current_step_idx = 0
        self._completed_steps.clear()

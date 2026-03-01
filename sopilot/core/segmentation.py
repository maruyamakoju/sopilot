import numpy as np

from sopilot.core.math_utils import cosine_distance


def detect_step_boundaries(
    embeddings: np.ndarray,
    min_gap: int = 2,
    z_threshold: float = 1.0,
) -> list[int]:
    """
    Return clip indices where a new step likely starts.
    If a boundary is returned as `k`, transition is between clip `k-1` and `k`.
    """
    if embeddings.ndim != 2 or len(embeddings) < 2:
        return []

    deltas = np.array(
        [cosine_distance(embeddings[i], embeddings[i - 1]) for i in range(1, len(embeddings))]
    )
    mean = float(np.mean(deltas))
    std = float(np.std(deltas))
    threshold = mean + z_threshold * std

    boundaries: list[int] = []
    last = -min_gap
    for offset, delta in enumerate(deltas, start=1):
        if delta < threshold:
            continue
        if offset - last < min_gap:
            continue
        boundaries.append(offset)
        last = offset

    return boundaries


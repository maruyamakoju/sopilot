"""Shared numerical constants for neural modules.

Centralizes epsilon / infinity / clamp values to avoid magic numbers
and make numerical stability policy easy to audit and tune.
"""

# DP table fill value (large but finite to avoid true inf arithmetic).
INF: float = 1e9

# Minimum smoothing parameter for Soft-DTW gamma.
# Below this, logsumexp becomes numerically equivalent to hard min.
GAMMA_MIN: float = 1e-4

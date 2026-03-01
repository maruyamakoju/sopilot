import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(v))
    if denom <= 1e-12:
        return np.array(v, copy=True)
    return np.asarray(v / denom)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    value = float(1.0 - np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return value



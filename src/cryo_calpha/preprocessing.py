from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_density(
    volume: ArrayLike,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-6,
) -> NDArray[np.float32]:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("density volume must be three-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("density volume contains NaN or infinity")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("normalization percentiles are invalid")
    lower, upper = np.percentile(array, [lower_percentile, upper_percentile])
    clipped = np.clip(array, lower, upper)
    standard_deviation = float(clipped.std())
    if standard_deviation < eps:
        raise ValueError("density volume has near-zero variance")
    return ((clipped - clipped.mean()) / standard_deviation).astype(np.float32)

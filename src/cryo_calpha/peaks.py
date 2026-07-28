from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from .spatial import index_zyx_to_world_xyz


@dataclass(frozen=True)
class PeakSet:
    coordinates_xyz: NDArray[np.float64]
    scores: NDArray[np.float32]
    indices_zyx: NDArray[np.float64]


def _subvoxel_centroid(
    probability: np.ndarray,
    index_zyx: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    lower = np.maximum(0, index_zyx - radius)
    upper = np.minimum(np.asarray(probability.shape), index_zyx + radius + 1)
    slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    patch = probability[slices]
    total = float(patch.sum())
    if total <= 0:
        return index_zyx.astype(np.float64)
    axes = [np.arange(lower[axis], upper[axis], dtype=np.float64) for axis in range(3)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return np.array(
        [
            float((zz * patch).sum() / total),
            float((yy * patch).sum() / total),
            float((xx * patch).sum() / total),
        ]
    )


def extract_calpha_peaks(
    probability_zyx: ArrayLike,
    *,
    origin_xyz: ArrayLike,
    voxel_size_xyz: ArrayLike,
    threshold: float,
    nms_radius_angstrom: float,
) -> PeakSet:
    probability = np.asarray(probability_zyx, dtype=np.float32)
    if probability.ndim != 3 or not np.isfinite(probability).all():
        raise ValueError("probability_zyx must be a finite three-dimensional array")
    if not 0 <= threshold <= 1 or nms_radius_angstrom <= 0:
        raise ValueError("threshold must be in [0, 1] and NMS radius must be positive")
    local_maxima = probability == maximum_filter(probability, size=3, mode="constant")
    candidate_indices = np.argwhere(local_maxima & (probability >= threshold))
    if len(candidate_indices) == 0:
        return PeakSet(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
        )
    scores = probability[tuple(candidate_indices.T)]
    refined = np.stack([_subvoxel_centroid(probability, index) for index in candidate_indices])
    coordinates = index_zyx_to_world_xyz(refined, origin_xyz, voxel_size_xyz)
    tree = cKDTree(coordinates)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(scores), dtype=bool)
    kept: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        kept.append(int(index))
        neighbors = tree.query_ball_point(coordinates[index], nms_radius_angstrom)
        suppressed[np.asarray(neighbors, dtype=int)] = True
    kept_array = np.asarray(kept, dtype=int)
    return PeakSet(coordinates[kept_array], scores[kept_array], refined[kept_array])

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spatial import world_xyz_to_index_zyx, world_xyz_to_voxel_xyz


def create_calpha_targets(
    shape_zyx: tuple[int, int, int],
    ca_coords_xyz: ArrayLike,
    origin_xyz: ArrayLike,
    voxel_size_xyz: ArrayLike,
    *,
    sigma_angstrom: float,
    truncate_sigma: float = 3.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if len(shape_zyx) != 3 or any(dimension <= 0 for dimension in shape_zyx):
        raise ValueError("shape_zyx must contain three positive dimensions")
    if sigma_angstrom <= 0 or truncate_sigma <= 0:
        raise ValueError("sigma_angstrom and truncate_sigma must be positive")

    coords = np.asarray(ca_coords_xyz, dtype=np.float64)
    if coords.size == 0:
        return np.zeros(shape_zyx, np.float32), np.zeros(shape_zyx, np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("ca_coords_xyz must have shape [N, 3]")

    hard = np.zeros(shape_zyx, dtype=np.float32)
    heatmap = np.zeros(shape_zyx, dtype=np.float32)
    nearest_zyx = world_xyz_to_index_zyx(coords, origin_xyz, voxel_size_xyz)
    valid = np.all((nearest_zyx >= 0) & (nearest_zyx < np.asarray(shape_zyx)), axis=1)
    for index in nearest_zyx[valid]:
        hard[tuple(index)] = 1.0

    centers_zyx = world_xyz_to_voxel_xyz(coords, origin_xyz, voxel_size_xyz)[:, ::-1]
    voxel_size_zyx = np.asarray(voxel_size_xyz, dtype=np.float64)[::-1]
    radius_zyx = np.ceil(truncate_sigma * sigma_angstrom / voxel_size_zyx).astype(int)
    shape = np.asarray(shape_zyx)

    for center in centers_zyx:
        lower = np.maximum(0, np.floor(center - radius_zyx).astype(int))
        upper = np.minimum(shape, np.ceil(center + radius_zyx).astype(int) + 1)
        if np.any(lower >= upper):
            continue
        axes = [np.arange(lower[axis], upper[axis]) for axis in range(3)]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        delta_zyx = np.stack((zz - center[0], yy - center[1], xx - center[2]), axis=-1)
        distance_squared = np.sum((delta_zyx * voxel_size_zyx) ** 2, axis=-1)
        gaussian = np.exp(-distance_squared / (2.0 * sigma_angstrom**2)).astype(np.float32)
        slices = tuple(slice(lower[axis], upper[axis]) for axis in range(3))
        heatmap[slices] = np.maximum(heatmap[slices], gaussian)

    return hard, heatmap

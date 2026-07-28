from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _points_xyz(points: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _vector_xyz(value: ArrayLike, name: str, *, positive: bool = False) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three XYZ values")
    if not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must contain {qualifier} values")
    return array


def world_xyz_to_voxel_xyz(
    points_xyz: ArrayLike,
    origin_xyz: ArrayLike,
    voxel_size_xyz: ArrayLike,
) -> NDArray[np.float64]:
    points = _points_xyz(points_xyz, "points_xyz")
    origin = _vector_xyz(origin_xyz, "origin_xyz")
    voxel_size = _vector_xyz(voxel_size_xyz, "voxel_size_xyz", positive=True)
    return (points - origin) / voxel_size


def world_xyz_to_index_zyx(
    points_xyz: ArrayLike,
    origin_xyz: ArrayLike,
    voxel_size_xyz: ArrayLike,
    *,
    rounding: Literal["nearest", "floor"] = "nearest",
) -> NDArray[np.int64]:
    voxel_xyz = world_xyz_to_voxel_xyz(points_xyz, origin_xyz, voxel_size_xyz)
    if rounding == "nearest":
        discrete_xyz = np.rint(voxel_xyz)
    elif rounding == "floor":
        discrete_xyz = np.floor(voxel_xyz)
    else:
        raise ValueError("rounding must be 'nearest' or 'floor'")
    return discrete_xyz[:, ::-1].astype(np.int64)


def index_zyx_to_world_xyz(
    indices_zyx: ArrayLike,
    origin_xyz: ArrayLike,
    voxel_size_xyz: ArrayLike,
) -> NDArray[np.float64]:
    indices = _points_xyz(indices_zyx, "indices_zyx")
    origin = _vector_xyz(origin_xyz, "origin_xyz")
    voxel_size = _vector_xyz(voxel_size_xyz, "voxel_size_xyz", positive=True)
    return origin + indices[:, ::-1] * voxel_size


def crop_origin_xyz(
    global_origin_xyz: ArrayLike,
    start_zyx: ArrayLike,
    voxel_size_xyz: ArrayLike,
) -> NDArray[np.float64]:
    origin = _vector_xyz(global_origin_xyz, "global_origin_xyz")
    start = _vector_xyz(start_zyx, "start_zyx")
    voxel_size = _vector_xyz(voxel_size_xyz, "voxel_size_xyz", positive=True)
    return origin + start[::-1] * voxel_size

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DensityMap:
    grid_zyx: NDArray[np.float32]
    voxel_size_xyz: NDArray[np.float64]
    global_origin_xyz: NDArray[np.float64]


def _validate_map(grid: np.ndarray, voxel_size: np.ndarray, origin: np.ndarray) -> DensityMap:
    if grid.ndim != 3 or not np.isfinite(grid).all():
        raise ValueError("grid must be a finite three-dimensional array")
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0) or not np.isfinite(voxel_size).all():
        raise ValueError("voxel_size must contain three positive finite XYZ values")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("global_origin must contain three finite XYZ values")
    return DensityMap(
        grid.astype(np.float32), voxel_size.astype(np.float64), origin.astype(np.float64)
    )


def load_npz_map(path: str | Path) -> DensityMap:
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"grid", "voxel_size", "global_origin"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ map is missing keys: {sorted(missing)}")
        return _validate_map(
            np.asarray(data["grid"]),
            np.asarray(data["voxel_size"], dtype=np.float64),
            np.asarray(data["global_origin"], dtype=np.float64),
        )


def load_mrc_map(path: str | Path) -> DensityMap:
    try:
        import mrcfile
    except ImportError as error:
        raise RuntimeError("mrcfile is required to read MRC/MAP files") from error
    with mrcfile.open(Path(path), permissive=False) as mrc:
        raw = np.asarray(mrc.data)
        world_axes_for_raw = [int(mrc.header.maps), int(mrc.header.mapr), int(mrc.header.mapc)]
        if sorted(world_axes_for_raw) != [1, 2, 3]:
            raise ValueError(f"unsupported MRC axis mapping: {world_axes_for_raw}")
        permutation = tuple(world_axes_for_raw.index(axis) for axis in (3, 2, 1))
        grid_zyx = np.transpose(raw, permutation)
        voxel_size = np.array(
            [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float64
        )
        origin = np.array(
            [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float64
        )
        if np.allclose(origin, 0):
            starts_raw = np.array(
                [mrc.header.nzstart, mrc.header.nystart, mrc.header.nxstart], dtype=np.float64
            )
            starts_xyz = np.empty(3, dtype=np.float64)
            for raw_axis, world_axis in enumerate(world_axes_for_raw):
                starts_xyz[world_axis - 1] = starts_raw[raw_axis]
            origin = starts_xyz * voxel_size
    return _validate_map(grid_zyx, voxel_size, origin)


def load_density_map(path: str | Path) -> DensityMap:
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return load_npz_map(path)
    if suffix in {".mrc", ".map"}:
        return load_mrc_map(path)
    raise ValueError(f"unsupported map format: {suffix}")


def parse_calpha_coordinates(path: str | Path) -> NDArray[np.float64]:
    try:
        from Bio.PDB import MMCIFParser
    except ImportError as error:
        raise RuntimeError("Biopython is required to parse CIF/mmCIF structures") from error
    structure = MMCIFParser(QUIET=True).get_structure("structure", str(path))
    first_model = next(structure.get_models(), None)
    if first_model is None:
        raise ValueError(f"structure contains no models: {path}")
    coordinates = [
        atom.coord
        for residue in first_model.get_residues()
        if residue.has_id("CA")
        for atom in [residue["CA"]]
    ]
    if not coordinates:
        raise ValueError(f"structure contains no C-alpha atoms: {path}")
    return np.asarray(coordinates, dtype=np.float64)

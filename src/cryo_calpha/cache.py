from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import AppConfig
from .io import load_npz_map, parse_calpha_coordinates
from .manifest import SampleRecord, manifest_sha256, read_manifest
from .preprocessing import normalize_density
from .sliding_window import pad_to_window, sliding_window_starts
from .spatial import crop_origin_xyz
from .targets import create_calpha_targets

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CropRecord:
    cache_file: str
    source_sample_id: str
    split: str
    start_zyx: tuple[int, int, int]
    positive: bool


def _config_hash(config: AppConfig) -> str:
    payload = {
        "crop_size_zyx": config.data.crop_size_zyx,
        "overlap_fraction": config.data.overlap_fraction,
        "heatmap_sigma_angstrom": config.data.heatmap_sigma_angstrom,
        "heatmap_truncate_sigma": config.data.heatmap_truncate_sigma,
        "empty_to_positive_ratio": config.data.empty_to_positive_ratio,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_windows(
    starts: list[tuple[int, int, int]],
    hard_label: np.ndarray,
    crop_size: tuple[int, int, int],
    empty_to_positive_ratio: float,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, int, int], bool]]:
    positives: list[tuple[tuple[int, int, int], bool]] = []
    negatives: list[tuple[tuple[int, int, int], bool]] = []
    dz, dy, dx = crop_size
    for start in starts:
        z, y, x = start
        positive = bool(hard_label[z : z + dz, y : y + dy, x : x + dx].any())
        (positives if positive else negatives).append((start, positive))
    if not positives:
        return []
    negative_count = min(len(negatives), int(round(len(positives) * empty_to_positive_ratio)))
    if negative_count:
        indices = rng.choice(len(negatives), size=negative_count, replace=False)
        positives.extend(negatives[int(index)] for index in sorted(indices))
    return sorted(positives, key=lambda item: item[0])


def _cache_sample(
    record: SampleRecord,
    destination: Path,
    config: AppConfig,
    rng: np.random.Generator,
) -> list[CropRecord]:
    density = load_npz_map(record.map_path)
    volume = normalize_density(density.grid_zyx)
    coords = parse_calpha_coordinates(record.structure_path)
    hard, heatmap = create_calpha_targets(
        volume.shape,
        coords,
        density.global_origin_xyz,
        density.voxel_size_xyz,
        sigma_angstrom=config.data.heatmap_sigma_angstrom,
        truncate_sigma=config.data.heatmap_truncate_sigma,
    )
    volume, _ = pad_to_window(volume, config.data.crop_size_zyx)
    hard, _ = pad_to_window(hard, config.data.crop_size_zyx)
    heatmap, _ = pad_to_window(heatmap, config.data.crop_size_zyx)
    starts = sliding_window_starts(
        volume.shape, config.data.crop_size_zyx, config.data.overlap_fraction
    )
    selected = _select_windows(
        starts,
        hard,
        config.data.crop_size_zyx,
        config.data.empty_to_positive_ratio,
        rng,
    )
    dz, dy, dx = config.data.crop_size_zyx
    rows: list[CropRecord] = []
    split_dir = destination / record.split
    for index, (start, positive) in enumerate(selected):
        z, y, x = start
        filename = f"{record.sample_id}-{index:06d}.npz"
        relative = Path(record.split) / filename
        crop_origin = crop_origin_xyz(density.global_origin_xyz, start, density.voxel_size_xyz)
        _atomic_save_npz(
            split_dir / filename,
            volume=volume[z : z + dz, y : y + dy, x : x + dx],
            hard_label=hard[z : z + dz, y : y + dy, x : x + dx],
            heatmap=heatmap[z : z + dz, y : y + dy, x : x + dx],
            voxel_size_xyz=density.voxel_size_xyz,
            global_origin_xyz=density.global_origin_xyz,
            crop_origin_xyz=crop_origin,
            start_zyx=np.asarray(start, dtype=np.int64),
            source_sample_id=np.asarray(record.sample_id),
            split=np.asarray(record.split),
        )
        rows.append(
            CropRecord(
                str(relative).replace("\\", "/"), record.sample_id, record.split, start, positive
            )
        )
    return rows


def build_cache(config: AppConfig, *, force: bool = False) -> Path:
    records = read_manifest(config.data.manifest_path)
    destination = config.data.cache_dir.resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"cache already exists: {destination}; pass force=True to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        rng = np.random.default_rng(config.data.seed)
        crop_records: list[CropRecord] = []
        for record in records:
            crop_records.extend(_cache_sample(record, temporary, config, rng))
        if not crop_records:
            raise ValueError("cache generation produced no positive training crops")
        manifest_text = "".join(
            json.dumps(asdict(record), sort_keys=True) + "\n" for record in crop_records
        )
        (temporary / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_manifest_sha256": manifest_sha256(records),
            "config_sha256": _config_hash(config),
            "crop_count": len(crop_records),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "COMPLETE").write_text("ok\n", encoding="ascii")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def read_cache_manifest(cache_dir: str | Path) -> list[CropRecord]:
    root = Path(cache_dir)
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"cache is incomplete: {root}")
    rows: list[CropRecord] = []
    for line_number, line in enumerate((root / "manifest.jsonl").read_text().splitlines(), 1):
        try:
            payload = json.loads(line)
            payload["start_zyx"] = tuple(payload["start_zyx"])
            rows.append(CropRecord(**payload))
        except (TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid cache manifest line {line_number}: {error}") from error
    return rows

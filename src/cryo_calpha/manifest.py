from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    map_path: str
    structure_path: str
    split: Split


def _discover_pair(directory: Path) -> tuple[Path, Path] | None:
    maps = sorted(directory.glob("*.npz"))
    structures = sorted([*directory.glob("*.cif"), *directory.glob("*.mmcif")])
    if not maps and not structures:
        return None
    if len(maps) != 1 or len(structures) != 1:
        raise ValueError(
            f"{directory} must contain exactly one NPZ map and one CIF/mmCIF structure; "
            f"found {len(maps)} map(s) and {len(structures)} structure(s)"
        )
    return maps[0].resolve(), structures[0].resolve()


def discover_samples(root_dir: str | Path) -> list[tuple[str, Path, Path]]:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    samples: list[tuple[str, Path, Path]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        pair = _discover_pair(directory)
        if pair is not None:
            samples.append((directory.name, pair[0], pair[1]))
    if not samples:
        raise ValueError(f"no valid sample directories found under {root}")
    return samples


def split_samples(
    samples: Iterable[tuple[str, Path, Path]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[SampleRecord]:
    samples = sorted(samples, key=lambda item: item[0])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    train_end = int(round(len(samples) * train_ratio))
    val_end = train_end + int(round(len(samples) * val_ratio))
    train_end = min(train_end, len(samples))
    val_end = min(val_end, len(samples))
    split_by_index: dict[int, Split] = {}
    for position, index in enumerate(order):
        split_by_index[int(index)] = (
            "train" if position < train_end else "val" if position < val_end else "test"
        )
    return [
        SampleRecord(sample_id, str(map_path), str(structure_path), split_by_index[index])
        for index, (sample_id, map_path, structure_path) in enumerate(samples)
    ]


def write_manifest(records: Iterable[SampleRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda record: record.sample_id)
    sample_ids = [record.sample_id for record in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique")
    text = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in rows)
    destination.write_text(text, encoding="utf-8")
    return destination


def read_manifest(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(SampleRecord(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError("manifest is empty")
    return records


def manifest_sha256(records: Iterable[SampleRecord]) -> str:
    payload = "".join(
        json.dumps(asdict(record), sort_keys=True) + "\n"
        for record in sorted(records, key=lambda item: item.sample_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

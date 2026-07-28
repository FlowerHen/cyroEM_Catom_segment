from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig, load_config


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False))


def _build_model(config: AppConfig) -> object:
    from .models import UNet3D

    return UNet3D(
        base_channels=config.model.base_channels,
        depth=config.model.depth,
        dropout=config.model.dropout,
    )


def command_validate_config(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _print_json(asdict(config))


def command_build_manifest(args: argparse.Namespace) -> None:
    from .manifest import discover_samples, split_samples, write_manifest

    config = load_config(args.config)
    ratios = config.data.split_ratios
    records = split_samples(
        discover_samples(config.data.root_dir),
        train_ratio=ratios.train,
        val_ratio=ratios.val,
        test_ratio=ratios.test,
        seed=config.data.seed,
    )
    path = write_manifest(records, config.data.manifest_path)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }
    _print_json({"manifest": str(path), "samples": len(records), "split_counts": counts})


def command_build_cache(args: argparse.Namespace) -> None:
    from .cache import build_cache, read_cache_manifest

    config = load_config(args.config)
    path = build_cache(config, force=args.force)
    rows = read_cache_manifest(path)
    _print_json({"cache": str(path), "crops": len(rows)})


def command_train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from .datasets import CachedCropDataset
    from .trainer import Trainer, seed_everything

    config = load_config(args.config)
    seed_everything(config.training.seed)
    train_dataset = CachedCropDataset(
        config.data.cache_dir,
        split="train",
        augmentation=config.augmentation,
        seed=config.training.seed,
    )
    val_dataset = CachedCropDataset(config.data.cache_dir, split="val")
    generator = torch.Generator().manual_seed(config.training.seed)
    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    trainer = Trainer(
        _build_model(config), config, train_loader=train_loader, val_loader=val_loader
    )
    _print_json(trainer.fit(resume=args.resume))


def command_infer(args: argparse.Namespace) -> None:
    import torch

    from .io import load_density_map
    from .peaks import extract_calpha_peaks
    from .preprocessing import normalize_density
    from .sliding_window import predict_volume_logits
    from .trainer import resolve_device

    config = load_config(args.config)
    device = resolve_device(config.training.device)
    model = _build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    density = load_density_map(args.map)
    volume = normalize_density(density.grid_zyx)
    logits = predict_volume_logits(
        model,
        volume,
        window_size_zyx=config.data.crop_size_zyx,
        overlap_fraction=config.inference.overlap_fraction,
        batch_size=config.inference.window_batch_size,
        device=str(device),
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    peaks = extract_calpha_peaks(
        probability,
        origin_xyz=density.global_origin_xyz,
        voxel_size_xyz=density.voxel_size_xyz,
        threshold=config.inference.probability_threshold,
        nms_radius_angstrom=config.inference.nms_radius_angstrom,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        probability_zyx=probability.astype(np.float32),
        coordinates_xyz=peaks.coordinates_xyz,
        scores=peaks.scores,
        voxel_size_xyz=density.voxel_size_xyz,
        global_origin_xyz=density.global_origin_xyz,
    )
    _print_json({"output": str(output.resolve()), "prediction_count": len(peaks.scores)})


def command_evaluate(args: argparse.Namespace) -> None:
    from .io import parse_calpha_coordinates
    from .matching import match_points_one_to_one

    config = load_config(args.config)
    with np.load(args.predictions, allow_pickle=False) as predictions:
        coordinates = np.asarray(predictions["coordinates_xyz"], dtype=np.float64)
    truth = parse_calpha_coordinates(args.structure)
    result = match_points_one_to_one(
        coordinates, truth, radius_angstrom=config.inference.match_radius_angstrom
    )
    _print_json(result.metrics())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryo-calpha")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate and resolve configuration")
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=command_validate_config)

    manifest = subparsers.add_parser("build-manifest", help="discover and split source samples")
    manifest.add_argument("--config", required=True, type=Path)
    manifest.set_defaults(handler=command_build_manifest)

    cache = subparsers.add_parser("build-cache", help="build a versioned crop cache")
    cache.add_argument("--config", required=True, type=Path)
    cache.add_argument("--force", action="store_true", help="replace an existing cache")
    cache.set_defaults(handler=command_build_cache)

    train = subparsers.add_parser("train", help="train the 3D heatmap model")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--resume", type=Path)
    train.set_defaults(handler=command_train)

    infer = subparsers.add_parser("infer", help="run weighted sliding-window inference")
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument("--map", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.set_defaults(handler=command_infer)

    evaluate = subparsers.add_parser("evaluate", help="one-to-one point-set evaluation")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--structure", required=True, type=Path)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

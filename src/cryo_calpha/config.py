from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _strict_kwargs(cls: type[T], values: Mapping[str, Any], section: str) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration key(s) in {section}: {names}")
    return dict(values)


def _triple_int(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        result = tuple(int(item) for item in value)
    else:
        raise ValueError(f"{name} must be an integer or a sequence of three integers")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _probability(value: float, name: str, *, upper_inclusive: bool = True) -> float:
    value = float(value)
    upper_ok = value <= 1.0 if upper_inclusive else value < 1.0
    if value < 0.0 or not upper_ok:
        bound = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be in {bound}")
    return value


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("split ratios must sum to 1")


@dataclass(frozen=True)
class DataConfig:
    root_dir: Path
    manifest_path: Path
    cache_dir: Path
    crop_size_zyx: tuple[int, int, int] = (64, 64, 64)
    overlap_fraction: float = 0.25
    heatmap_sigma_angstrom: float = 1.5
    heatmap_truncate_sigma: float = 3.0
    empty_to_positive_ratio: float = 0.5
    split_ratios: SplitRatios = SplitRatios()
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop_size_zyx", _triple_int(self.crop_size_zyx, "crop_size_zyx"))
        object.__setattr__(
            self,
            "overlap_fraction",
            _probability(self.overlap_fraction, "overlap_fraction", upper_inclusive=False),
        )
        if self.heatmap_sigma_angstrom <= 0:
            raise ValueError("heatmap_sigma_angstrom must be positive")
        if self.heatmap_truncate_sigma <= 0:
            raise ValueError("heatmap_truncate_sigma must be positive")
        if self.empty_to_positive_ratio < 0:
            raise ValueError("empty_to_positive_ratio cannot be negative")


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    rotation_probability: float = 0.5
    gaussian_noise_probability: float = 0.3
    gaussian_noise_std: float = 0.05
    gaussian_blur_probability: float = 0.2
    gaussian_blur_sigma_voxels: tuple[float, float] = (0.5, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "rotation_probability",
            "gaussian_noise_probability",
            "gaussian_blur_probability",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std cannot be negative")
        sigma = tuple(float(item) for item in self.gaussian_blur_sigma_voxels)
        if len(sigma) != 2 or sigma[0] < 0 or sigma[1] < sigma[0]:
            raise ValueError("gaussian_blur_sigma_voxels must be [min, max] with 0 <= min <= max")
        object.__setattr__(self, "gaussian_blur_sigma_voxels", sigma)


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 16
    depth: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.depth < 2:
            raise ValueError("depth must be at least 2")
        object.__setattr__(self, "dropout", _probability(self.dropout, "dropout"))


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    amp: bool = True
    pos_weight: float = 50.0
    bce_weight: float = 1.0
    focal_weight: float = 0.5
    dice_weight: float = 0.5
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    early_stopping_patience: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "accumulation_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip_norm <= 0 or self.pos_weight <= 0:
            raise ValueError("gradient_clip_norm and pos_weight must be positive")
        for name in ("bce_weight", "focal_weight", "dice_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.bce_weight + self.focal_weight + self.dice_weight <= 0:
            raise ValueError("at least one loss weight must be positive")
        object.__setattr__(self, "focal_alpha", _probability(self.focal_alpha, "focal_alpha"))
        if self.focal_gamma < 0 or self.early_stopping_patience < 0:
            raise ValueError("focal_gamma and early_stopping_patience cannot be negative")


@dataclass(frozen=True)
class InferenceConfig:
    overlap_fraction: float = 0.5
    window_batch_size: int = 2
    probability_threshold: float = 0.35
    nms_radius_angstrom: float = 2.0
    match_radius_angstrom: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overlap_fraction",
            _probability(self.overlap_fraction, "overlap_fraction", upper_inclusive=False),
        )
        object.__setattr__(
            self,
            "probability_threshold",
            _probability(self.probability_threshold, "probability_threshold"),
        )
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        if self.nms_radius_angstrom <= 0 or self.match_radius_angstrom <= 0:
            raise ValueError("NMS and match radii must be positive")


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    required = {"data", "augmentation", "model", "training", "inference"}
    unknown = set(raw) - required
    missing = required - set(raw)
    if unknown or missing:
        raise ValueError(
            f"configuration sections mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    base_dir = config_path.parent
    data_values = _strict_kwargs(DataConfig, raw["data"], "data")
    split_values = data_values.get("split_ratios", {})
    data_values["split_ratios"] = SplitRatios(
        **_strict_kwargs(SplitRatios, split_values, "data.split_ratios")
    )
    for key in ("root_dir", "manifest_path", "cache_dir"):
        data_values[key] = _resolve_path(data_values[key], base_dir)

    training_values = _strict_kwargs(TrainingConfig, raw["training"], "training")
    training_values["output_dir"] = _resolve_path(training_values["output_dir"], base_dir)

    return AppConfig(
        data=DataConfig(**data_values),
        augmentation=AugmentationConfig(
            **_strict_kwargs(AugmentationConfig, raw["augmentation"], "augmentation")
        ),
        model=ModelConfig(**_strict_kwargs(ModelConfig, raw["model"], "model")),
        training=TrainingConfig(**training_values),
        inference=InferenceConfig(**_strict_kwargs(InferenceConfig, raw["inference"], "inference")),
    )

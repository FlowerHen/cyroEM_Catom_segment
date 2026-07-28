from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _triple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive values")
    return result


def sliding_window_starts(
    shape_zyx: Sequence[int],
    window_size_zyx: int | Sequence[int],
    overlap_fraction: float,
) -> list[tuple[int, int, int]]:
    shape = _triple(shape_zyx, "shape_zyx")
    window = _triple(window_size_zyx, "window_size_zyx")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    starts_per_axis: list[list[int]] = []
    for dimension, size in zip(shape, window, strict=True):
        if dimension <= size:
            starts_per_axis.append([0])
            continue
        stride = max(1, int(round(size * (1.0 - overlap_fraction))))
        starts = list(range(0, dimension - size + 1, stride))
        final_start = dimension - size
        if starts[-1] != final_start:
            starts.append(final_start)
        starts_per_axis.append(starts)
    return list(itertools.product(*starts_per_axis))


def pad_to_window(
    volume_zyx: ArrayLike,
    window_size_zyx: int | Sequence[int],
    *,
    value: float = 0.0,
) -> tuple[NDArray[np.float32], tuple[slice, slice, slice]]:
    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("volume_zyx must be three-dimensional")
    window = _triple(window_size_zyx, "window_size_zyx")
    padding = [
        (0, max(0, size - dimension)) for dimension, size in zip(volume.shape, window, strict=True)
    ]
    padded = np.pad(volume, padding, mode="constant", constant_values=value)
    original = tuple(slice(0, dimension) for dimension in volume.shape)
    return padded, original


def gaussian_importance_map(
    window_size_zyx: int | Sequence[int],
    *,
    sigma_scale: float = 0.125,
    minimum: float = 1e-3,
) -> NDArray[np.float32]:
    window = _triple(window_size_zyx, "window_size_zyx")
    if sigma_scale <= 0 or minimum <= 0:
        raise ValueError("sigma_scale and minimum must be positive")
    axes = []
    for size in window:
        coordinate = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
        sigma = max(size * sigma_scale, np.finfo(np.float64).eps)
        axes.append(np.exp(-0.5 * (coordinate / sigma) ** 2))
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weights /= weights.max()
    return np.maximum(weights, minimum).astype(np.float32)


def predict_volume_logits(
    model: object,
    volume_zyx: ArrayLike,
    *,
    window_size_zyx: int | Sequence[int],
    overlap_fraction: float,
    batch_size: int,
    device: str,
) -> NDArray[np.float32]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for sliding-window prediction") from error
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    window = _triple(window_size_zyx, "window_size_zyx")
    padded, original_slices = pad_to_window(volume_zyx, window)
    starts = sliding_window_starts(padded.shape, window, overlap_fraction)
    weights = gaussian_importance_map(window)
    weighted_sum = np.zeros(padded.shape, dtype=np.float64)
    weight_sum = np.zeros(padded.shape, dtype=np.float64)
    model.eval()

    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            crops = np.stack(
                [
                    padded[
                        z : z + window[0],
                        y : y + window[1],
                        x : x + window[2],
                    ]
                    for z, y, x in batch_starts
                ]
            )
            tensor = torch.from_numpy(crops[:, None]).to(device)
            logits = model(tensor)
            if logits.shape != tensor.shape:
                output_shape = tuple(logits.shape)
                input_shape = tuple(tensor.shape)
                raise ValueError(
                    f"model output shape {output_shape} does not match input shape {input_shape}"
                )
            predictions = logits[:, 0].float().cpu().numpy()
            for prediction, (z, y, x) in zip(predictions, batch_starts, strict=True):
                slices = (
                    slice(z, z + window[0]),
                    slice(y, y + window[1]),
                    slice(x, x + window[2]),
                )
                weighted_sum[slices] += prediction * weights
                weight_sum[slices] += weights

    if np.any(weight_sum <= 0):
        raise RuntimeError("sliding-window prediction did not cover the full volume")
    logits = (weighted_sum / weight_sum).astype(np.float32)
    return logits[original_slices]


def optimizer_steps_per_epoch(loader_length: int, accumulation_steps: int) -> int:
    if loader_length <= 0 or accumulation_steps <= 0:
        raise ValueError("loader_length and accumulation_steps must be positive")
    return math.ceil(loader_length / accumulation_steps)

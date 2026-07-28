from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache import read_cache_manifest
from .config import AugmentationConfig
from .transforms import random_cube_rotation

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as error:
    raise RuntimeError("PyTorch is required to use cryo_calpha.datasets") from error


class CachedCropDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str,
        augmentation: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.cache_dir = Path(cache_dir)
        self.records = [
            record for record in read_cache_manifest(cache_dir) if record.split == split
        ]
        if not self.records:
            raise ValueError(f"cache has no records for split {split}")
        self.augmentation = augmentation if split == "train" else None
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        path = self.cache_dir / record.cache_file
        with np.load(path, allow_pickle=False) as data:
            volume = np.asarray(data["volume"], dtype=np.float32)
            hard = np.asarray(data["hard_label"], dtype=np.float32)
            heatmap = np.asarray(data["heatmap"], dtype=np.float32)
            voxel_size = np.asarray(data["voxel_size_xyz"], dtype=np.float32)
            crop_origin = np.asarray(data["crop_origin_xyz"], dtype=np.float32)

        if self.augmentation and self.augmentation.enabled:
            if self.augmentation.rotation_probability > 0 and not np.allclose(
                voxel_size, voxel_size[0], rtol=1e-4, atol=1e-6
            ):
                raise ValueError(
                    "cube rotations require isotropic voxels; resample the source map first"
                )
            rng = np.random.default_rng(self.seed + index + int(torch.initial_seed() % (2**31)))
            if rng.random() < self.augmentation.rotation_probability:
                rotation = random_cube_rotation(rng)
                volume = rotation.apply(volume)
                hard = rotation.apply(hard)
                heatmap = rotation.apply(heatmap)
            if rng.random() < self.augmentation.gaussian_blur_probability:
                from scipy.ndimage import gaussian_filter

                sigma = rng.uniform(*self.augmentation.gaussian_blur_sigma_voxels)
                volume = gaussian_filter(volume, sigma=sigma).astype(np.float32)
            if rng.random() < self.augmentation.gaussian_noise_probability:
                noise = rng.normal(0.0, self.augmentation.gaussian_noise_std, volume.shape)
                volume = (volume + noise).astype(np.float32)

        return {
            "volume": torch.from_numpy(np.ascontiguousarray(volume[None])),
            "hard_label": torch.from_numpy(np.ascontiguousarray(hard[None])),
            "heatmap": torch.from_numpy(np.ascontiguousarray(heatmap[None])),
            "voxel_size_xyz": torch.from_numpy(voxel_size),
            "crop_origin_xyz": torch.from_numpy(crop_origin),
            "source_sample_id": record.source_sample_id,
        }

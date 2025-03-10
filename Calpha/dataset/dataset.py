import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import logging  # Added for logging errors in __getitem__
from .preprocess import CryoEMPreprocessor, VolumeAugmentor

def get_sliding_window_starts(volume_shape, crop_size, overlap):
    """
    Compute the sliding window start positions for each dimension.
    Args:
        volume_shape (tuple): Shape of the 3D volume (D1, D2, D3).
        crop_size (tuple): Crop size (C1, C2, C3).
        overlap (float): Overlap size (e.g., 2).
    Returns:
        List of tuples: Each tuple is (start_x, start_y, start_z).
    """
    dim_starts = []
    for dim, crop in zip(volume_shape, crop_size):
        stride = int(crop - overlap)
        if stride <= 0:
            stride = crop
        starts = list(range(0, dim - crop + 1, stride))
        if starts[-1] != dim - crop:
            starts.append(dim - crop)
        dim_starts.append(starts)
    candidate_windows = []
    for x in dim_starts[0]:
        for y in dim_starts[1]:
            for z in dim_starts[2]:
                candidate_windows.append((x, y, z))
    return candidate_windows

class CryoEMDataset(Dataset):
    """PyTorch dataset for Cryo-EM data using deterministic sliding window cropping"""
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.samples = self._load_samples()
        self.augmentor = VolumeAugmentor(config) if mode == 'train' else None
        self.preprocessor = CryoEMPreprocessor(config)
        # Ensure crop_size is a tuple
        self.crop_size = self.config['data']['crop_size']
        if isinstance(self.crop_size, int):
            self.crop_size = (self.crop_size, self.crop_size, self.crop_size)
        overlap = self.config['data'].get('overlap', 0.0)
        self.crop_windows = []  # Each element: (sample_index, (start_x, start_y, start_z))
        for sample_idx, sample in enumerate(self.samples):
            try:
                grid_data = self.preprocessor.load_npz(sample['npz'])
                ca_coords = self.preprocessor.parse_cif(sample['cif'])
                hard_label, _ = self.preprocessor.create_labels(grid_data, ca_coords)
            except Exception as e:
                logging.error(f"Error precomputing crop windows for sample {sample}: {e}")
                continue
            if any(dim < crop for dim, crop in zip(grid_data['grid'].shape, self.crop_size)):
                logging.warning(f"Skipping sample {sample} due to insufficient grid dimensions for crop size {self.crop_size}")
                continue
            candidate_windows = get_sliding_window_starts(grid_data['grid'].shape, self.crop_size, overlap)
            for start in candidate_windows:
                x, y, z = start
                cx, cy, cz = self.crop_size
                region = hard_label[x:x+cx, y:y+cy, z:z+cz]
                if np.sum(region) > 0:
                    self.crop_windows.append((sample_idx, start))
            # Optional: selection based on uniform standard deviation distribution can be added here.
        logging.info(f"Total valid crop windows: {len(self.crop_windows)}")

    def _load_samples(self):
        data_dir = self.config['data']['root_dir']
        crop_size = self.config['data']['crop_size']
        min_ca_atoms = self.config['data'].get('min_ca_atoms', 5)
        samples = []
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            npz_files = [f for f in os.listdir(entry_path) if f.endswith('.npz')]
            cif_files = [f for f in os.listdir(entry_path) if f.endswith('.cif')]
            if not npz_files or not cif_files:
                logging.warning(f"Skipping {entry}: missing NPZ/CIF files")
                continue
            npz_path = os.path.join(entry_path, npz_files[0])
            cif_path = os.path.join(entry_path, cif_files[0])
            try:
                grid_data = CryoEMPreprocessor(self.config).load_npz(npz_path)
                grid_shape = grid_data['grid'].shape
                if isinstance(crop_size, int):
                    crop_check = crop_size
                    if any(dim < crop_check for dim in grid_shape):
                        logging.warning(f"Skipping {entry}: grid {grid_shape} < crop {crop_check}")
                        continue
                else:
                    if any(dim < c for dim, c in zip(grid_shape, crop_size)):
                        logging.warning(f"Skipping {entry}: grid {grid_shape} < crop {crop_size}")
                        continue
                ca_coords = CryoEMPreprocessor(self.config).parse_cif(cif_path)
                if len(ca_coords) < min_ca_atoms:
                    logging.warning(f"Skipping {entry}: only {len(ca_coords)} CA atoms")
                    continue
                samples.append({'npz': npz_path, 'cif': cif_path})
            except Exception as e:
                logging.error(f"Skipping {entry} due to error: {str(e)}")
                continue
        return samples[:self.config['data']['max_samples']]

    def __len__(self):
        return len(self.crop_windows)

    def __getitem__(self, idx):
        try:
            # Use crop_windows mapping to retrieve the corresponding sample index and crop start
            sample_idx, start = self.crop_windows[idx]
            sample = self.samples[sample_idx]
            grid_data = self.preprocessor.load_npz(sample['npz'])
            ca_coords = self.preprocessor.parse_cif(sample['cif'])
            hard_label, soft_label = self.preprocessor.create_labels(grid_data, ca_coords)
            x, y, z = start
            cx, cy, cz = self.crop_size
            volume = grid_data['grid'][x:x+cx, y:y+cy, z:z+cz]
            cropped_hard = hard_label[x:x+cx, y:y+cy, z:z+cz]
            cropped_soft = soft_label[x:x+cx, y:y+cy, z:z+cz]
            # If in training mode, apply data augmentation
            if self.mode == 'train' and self.augmentor:
                volume, (cropped_hard, cropped_soft) = self.augmentor(volume, (cropped_hard, cropped_soft))
            return (
                torch.tensor(volume.copy(), dtype=torch.float32).unsqueeze(0),
                torch.tensor(cropped_hard, dtype=torch.float32).unsqueeze(0),
                torch.tensor(cropped_soft, dtype=torch.float32).unsqueeze(0),
                torch.tensor(grid_data['voxel_size'], dtype=torch.float32),
                torch.tensor(grid_data['global_origin'], dtype=torch.float32)
            )
        except Exception as e:
            # Log the error with index information and re-raise
            logging.error(f"Error in __getitem__ for index {idx}: {e}")
            raise e

def get_data_loaders(config):
    """Create train and validation data loaders. Use cached dataset if specified."""
    if 'cache_dir' in config['data'] and os.path.exists(config['data']['cache_dir']):
        from .dataset_cache import PrecachedCryoEMDataset
        full_dataset = PrecachedCryoEMDataset(config)
    else:
        full_dataset = CryoEMDataset(config, mode='train')
    
    train_size = int(config['data']['split_ratio']['train'] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader

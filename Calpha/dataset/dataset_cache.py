import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import VolumeAugmentor  

class PrecachedCryoEMDataset(Dataset):
    """
    Dataset that loads pre-cached cubes from a specified cache directory.
    Expected file format: NPZ files with keys 'volume', 'hard_label', 'soft_label',
    'voxel_size', and 'global_origin'
    """
    def __init__(self, config):
        self.cache_dir = config['data']['cache_dir']
        self.cached_files = sorted([os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith('.npz')])
        if not self.cached_files:
            raise RuntimeError(f"No cached NPZ files found in {self.cache_dir}")
        
        self.use_augmentation = config.get('augmentation', {}).get('enabled', False)
        if self.use_augmentation:
            self.augmentor = VolumeAugmentor(config)
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.cached_files)
    
    def __getitem__(self, idx):
        file_path = self.cached_files[idx]
        data = np.load(file_path)
        volume = data['volume']
        hard_label = data['hard_label']
        soft_label = data['soft_label']
        voxel_size = data['voxel_size']
        global_origin = data['global_origin']

        if self.augmentor is not None:
            volume, (hard_label, soft_label) = self.augmentor(volume, (hard_label, soft_label))
        
        return (
            torch.tensor(volume.copy(), dtype=torch.float32).unsqueeze(0),
            torch.tensor(hard_label.copy(), dtype=torch.float32).unsqueeze(0),
            torch.tensor(soft_label.copy(), dtype=torch.float32).unsqueeze(0),
            torch.tensor(voxel_size, dtype=torch.float32),
            torch.tensor(global_origin, dtype=torch.float32)
        )
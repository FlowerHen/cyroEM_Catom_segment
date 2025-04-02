import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from scipy.ndimage import zoom, gaussian_filter
from tqdm import tqdm
from .preprocess import CryoEMPreprocessor, VolumeAugmentor
from .dataset import get_sliding_window_starts
from . import data_cache_generator
from .cube_rotation import sample_rotation_params, apply_rotation


class BlockClassifierDataset(Dataset):
    """
    Dataset for block classification of 64³ blocks to determine C-alpha presence.
    This dataset loads blocks and their block labels indicating whether they contain C-alpha atoms.
    """
    def __init__(self, config, mode='train', cache_dir=None):
        self.config = config
        self.mode = mode
        self.preprocessor = CryoEMPreprocessor(config)
        self.augmentor = VolumeAugmentor(config)
        
        # Determine cache directory - use config's cache_dir if available
        self.cache_dir = config.get('data', {}).get('cache_dir', cache_dir)
        if self.cache_dir is None:
            self.cache_dir = config.get('block_cache_dir',
                                       os.path.join(os.path.dirname(config['block_classifier']['checkpoint_dir']),
                                                    'block_classifier_cache'))
        
        # Check if cache directory exists, if not create it and generate cache
        if not os.path.exists(self.cache_dir) or len(os.listdir(self.cache_dir)) == 0:
            logging.info(f"Cache directory {self.cache_dir} does not exist or is empty. Generating cache...")
            # For block classifier, we need to keep empty blocks
            if 'data' not in config:
                config['data'] = {}
            config['data']['keep_empty_block'] = True
            self.cache_dir = data_cache_generator.generate_data_cache(config)
        
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load sample paths from the cache directory"""
        samples = []
        
        # Get all cached files
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npz')]
        
        if not cache_files:
            logging.warning(f"No cached files found in {self.cache_dir}")
            return samples
        
        # Split into train/val based on config split ratio
        split_ratio = self.config['data'].get('split_ratio', {'train': 0.8, 'val': 0.2})
        
        # Group files by sample ID to ensure blocks from the same sample stay together
        sample_groups = {}
        for file_name in cache_files:
            # Extract sample ID from filename (format: EMD-XXXXX_cX_rX.npz)
            # We need to get the base sample ID without the crop/rotation/scale suffixes
            parts = file_name.split('_')
            # The sample ID is everything before the first underscore followed by 'c'
            # This ensures blocks from the same original sample stay together
            sample_id = '_'.join([p for p in parts if not (p.startswith('c') and p[1:].isdigit())])
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(file_name)
        
        # Shuffle sample IDs deterministically
        random_seed = self.config['block_classifier'].get('seed', 42)
        rng = np.random.RandomState(random_seed)
        sample_ids = sorted(sample_groups.keys())  # Sort for deterministic order
        rng.shuffle(sample_ids)
        
        # Split based on mode
        num_train = int(len(sample_ids) * split_ratio['train'])
        if self.mode == 'train':
            selected_ids = sample_ids[:num_train]
        else:  # val mode
            selected_ids = sample_ids[num_train:]
        
        # Create sample list
        for sample_id in selected_ids:
            for file_name in sample_groups[sample_id]:
                file_path = os.path.join(self.cache_dir, file_name)
                samples.append({'cache_file': file_path})
        
        # Balance dataset if needed (for training)
        if self.mode == 'train' and self.config['block_classifier'].get('balance_classes', True):
            # Count positive and negative samples
            pos_samples = []
            neg_samples = []
            
            for sample in samples:
                try:
                    cache_data = np.load(sample['cache_file'])
                    # Always use with_calpha
                    with_calpha = cache_data['with_calpha']
                    # Handle both 0-dim and 1-dim with_calpha
                    if with_calpha.ndim == 0:
                        block_label = float(with_calpha)
                    else:
                        block_label = with_calpha[0]
                    
                    if block_label > 0.5:
                        pos_samples.append(sample)
                    else:
                        neg_samples.append(sample)
                except Exception as e:
                    logging.error(f"Error loading sample {sample['cache_file']}: {e}")
            
            # Balance the dataset
            pos_count = len(pos_samples)
            neg_count = len(neg_samples)
            
            logging.info(f"Before balancing: {pos_count} positive, {neg_count} negative samples")
            
            # If we have no positive samples but have negative samples, use some negative samples
            if pos_count == 0 and neg_count > 0:
                max_samples = min(neg_count, 1000)  # Use up to 1000 negative samples
                rng.shuffle(neg_samples)
                balanced_samples = neg_samples[:max_samples]
            # If we have no negative samples but have positive samples, use some positive samples
            elif neg_count == 0 and pos_count > 0:
                max_samples = min(pos_count, 1000)  # Use up to 1000 positive samples
                rng.shuffle(pos_samples)
                balanced_samples = pos_samples[:max_samples]
            # Normal case: balance by undersampling the majority class
            else:
                min_count = min(pos_count, neg_count)
                if min_count == 0:
                    min_count = 1  # Avoid division by zero
                
                max_neg_ratio = self.config['block_classifier'].get('max_negative_ratio', 3.0)
                neg_count = min(neg_count, int(min_count * max_neg_ratio))
                
                # Shuffle and select samples
                rng.shuffle(pos_samples)
                rng.shuffle(neg_samples)
                balanced_samples = pos_samples[:min_count] + neg_samples[:neg_count]
                rng.shuffle(balanced_samples)
            
            logging.info(f"Balanced dataset: {min_count} positive, {neg_count} negative samples")
            samples = balanced_samples
        
        logging.info(f"Loaded {len(samples)} samples for {self.mode} set")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        try:
            # Load from cache file
            cache_data = np.load(sample['cache_file'])
            volume = cache_data['volume'].copy()
            with_calpha = cache_data['with_calpha'].copy()  # Load with_Calpha
            voxel_size = cache_data['voxel_size']

            
            # Apply augmentation for training mode on every epoch
            if self.mode == 'train':
                # 1. Apply random rotation augmentation
                if self.config['augmentation'].get('rotation_samples', 0) > 0:
                    # Sample one random rotation
                    rotations = sample_rotation_params(1)
                    rotation_param = rotations[0]
                    # Apply rotation to volume
                    volume = apply_rotation(volume, rotation_param)
                    volume = np.ascontiguousarray(volume)  # Ensure contiguous memory layout
                
                # 2. Apply resolution reduction if enabled
                if self.config['augmentation'].get('use_resolution_reduction', False):
                    if np.random.rand() < self.config['augmentation'].get('resolution_prob', 0.5):
                        sigma = np.random.uniform(
                            *self.config['augmentation'].get('resolution_sigma_range', [0.5, 1.5])
                        )
                        volume = gaussian_filter(volume, sigma=sigma)
                
                # 3. Apply random noise if enabled
                if np.random.rand() < self.config['augmentation'].get('noise_prob', 0.5):
                    noise_std = self.config['block_classifier'].get('noise_std', 0.05)
                    noise = np.random.normal(0, noise_std, volume.shape)
                    volume = np.clip(volume + noise, 0, 1).astype(np.float32)
            
            # Ensure arrays are contiguous before converting to tensors
            volume = np.ascontiguousarray(volume)
            
            # Convert to tensors
            volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

            # Use with_Calpha as the label
            label_tensor = torch.tensor(with_calpha, dtype=torch.float32)
            
            return volume_tensor, label_tensor
            
        except Exception as e:
            logging.error(f"Error processing sample {sample}: {str(e)}")
            logging.error(f"Sample details: {sample}")
            # Instead of returning a dummy sample, re-raise the exception
            # This will help identify and fix issues more easily
            raise RuntimeError(f"Failed to process sample {sample}: {str(e)}") from e

def get_block_classifier_data_loaders(config):
    """Create data loaders for block-based C-alpha classification"""
    # Determine cache directory - use config's cache_dir if available
    cache_dir = config.get('data', {}).get('cache_dir',
                          os.path.join(os.path.dirname(config['block_classifier']['checkpoint_dir']),
                                      'block_classifier_cache'))
    
    # Check if cache directory exists, if not create it and generate cache
    if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        logging.info(f"Cache directory {cache_dir} does not exist or is empty. Generating cache...")
        # For block classifier, we need to keep empty blocks
        if 'data' not in config:
            config['data'] = {}
        config['data']['keep_empty_block'] = True
        cache_dir = data_cache_generator.generate_data_cache(config)
    
    # Create datasets
    train_dataset = BlockClassifierDataset(config, mode='train', cache_dir=cache_dir)
    val_dataset = BlockClassifierDataset(config, mode='val', cache_dir=cache_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['block_classifier'].get('batch_size', 16),
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['block_classifier'].get('batch_size', 16),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader

# Alias for backward compatibility
get_block_data_loaders = get_block_classifier_data_loaders
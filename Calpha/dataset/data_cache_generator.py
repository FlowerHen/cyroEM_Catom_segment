import os
import numpy as np
import yaml
import logging
from tqdm import tqdm
# Use absolute imports when running as a package
from Calpha.dataset.preprocess import CryoEMPreprocessor, VolumeAugmentor
from Calpha.dataset.dataset import get_sliding_window_starts
from Calpha.dataset.cube_rotation import sample_rotation_params, apply_rotation

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_data_cache(config):
    data_dir = config['data']['root_dir']
    cache_dir = config['data'].get('cache_dir')
    if cache_dir is None:
        raise ValueError("cache_dir must be specified in config['data']")
    os.makedirs(cache_dir, exist_ok=True)
    
    crop_size = config['data']['crop_size']
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)
    overlap = config['data'].get('overlap', 0.0)
    min_ca_atoms = config['data'].get('min_ca_atoms', 5)
    max_samples = config['data'].get('max_samples', 1000)
    
    # Log global meta information before processing
    rotation_samples = config['augmentation'].get('rotation_samples', 0)
    logging.info(f"Global Config Meta: crop_size = {crop_size}, overlap = {overlap}, "
                 f"min_ca_atoms = {min_ca_atoms}, max_samples = {max_samples}, "
                 f"rotation_samples = {rotation_samples}")

    preprocessor = CryoEMPreprocessor(config)
    augmentor = VolumeAugmentor(config)
    
    # Load samples (similar to CryoEMDataset._load_samples)
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
            grid_data = preprocessor.load_npz(npz_path)
            grid_shape = grid_data['grid'].shape
            if any(dim < c for dim, c in zip(grid_shape, crop_size)):
                logging.warning(f"Skipping {entry}: grid {grid_shape} < crop {crop_size}")
                continue
            ca_coords = preprocessor.parse_cif(cif_path)
            if len(ca_coords) < min_ca_atoms:
                logging.warning(f"Skipping {entry}: only {len(ca_coords)} CA atoms")
                continue
            samples.append({'npz': npz_path, 'cif': cif_path})
        except Exception as e:
            logging.error(f"Skipping {entry} due to error: {str(e)}")
            continue
        if len(samples) >= max_samples:
            break
    
    logging.info("Starting data caching and preprocessing...")
    # Process each sample independently
    for sample in tqdm(samples, desc="Processing Samples"):
        npz_path = sample['npz']
        # Use the base name (without extension) as prefix
        file_prefix = os.path.splitext(os.path.basename(npz_path))[0]
        # Initialize a local crop counter for this sample
        local_crop_counter = 0
        
        try:
            grid_data = preprocessor.load_npz(sample['npz'])
            ca_coords = preprocessor.parse_cif(sample['cif'])
            hard_label, soft_label = preprocessor.create_labels(grid_data, ca_coords)
        except Exception as e:
            logging.error(f"Error in preprocessing sample {sample}: {e}")
            continue
        
        grid = grid_data['grid']
        if any(dim < c for dim, c in zip(grid.shape, crop_size)):
            logging.warning(f"Skipping sample {sample} due to insufficient grid dimensions")
            continue
        
        candidate_windows = get_sliding_window_starts(grid.shape, crop_size, overlap)
        for start in candidate_windows:
            x, y, z = start
            cx, cy, cz = crop_size
            region_hard = hard_label[x:x+cx, y:y+cy, z:z+cz]
            if np.sum(region_hard) > 0:
                volume_crop = grid[x:x+cx, y:y+cy, z:z+cz]
                hard_crop = hard_label[x:x+cx, y:y+cy, z:z+cz]
                soft_crop = soft_label[x:x+cx, y:y+cy, z:z+cz]
                
                # Determine rotation samples – if 0 then no rotation
                if rotation_samples > 0:
                    rotations = sample_rotation_params(rotation_samples)
                else:
                    rotations = [None]  # No rotation
                
                for r_idx, rot_param in enumerate(rotations):
                    vol_aug = volume_crop.copy()
                    hard_aug = hard_crop.copy()
                    soft_aug = soft_crop.copy()
                    if rot_param is not None:
                        vol_aug = apply_rotation(vol_aug, rot_param).copy()
                        hard_aug = apply_rotation(hard_aug, rot_param).copy()
                        soft_aug = apply_rotation(soft_aug, rot_param).copy()
                    # Apply random noise augmentation if needed
                    if np.random.rand() < augmentor.noise_prob:
                        noise = np.random.normal(0, augmentor.noise_std * augmentor.noise_multiplier, vol_aug.shape)
                        vol_aug = np.clip(vol_aug + noise, 0, 1).astype(np.float32)
                    
                    # Construct file name using the original NPZ file prefix and local counters
                    filename = f"{file_prefix}_c{local_crop_counter}_r{r_idx}.npz"
                    file_path = os.path.join(cache_dir, filename)
                    np.savez_compressed(file_path,
                                        volume=vol_aug,
                                        hard_label=hard_aug,
                                        soft_label=soft_aug,
                                        voxel_size=grid_data['voxel_size'],
                                        global_origin=grid_data['global_origin'])
                # Increment crop counter for this sample (local counter)
                local_crop_counter += 1
        logging.info(f"Processed sample {file_prefix}: generated {local_crop_counter} cropped cubes.")
    logging.info("Data caching completed.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data Cache Generator for Cryo-EM Dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO)
    generate_data_cache(config)

if __name__ == "__main__":
    main()
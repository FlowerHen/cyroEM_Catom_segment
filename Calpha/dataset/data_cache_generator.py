import os
import numpy as np
import yaml
import logging
from tqdm import tqdm
from scipy.ndimage import zoom
from Calpha.dataset.preprocess import CryoEMPreprocessor, VolumeAugmentor
from Calpha.dataset.dataset import get_sliding_window_starts
from Calpha.dataset.cube_rotation import sample_rotation_params, apply_rotation


def load_config(config_path):
    """
    Load the YAML configuration file from config_path.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_data_cache(config):
    """
    Generate a cache of processed sample crops from the raw Cryo-EM dataset.
    """
    data_dir = config['data']['root_dir']
    cache_dir = config['data'].get('cache_dir')
    if cache_dir is None:
        raise ValueError("cache_dir must be specified in config['data']")
    os.makedirs(cache_dir, exist_ok=True)
    
    logging.info(f"Using cache directory: {cache_dir}")

    crop_size = config['data']['crop_size']
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)
    overlap = config['data'].get('overlap', 0.0)
    min_ca_atoms = config['data'].get('min_ca_atoms', 5)
    max_samples = config['data'].get('max_samples', 1000)
    
    # Retrieve the max_empty_ratio from config. This controls the balance:
    # only allow saving an empty crop if: empty_count < non_empty_count * max_empty_ratio.
    max_empty_ratio = config['data'].get('max_empty_ratio', 1.0)

    # Augmentation parameters.
    rotation_samples = config['augmentation'].get('rotation_samples', 0)
    
    logging.info(
        f"Global Config Meta: crop_size={crop_size}, overlap={overlap}, min_ca_atoms={min_ca_atoms}, max_samples={max_samples}")
    logging.info(f"Augmentation Config: rotation_samples={rotation_samples}")
    logging.info(f"Empty-to-nonempty ratio config: max_empty_ratio={max_empty_ratio}")

    preprocessor = CryoEMPreprocessor(config)

    # Load samples from the data directory
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

    # Global counters for empty and non-empty saved crops (across all samples)
    empty_count = 0
    non_empty_count = 0
    max_empty_per_sample = 50  # Maximum number of empty blocks per sample

    logging.info("Starting data caching and preprocessing...")
    for sample in tqdm(samples, desc="Processing Samples"):
        npz_path = sample['npz']
        file_prefix = os.path.splitext(os.path.basename(npz_path))[0]
        local_crop_counter = 0
        local_empty_counter = 0

        try:
            grid_data = preprocessor.load_npz(sample['npz'])
            ca_coords = preprocessor.parse_cif(sample['cif'])

            # Scaling-related operations have been commented out.
            # Always use the original grid data without scaling.
            scaled_versions = [(
                grid_data['grid'],
                ca_coords,
                grid_data['voxel_size'],
                grid_data['global_origin']
            )]

            for version_idx, (grid, coords, voxel_size, origin) in enumerate(scaled_versions):
                if any(dim < c for dim, c in zip(grid.shape, crop_size)):
                    logging.warning(f"Skipping version {version_idx} of {file_prefix}: grid {grid.shape} < crop {crop_size}")
                    continue

                grid_data_version = {
                    'grid': grid,
                    'voxel_size': voxel_size,
                    'global_origin': origin
                }
                hard_label, soft_label = preprocessor.create_labels(grid_data_version, coords)

                candidate_windows = get_sliding_window_starts(grid.shape, crop_size, overlap)
                for start in candidate_windows:
                    x, y, z = start
                    cx, cy, cz = crop_size
                    volume_crop = grid[x:x + cx, y:y + cy, z:z + cz]
                    hard_crop = hard_label[x:x + cx, y:y + cy, z:z + cz]
                    soft_crop = soft_label[x:x + cx, y:y + cy, z:z + cz]

                    is_empty = np.sum(hard_crop) == 0

                    keep_empty = config['data'].get('keep_empty_block', False)
                    # Decide whether to save this crop, updating the global counters.
                    if not is_empty:
                        should_save = True
                        non_empty_count += 1
                    elif keep_empty and local_empty_counter < max_empty_per_sample:
                        # For empty crops, check the balancing ratio.
                        # For the very first overall saved crop (i.e., when no crop has been saved),
                        # allow saving regardless of empty/non-empty balance.
                        if (non_empty_count + empty_count) == 0:
                            should_save = True
                            empty_count += 1
                            local_empty_counter += 1
                        # If there are no non-empty crops yet (but already saved one empty),
                        # do not allow additional empty crops.
                        elif non_empty_count == 0:
                            should_save = False
                        else:
                            # Enforce the ratio: save an empty crop only if
                            # empty_count < non_empty_count * max_empty_ratio.
                            if empty_count < non_empty_count * max_empty_ratio:
                                should_save = True
                                empty_count += 1
                                local_empty_counter += 1
                            else:
                                should_save = False
                    else:
                        should_save = False

                    if should_save:
                        if rotation_samples > 0:
                            rotations = sample_rotation_params(rotation_samples)
                        else:
                            rotations = [None]

                        for r_idx, rot_param in enumerate(rotations):
                            vol_aug = volume_crop.copy()
                            hard_aug = hard_crop.copy()
                            soft_aug = soft_crop.copy()

                            if rot_param is not None:
                                vol_aug = apply_rotation(vol_aug, rot_param).copy()
                                hard_aug = apply_rotation(hard_aug, rot_param).copy()
                                soft_aug = apply_rotation(soft_aug, rot_param).copy()

                            rotation_suffix = f"_r{r_idx}" if rotation_samples > 0 else ""
                            # Append an empty suffix to the filename if this crop is empty.
                            empty_suffix = "_E" if is_empty else ""
                            filename = f"{file_prefix}_c{local_crop_counter}{rotation_suffix}{empty_suffix}.npz"
                            file_path = os.path.join(cache_dir, filename)
                            with_calpha = np.array([not is_empty], dtype=np.float32)

                            np.savez_compressed(file_path,
                                                volume=vol_aug,
                                                hard_label=hard_aug,
                                                soft_label=soft_aug,
                                                voxel_size=voxel_size,
                                                global_origin=origin,
                                                with_calpha=with_calpha)
                        local_crop_counter += 1

                logging.debug(f"Processed {file_prefix} version {version_idx}: {local_crop_counter} crops")
        except Exception as e:
            logging.error(f"Error processing {sample}: {e}")
            continue

    logging.info("Data caching completed.")
    logging.info(f"Generated {non_empty_count} non-empty blocks and {empty_count} empty blocks")
    
    return cache_dir

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

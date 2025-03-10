import os
import numpy as np
import torch
import random
from sklearn.cluster import MeanShift

# Import the sliding window function from dataset.py
from Calpha.dataset.dataset import get_sliding_window_starts
# Import the preprocessor class from preprocess.py
from Calpha.dataset.preprocess import CryoEMPreprocessor
# Import your postprocessing class (assumed to be defined in inference/postprocess.py)
from Calpha.inference.postprocess import CryoPostprocessor

# Fixed seed for reproducibility (ensures the same files are selected every time)
np.random.seed(42)
random.seed(42)

def reconstruct_global_prediction(volume, crop_size=64, overlap=2, model=None, device='cpu'):
    """
    Reconstruct the full density map via sliding window extraction.
    Uses get_sliding_window_starts from dataset.py to obtain indices.
    In this example, we assume crop_size is an integer and convert it to a tuple.
    Overlap here is interpreted as overlap size (like 2) as used by your functions.
    """
    # Convert crop_size to tuple if necessary.
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)
    
    # Get sliding window start positions using your provided function.
    indices = get_sliding_window_starts(volume.shape, crop_size, overlap)
    
    # Initialize a global prediction map (using -infinity so that max fusion works correctly)
    global_pred = np.full(volume.shape, -np.inf, dtype=np.float32)
    
    for (i, j, k) in indices:
        # Extract the sub-cube for this window.
        crop = volume[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]
        # If a model is specified, pass the crop (after converting to tensor) for prediction.
        if model is not None:
            crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(crop_tensor).cpu().squeeze().numpy()
        else:
            # Otherwise, use the crop directly (assume it encodes predictions).
            pred = crop
        # Fuse predicted values via elementwise maximum
        global_pred[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]] = np.maximum(
            global_pred[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]],
            pred
        )
    # Replace -infinity with zeros where no prediction was made and apply the sigmoid.
    global_pred[global_pred == -np.inf] = 0
    global_prob = 1 / (1 + np.exp(-global_pred))
    return global_prob

def evaluate_file(npz_path, model=None, device='cpu', crop_size=64, overlap=2):
    """
    Evaluate one file:
    • Reads npz and cif from the same subfolder using CryoEMPreprocessor.
    • Reconstructs the global prediction map via sliding window fusion.
    • Applies MeanShift to cluster the predicted CA coordinates.
    • Computes average distance error vs. ground-truth CA positions.
    """
    # Create a minimal config for the preprocessor (the npz file already provides voxel_size and global_origin)
    config = {'data': {'voxel_size': 1.6638, 'global_origin': np.array([0, 0, 0]), 'd0': 3.0}}
    preprocessor = CryoEMPreprocessor(config)
    data = np.load(npz_path, allow_pickle=True)
    # The npz includes 'grid' that is the density map.
    global_volume = data['grid']
    
    # Reconstruct global prediction
    global_pred = reconstruct_global_prediction(global_volume, crop_size, overlap, model, device)
    
    # Use postprocessor (MeanShift clustering) as defined in your inference/postprocess.py
    cp = CryoPostprocessor(voxel_size=config['data']['voxel_size'], origin=config['data']['global_origin'])
    pred_coords = cp.refine_coordinates(global_pred, bandwidth=1.7)
    
    # Ground-truth CA coordinates (cif file is assumed to be in the same folder)
    cif_path = npz_path.replace('.npz', '.cif')
    if os.path.exists(cif_path):
        gt_coords = preprocessor.parse_cif(cif_path)
    else:
        gt_coords = np.empty((0, 3))
    
    # Compute average Euclidean distance error:
    errors = []
    for p in pred_coords:
        d = np.linalg.norm(gt_coords - p, axis=1)
        if len(d) > 0:
            errors.append(np.min(d))
    mean_error = np.mean(errors) if errors else np.nan
    
    return pred_coords, gt_coords, mean_error

def main_evaluation(data_dir, model=None, device='cpu', crop_size=64, overlap=2, n_files=30):
    """
    Main evaluation routine:
    • List all subfolders under data_dir (each has one npz and one cif).
    • Randomly select n_files (fixed seed ensures consistency).
    • Evaluate each file and print per-file and overall error metrics.
    """
    npz_files = []
    for entry in os.listdir(data_dir):
        subfolder = os.path.join(data_dir, entry)
        if os.path.isdir(subfolder):
            for f in os.listdir(subfolder):
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(subfolder, f))
    npz_files = sorted(npz_files)
    random.shuffle(npz_files)
    selected = npz_files[:n_files]
    all_errors = []
    for f in selected:
        pred_coords, gt_coords, err = evaluate_file(f, model, device, crop_size, overlap)
        print(f"{f}: Mean distance error = {err:.3f}")
        all_errors.append(err)
    overall_error = np.nanmean(all_errors)
    print(f"Overall average distance error on {len(selected)} files: {overall_error:.3f}")

if __name__ == "__main__":
    # Change data_directory as needed.
    data_directory = "/root/autodl-tmp/CryFold/Datasets"
    # Model can be passed if available; here we run evaluation without a model.
    main_evaluation(data_directory, model=None, device='cuda', crop_size=64, overlap=2, n_files=30)

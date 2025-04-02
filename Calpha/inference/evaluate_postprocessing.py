import os
import numpy as np
import torch
import random
import logging
from sklearn.cluster import MeanShift

# Import the sliding window function from dataset.py
from Calpha.dataset.dataset import get_sliding_window_starts
# Import the preprocessor class from preprocess.py
from Calpha.dataset.preprocess import CryoEMPreprocessor
# Import your postprocessing class (assumed to be defined in inference/postprocess.py)
from Calpha.inference.postprocess import CryoPostprocessor
# Import binary classifier inference
from Calpha.inference.binary_inference import BinaryInferenceProcessor
from Calpha.model.binary_classifier import BinaryClassifierUNet

# Fixed seed for reproducibility (ensures the same files are selected every time)
np.random.seed(42)
random.seed(42)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reconstruct_global_prediction(volume, crop_size=64, overlap=2, model=None, binary_model=None, config=None, device='cpu'):
    """
    Reconstruct the full density map via sliding window extraction.
    Uses get_sliding_window_starts from dataset.py to obtain indices.
    In this example, we assume crop_size is an integer and convert it to a tuple.
    Overlap here is interpreted as overlap size (like 2) as used by your functions.
    
    If binary_model is provided, it will be used to filter regions without C-alpha atoms.
    """
    # Convert crop_size to tuple if necessary.
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)
    
    # Get sliding window start positions using your provided function.
    indices = get_sliding_window_starts(volume.shape, crop_size, overlap)
    
    # Initialize a global prediction map (using -infinity so that max fusion works correctly)
    global_pred = np.full(volume.shape, -np.inf, dtype=np.float32)
    
    # If binary model is provided, use it to filter regions
    binary_mask = None
    if binary_model is not None and config is not None:
        logging.info("Using binary classifier to filter regions...")
        processor = BinaryInferenceProcessor(binary_model, config, device)
        binary_mask, _ = processor.process_volume(volume)
        logging.info(f"Binary mask created with {np.sum(binary_mask)} positive voxels out of {binary_mask.size}")
    
    for (i, j, k) in indices:
        # Extract the sub-cube for this window.
        crop = volume[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]
        
        # Skip regions without C-alpha atoms according to binary classifier
        if binary_mask is not None:
            region_mask = binary_mask[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]
            if np.sum(region_mask) < 10:  # Skip if less than 10 positive voxels
                continue
        
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
    
    # Apply binary mask as a final filter if available
    if binary_mask is not None:
        global_prob = global_prob * binary_mask
        
    return global_prob

def evaluate_file(npz_path, model=None, binary_model=None, device='cpu', crop_size=64, overlap=2):
    """
    Evaluate one file:
    • Reads npz and cif from the same subfolder using CryoEMPreprocessor.
    • Reconstructs the global prediction map via sliding window fusion.
    • Applies MeanShift to cluster the predicted CA coordinates.
    • Computes average distance error vs. ground-truth CA positions.
    If binary_model is provided, it will be used to filter regions without C-alpha atoms.
    """
    # Create a minimal config for the preprocessor
    config = {'data': {'voxel_size': 1.6638, 'global_origin': np.array([0, 0, 0]), 'd0': 3.0}}
    
    # Add inference config if binary model is provided
    if binary_model is not None:
        config['inference'] = {
            'step_size': 64,
            'overlap': 16
        }
    
    preprocessor = CryoEMPreprocessor(config)
    data = np.load(npz_path, allow_pickle=True)
    # The npz includes 'grid' that is the density map.
    global_volume = data['grid']
    
    # Reconstruct global prediction
    global_pred = reconstruct_global_prediction(
        global_volume, crop_size, overlap, model, binary_model, config, device
    )
    
    # Use postprocessor (MeanShift clustering) as defined in your inference/postprocess.py
    cp = CryoPostprocessor(voxel_size=config['data']['voxel_size'], origin=config['data']['global_origin'])
    pred_coords = cp.refine_coordinates(global_pred, bandwidth=1.7)
    
    # Ground-truth CA coordinates (cif file is assumed to be in the same folder)
    cif_path = npz_path.replace('.npz', '.cif')
    if os.path.exists(cif_path):
        gt_coords = preprocessor.parse_cif(cif_path)
    else:
        gt_coords = np.empty((0, 3))
    
    # Compute metrics
    metrics = {}
    
    # Compute average Euclidean distance error
    errors = []
    for p in pred_coords:
        d = np.linalg.norm(gt_coords - p, axis=1)
        if len(d) > 0:
            errors.append(np.min(d))
    
    metrics['mean_error'] = np.mean(errors) if errors else np.nan
    
    # Calculate RMSE
    metrics['rmse'] = np.sqrt(np.mean(np.array(errors)**2)) if errors else np.nan
    
    # Calculate precision and recall
    if len(gt_coords) > 0 and len(pred_coords) > 0:
        # For each ground truth point, find the closest predicted point
        gt_to_pred = []
        for g in gt_coords:
            d = np.linalg.norm(pred_coords - g, axis=1)
            gt_to_pred.append(np.min(d))
        
        # Calculate precision (% of predicted points that are close to ground truth)
        metrics['precision'] = np.mean(np.array(errors) < 2.0) if errors else 0.0
        
        # Calculate recall (% of ground truth points that are close to predictions)
        metrics['recall'] = np.mean(np.array(gt_to_pred) < 2.0)
        
        # Calculate F1 score
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
    
    metrics['num_pred'] = len(pred_coords)
    metrics['num_gt'] = len(gt_coords)
    
    return pred_coords, gt_coords, metrics

def main_evaluation(data_dir, model=None, binary_model=None, device='cpu', crop_size=64, overlap=2, n_files=30):
    """
    Main evaluation routine:
    • List all subfolders under data_dir (each has one npz and one cif).
    • Randomly select n_files (fixed seed ensures consistency).
    • Evaluate each file and print per-file and overall error metrics.
    
    If binary_model is provided, it will be used to filter regions without C-alpha atoms.
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
    
    # Initialize metrics
    all_metrics = {
        'mean_error': [],
        'rmse': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'num_pred': [],
        'num_gt': []
    }
    
    # Process each file
    for f in selected:
        pred_coords, gt_coords, metrics = evaluate_file(f, model, binary_model, device, crop_size, overlap)
        
        # Print file-specific metrics
        print(f"{f}:")
        print(f"  Mean distance error = {metrics['mean_error']:.3f}")
        print(f"  RMSE = {metrics['rmse']:.3f}")
        print(f"  Precision = {metrics['precision']:.3f}")
        print(f"  Recall = {metrics['recall']:.3f}")
        print(f"  F1 Score = {metrics['f1']:.3f}")
        print(f"  Predicted points: {metrics['num_pred']}, Ground truth points: {metrics['num_gt']}")
        
        # Accumulate metrics
        for key, value in metrics.items():
            all_metrics[key].append(value)
    
    # Calculate overall metrics
    overall_metrics = {}
    for key, values in all_metrics.items():
        overall_metrics[key] = np.nanmean(values)
    
    # Print overall metrics
    print("\nOverall metrics on {} files:".format(len(selected)))
    print(f"  Mean distance error = {overall_metrics['mean_error']:.3f}")
    print(f"  RMSE = {overall_metrics['rmse']:.3f}")
    print(f"  Precision = {overall_metrics['precision']:.3f}")
    print(f"  Recall = {overall_metrics['recall']:.3f}")
    print(f"  F1 Score = {overall_metrics['f1']:.3f}")
    print(f"  Average predicted points: {overall_metrics['num_pred']:.1f}")
    print(f"  Average ground truth points: {overall_metrics['num_gt']:.1f}")
    
    return overall_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate C-alpha prediction models")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/CryFold/Datasets",
                        help="Directory containing dataset folders")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to segmentation model checkpoint")
    parser.add_argument("--binary_model", type=str, default=None,
                        help="Path to binary classifier model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--crop_size", type=int, default=64,
                        help="Crop size for sliding window")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Overlap size for sliding window")
    parser.add_argument("--n_files", type=int, default=30,
                        help="Number of files to evaluate")
    
    args = parser.parse_args()
    
    # Load models if specified
    segmentation_model = None
    if args.model is not None:
        from Calpha.model.segmentation_model_droupout import SegmentationModelMini
        segmentation_model = SegmentationModelMini()
        checkpoint = torch.load(args.model, map_location=args.device)
        segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        segmentation_model.to(args.device)
        segmentation_model.eval()
        logging.info(f"Loaded segmentation model from {args.model}")
    
    binary_model = None
    if args.binary_model is not None:
        binary_model = BinaryClassifierUNet()
        checkpoint = torch.load(args.binary_model, map_location=args.device)
        binary_model.load_state_dict(checkpoint['model_state_dict'])
        binary_model.to(args.device)
        binary_model.eval()
        logging.info(f"Loaded binary classifier model from {args.binary_model}")
    
    # Run evaluation
    main_evaluation(
        args.data_dir,
        model=segmentation_model,
        binary_model=binary_model,
        device=args.device,
        crop_size=args.crop_size,
        overlap=args.overlap,
        n_files=args.n_files
    )

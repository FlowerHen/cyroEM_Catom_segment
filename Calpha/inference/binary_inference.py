import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from scipy.ndimage import zoom

class BinaryInferenceProcessor:
    """
    Processor for binary classification inference with full coverage strategy.
    This class handles the inference process for the binary classifier model,
    implementing a sliding window approach with full coverage of the volume.
    """
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.crop_size = 64  # Default crop size
        self.step_size = config['inference'].get('step_size', 64)  # Default step size for sliding window
        self.overlap = config['inference'].get('overlap', 16)  # Default overlap
        self.threshold = config['inference'].get('threshold', 0.5)  # Default threshold for binary classification
        
    def process_volume(self, volume):
        """
        Process a full volume using sliding window with full coverage.
        
        Args:
            volume: 3D numpy array representing the density map
            
        Returns:
            binary_map: 3D numpy array with binary classification results (0 or 1)
            noise_levels: 3D numpy array with estimated noise levels
        """
        # Initialize output maps
        binary_map = np.zeros_like(volume, dtype=np.float32)
        noise_map = np.zeros_like(volume, dtype=np.float32)
        count_map = np.zeros_like(volume, dtype=np.int32)
        
        # Get sliding window starts with step size
        main_indices = self._get_main_indices(volume.shape)
        overlap_indices = self._get_overlap_indices(volume.shape)
        total_windows = len(main_indices) + len(overlap_indices)
        pbar = tqdm(total=total_windows, desc="Processing volume")
        
        # Process main grid
        logging.info(f"Processing main grid with {len(main_indices)} windows")
        for (i, j, k) in main_indices:
            # Extract crop
            crop = self._extract_crop(volume, i, j, k)
            
            # Process crop
            with torch.no_grad():
                crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                pred = self.model(crop_tensor)
                binary_pred = (torch.sigmoid(pred) > 0.5).float().cpu().squeeze().numpy()
                noise_level = self.model.compute_background_noise(crop_tensor).cpu().numpy()[0]
            
            self._update_maps(binary_map, noise_map, count_map, binary_pred, noise_level, i, j, k)
            pbar.update(1)
        
        # Process overlap regions
        logging.info(f"Processing overlap regions with {len(overlap_indices)} windows")
        for (i, j, k) in overlap_indices:
            # Extract crop
            crop = self._extract_crop(volume, i, j, k)
            
            # Process crop
            with torch.no_grad():
                crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                pred = self.model(crop_tensor)
                binary_pred = (torch.sigmoid(pred) > 0.5).float().cpu().squeeze().numpy()
                noise_level = self.model.compute_background_noise(crop_tensor).cpu().numpy()[0]
            
            self._update_maps(binary_map, noise_map, count_map, binary_pred, noise_level, i, j, k)
            pbar.update(1)
        
        # Average the results and convert to binary (0 or 1)
        binary_map = np.divide(binary_map, count_map, out=np.zeros_like(binary_map), where=count_map > 0)
        noise_map = np.divide(noise_map, count_map, out=np.zeros_like(noise_map), where=count_map > 0)
        
        # Convert to binary (0 or 1) - any value > 0.5 means the majority of predictions were positive
        binary_map = (binary_map > 0.5).astype(np.float32)
        
        return binary_map, noise_map
    
    def _get_main_indices(self, volume_shape):
        """Get indices for the main grid with step size"""
        indices = []
        for i in range(0, volume_shape[0], self.step_size):
            for j in range(0, volume_shape[1], self.step_size):
                for k in range(0, volume_shape[2], self.step_size):
                    # Check if we can extract a full crop
                    if (i + self.crop_size <= volume_shape[0] and
                        j + self.crop_size <= volume_shape[1] and
                        k + self.crop_size <= volume_shape[2]):
                        indices.append((i, j, k))
        return indices
    
    def _get_overlap_indices(self, volume_shape):
        """Get indices for the overlap regions at the edges"""
        indices = []
        
        # X-axis overlap
        if volume_shape[0] % self.step_size != 0:
            i = max(0, volume_shape[0] - self.crop_size)
            for j in range(0, volume_shape[1], self.step_size):
                for k in range(0, volume_shape[2], self.step_size):
                    if (j + self.crop_size <= volume_shape[1] and
                        k + self.crop_size <= volume_shape[2]):
                        indices.append((i, j, k))
        
        # Y-axis overlap
        if volume_shape[1] % self.step_size != 0:
            j = max(0, volume_shape[1] - self.crop_size)
            for i in range(0, volume_shape[0], self.step_size):
                for k in range(0, volume_shape[2], self.step_size):
                    if (i + self.crop_size <= volume_shape[0] and
                        k + self.crop_size <= volume_shape[2]):
                        indices.append((i, j, k))
        
        # Z-axis overlap
        if volume_shape[2] % self.step_size != 0:
            k = max(0, volume_shape[2] - self.crop_size)
            for i in range(0, volume_shape[0], self.step_size):
                for j in range(0, volume_shape[1], self.step_size):
                    if (i + self.crop_size <= volume_shape[0] and
                        j + self.crop_size <= volume_shape[1]):
                        indices.append((i, j, k))
        
        # Corner cases (XY, XZ, YZ, XYZ)
        if volume_shape[0] % self.step_size != 0 and volume_shape[1] % self.step_size != 0:
            i = max(0, volume_shape[0] - self.crop_size)
            j = max(0, volume_shape[1] - self.crop_size)
            for k in range(0, volume_shape[2], self.step_size):
                if k + self.crop_size <= volume_shape[2]:
                    indices.append((i, j, k))
        
        if volume_shape[0] % self.step_size != 0 and volume_shape[2] % self.step_size != 0:
            i = max(0, volume_shape[0] - self.crop_size)
            k = max(0, volume_shape[2] - self.crop_size)
            for j in range(0, volume_shape[1], self.step_size):
                if j + self.crop_size <= volume_shape[1]:
                    indices.append((i, j, k))
        
        if volume_shape[1] % self.step_size != 0 and volume_shape[2] % self.step_size != 0:
            j = max(0, volume_shape[1] - self.crop_size)
            k = max(0, volume_shape[2] - self.crop_size)
            for i in range(0, volume_shape[0], self.step_size):
                if i + self.crop_size <= volume_shape[0]:
                    indices.append((i, j, k))
        
        if (volume_shape[0] % self.step_size != 0 and 
            volume_shape[1] % self.step_size != 0 and 
            volume_shape[2] % self.step_size != 0):
            i = max(0, volume_shape[0] - self.crop_size)
            j = max(0, volume_shape[1] - self.crop_size)
            k = max(0, volume_shape[2] - self.crop_size)
            indices.append((i, j, k))
        
        return indices
    
    def _extract_crop(self, volume, i, j, k):
        """Extract a crop from the volume"""
        # Handle boundary cases
        end_i = min(i + self.crop_size, volume.shape[0])
        end_j = min(j + self.crop_size, volume.shape[1])
        end_k = min(k + self.crop_size, volume.shape[2])
        
        # Extract crop
        crop = volume[i:end_i, j:end_j, k:end_k]
        
        # Pad if necessary
        if crop.shape != (self.crop_size, self.crop_size, self.crop_size):
            pad_i = self.crop_size - crop.shape[0]
            pad_j = self.crop_size - crop.shape[1]
            pad_k = self.crop_size - crop.shape[2]
            
            crop = np.pad(crop, ((0, pad_i), (0, pad_j), (0, pad_k)), mode='constant')
        
        return crop
    
    def _update_maps(self, binary_map, noise_map, count_map, binary_pred, noise_level, i, j, k):
        """Update the output maps with the prediction results"""
        # Handle boundary cases
        end_i = min(i + self.crop_size, binary_map.shape[0])
        end_j = min(j + self.crop_size, binary_map.shape[1])
        end_k = min(k + self.crop_size, binary_map.shape[2])
        
        # Calculate crop dimensions
        crop_i = end_i - i
        crop_j = end_j - j
        crop_k = end_k - k
        
        # Update maps
        binary_map[i:end_i, j:end_j, k:end_k] += binary_pred[:crop_i, :crop_j, :crop_k]
        noise_map[i:end_i, j:end_j, k:end_k] += noise_level
        count_map[i:end_i, j:end_j, k:end_k] += 1

def run_binary_inference(model_path, volume_path, output_dir, config):
    """
    Run binary inference on a volume.
    
    Args:
        model_path: Path to the trained model checkpoint
        volume_path: Path to the volume file (.npz)
        output_dir: Directory to save the results
        config: Configuration dictionary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'binary_inference.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load model
    device = torch.device(config['inference'].get('device', 'cuda'))
    from Calpha.model.binary_classifier import SimpleBinaryClassifier
    model = SimpleBinaryClassifier(
        window_size=config['binary_training']['window_size'],
        stride=config['binary_training']['stride']
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Loaded model from {model_path}")
    
    # Load volume
    from Calpha.dataset.preprocess import CryoEMPreprocessor
    preprocessor = CryoEMPreprocessor(config)
    grid_data = preprocessor.load_npz(volume_path)
    volume = grid_data['grid']
    
    logging.info(f"Loaded volume from {volume_path} with shape {volume.shape}")
    
    # Initialize inference processor
    processor = BinaryInferenceProcessor(model, config, device)
    
    # Process volume
    binary_map, noise_map = processor.process_volume(volume)
    
    # Save results
    volume_name = os.path.splitext(os.path.basename(volume_path))[0]
    np.savez_compressed(
        os.path.join(output_dir, f"{volume_name}_binary.npz"),
        binary_map=binary_map,
        noise_map=noise_map,
        voxel_size=grid_data['voxel_size'],
        global_origin=grid_data['global_origin']
    )
    
    logging.info(f"Saved results to {output_dir}")
    
    return binary_map, noise_map

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Binary classifier inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--volume", type=str, required=True, help="Path to volume file (.npz)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="/root/project/Calpha/Calpha/config/base.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run inference
    run_binary_inference(args.model, args.volume, args.output, config)
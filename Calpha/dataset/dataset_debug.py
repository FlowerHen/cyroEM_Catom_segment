import os
import numpy as np
import torch
import yaml
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, List

from Calpha.dataset.preprocess import CryoEMPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global sampling parameters
SAMPLE_SIZE_ORIGINAL = 50  # Number of original samples to analyze
SAMPLE_SIZE_CACHED = 500   # Number of cached samples to analyze
SAMPLE_SIZE_VISUALIZE = 5  # Number of samples to visualize

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_data(config: Dict) -> None:
    """
    Main function to analyze dataset characteristics and generate debug outputs.
    Outputs are saved in data_cache/.debug directory.
    """
    
    # Create output directory
    debug_dir = os.path.join(config['data']['cache_dir'], '.debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Analyze original samples
    original_stats = analyze_original_samples(config, debug_dir)
    
    # 2. Analyze cached data
    cached_stats = analyze_cached_data(config, debug_dir)
    
    # 3. Check cache integrity
    integrity_stats = check_data_cache_integrity(config, debug_dir)
    
    # Generate summary report
    generate_summary_report(original_stats, cached_stats, integrity_stats, debug_dir)

def analyze_original_samples(config: Dict, output_dir: str) -> Dict:
    """Analyze characteristics of original samples (before cropping)."""
    data_dir = config['data']['root_dir']
    preprocessor = CryoEMPreprocessor(config)
    
    # Statistics containers
    stats = {
        'ca_counts': [],
        'ca_density_values': [],
        'non_ca_density_values': [],
        'ca_spatial_distances': [],
        'per_sample_densities': defaultdict(list)  # New: store densities per sample
    }
    
    entries = [e for e in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, e))]
    sample_size = min(len(entries), SAMPLE_SIZE_ORIGINAL)
    
    logging.info(f"Analyzing {sample_size} original samples")
    
    for entry in tqdm(entries[:sample_size], desc="Original samples"):
        entry_path = os.path.join(data_dir, entry)
        npz_files = [f for f in os.listdir(entry_path) if f.endswith('.npz')]
        cif_files = [f for f in os.listdir(entry_path) if f.endswith('.cif')]
        
        if not npz_files or not cif_files:
            continue
            
        try:
            # Load data
            grid_data = preprocessor.load_npz(os.path.join(entry_path, npz_files[0]))
            ca_coords = preprocessor.parse_cif(os.path.join(entry_path, cif_files[0]))
            
            # Collect statistics
            stats['ca_counts'].append(len(ca_coords))
            
            # Convert coordinates to grid indices
            grid_indices = ((ca_coords - grid_data['global_origin']) / 
                          grid_data['voxel_size']).astype(int)
            
            # Filter valid indices
            valid_mask = np.all((grid_indices >= 0) & 
                             (grid_indices < np.array(grid_data['grid'].shape)), axis=1)
            valid_indices = grid_indices[valid_mask]
            
            # Extract density values
            for idx in valid_indices:
                stats['ca_density_values'].append(grid_data['grid'][tuple(idx)])
            
            # Sample non-C-alpha positions
            for _ in range(min(len(valid_indices), 1000)):
                x, y, z = np.random.randint(0, grid_data['grid'].shape, size=3)
                if not any(np.array_equal([x,y,z], ca_idx) for ca_idx in valid_indices):
                    stats['non_ca_density_values'].append(grid_data['grid'][x, y, z])
            
            # Calculate distances
            if len(ca_coords) > 1:
                for i in range(len(ca_coords)):
                    for j in range(i+1, min(i+10, len(ca_coords))):  # Limit pairs
                        stats['ca_spatial_distances'].append(np.linalg.norm(ca_coords[i] - ca_coords[j]))
            
        except Exception as e:
            logging.error(f"Error analyzing {entry}: {e}")
    
    # Generate visualizations
    plot_ca_atom_distribution(stats['ca_counts'], output_dir)
    plot_density_distribution(stats, output_dir)
    plot_spatial_distances(stats['ca_spatial_distances'], output_dir)
    plot_per_sample_densities(stats['per_sample_densities'], output_dir)  # New visualization
    
    return stats

def analyze_cached_data(config: Dict, output_dir: str) -> Dict:
    """Analyze characteristics of cached data (after cropping)."""
    cache_dir = config['data']['cache_dir']
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
    sample_size = min(len(cache_files), SAMPLE_SIZE_CACHED)
    
    stats = {
        'positive_ratio': [],
        'positive_density_values': [],
        'negative_density_values': [],
        'positive_spatial_distribution': defaultdict(int),
        'per_sample_means': []  # New: store mean density per cached sample
    }
    
    logging.info(f"Analyzing {sample_size} cached samples")
    
    for file_name in tqdm(cache_files[:sample_size], desc="Cached data"):
        try:
            data = np.load(os.path.join(cache_dir, file_name))
            volume = data['volume']
            hard_label = data['hard_label']
            
            # Calculate positive ratio
            pos_ratio = np.sum(hard_label) / hard_label.size
            stats['positive_ratio'].append(pos_ratio)
            
            # Sample positive and negative voxels using threshold from config
            threshold = config['training']['prediction_threshold']
            pos_mask = hard_label > threshold
            neg_mask = ~pos_mask

            if np.any(pos_mask):
                pos_indices = np.argwhere(pos_mask)
                sample_size_pos = min(len(pos_indices), 1000)
                for idx in pos_indices[np.random.choice(len(pos_indices), sample_size_pos, replace=False)]:
                    stats['positive_density_values'].append(volume[tuple(idx)])
                    
                    # Track spatial distribution
                    norm_pos = idx / np.array(volume.shape)
                    bin_pos = tuple((norm_pos * 10).astype(int))
                    stats['positive_spatial_distribution'][bin_pos] += 1
            
            if np.any(neg_mask):
                neg_indices = np.argwhere(neg_mask)
                sample_size_neg = min(len(neg_indices), 1000)
                for idx in neg_indices[np.random.choice(len(neg_indices), sample_size_neg, replace=False)]:
                    stats['negative_density_values'].append(volume[tuple(idx)])
            
            # Calculate and store mean density for this cached sample
            if len(stats['positive_density_values']) > 0:
                sample_mean = np.mean(stats['positive_density_values'][-1000:])  # Use last 1000 values
                stats['per_sample_means'].append(sample_mean)
            
        except Exception as e:
            logging.error(f"Error analyzing {file_name}: {e}")
    
    # Generate visualizations
    plot_positive_ratio_distribution(stats['positive_ratio'], output_dir)
    plot_cached_density_distribution(stats, output_dir)
    plot_spatial_heatmap(stats['positive_spatial_distribution'], output_dir)
    plot_cached_means_distribution(stats['per_sample_means'], output_dir)  # New visualization
    
    # Perform statistical tests
    if len(stats['per_sample_means']) > 1:
        overall_mean = np.mean(stats['per_sample_means'])
        stats['cached_means_stats'] = {
            'mean': overall_mean,
            'std': np.std(stats['per_sample_means']),
            'min': np.min(stats['per_sample_means']),
            'max': np.max(stats['per_sample_means'])
        }
    
    return stats

def check_data_cache_integrity(config: Dict, output_dir: str) -> Dict:
    """Check for potential issues in the data cache."""
    cache_dir = config['data']['cache_dir']
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
    
    stats = {
        'total_files': len(cache_files),
        'corrupted_files': 0,
        'empty_volumes': 0,
        'empty_labels': 0,
        'inconsistent_shapes': 0,
        'total_voxels': 0,
        'total_positive_voxels': 0
    }
    
    expected_keys = {'volume', 'hard_label', 'soft_label', 'voxel_size', 'global_origin'}
    
    logging.info(f"Checking integrity of {len(cache_files)} cached files")
    
    for file_name in tqdm(cache_files, desc="Cache integrity"):
        try:
            data = np.load(os.path.join(cache_dir, file_name))
            
            # Check keys
            if not expected_keys.issubset(data.keys()):
                stats['corrupted_files'] += 1
                continue
                
            volume = data['volume']
            hard_label = data['hard_label']
            
            # Check shapes
            if volume.shape != hard_label.shape:
                stats['inconsistent_shapes'] += 1
                
            # Check empty
            if np.all(volume == 0):
                stats['empty_volumes'] += 1
                
            if np.all(hard_label == 0):
                stats['empty_labels'] += 1
                
            # Count positives
            stats['total_voxels'] += hard_label.size
            stats['total_positive_voxels'] += np.sum(hard_label)
            
        except Exception as e:
            stats['corrupted_files'] += 1
            logging.error(f"Error loading {file_name}: {e}")
    
    # Generate integrity report
    generate_integrity_report(stats, output_dir)
    return stats

def generate_integrity_report(stats: Dict, output_dir: str) -> None:
    """Generate markdown report of cache integrity check."""
    report_path = os.path.join(output_dir, 'integrity_report.md')
    positive_ratio = stats['total_positive_voxels'] / stats['total_voxels'] if stats['total_voxels'] > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("# Data Cache Integrity Report\n\n")
        f.write(f"- **Total files:** {stats['total_files']}\n")
        f.write(f"- **Corrupted files:** {stats['corrupted_files']} ({stats['corrupted_files']/stats['total_files']*100:.1f}%)\n")
        f.write(f"- **Empty volumes:** {stats['empty_volumes']}\n")
        f.write(f"- **Empty labels:** {stats['empty_labels']}\n")
        f.write(f"- **Inconsistent shapes:** {stats['inconsistent_shapes']}\n")
        f.write(f"- **Overall positive ratio:** {positive_ratio:.4f}\n")
        f.write(f"- **Imbalance ratio:** {(1-positive_ratio)/positive_ratio:.1f}:1\n")

def generate_summary_report(original_stats: Dict, cached_stats: Dict, integrity_stats: Dict, output_dir: str) -> None:
    """Generate markdown summary report with key findings."""
    report_path = os.path.join(output_dir, 'summary_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Dataset Debug Summary\n\n")
        
        # Original samples section
        f.write("## Original Samples Analysis\n")
        f.write(f"- Samples analyzed: {len(original_stats['ca_counts'])}\n")
        f.write(f"- Average C-alpha atoms: {np.mean(original_stats['ca_counts']):.1f}\n")
        f.write(f"- C-alpha density mean: {np.mean(original_stats['ca_density_values']):.4f}\n")
        f.write(f"- Non-C-alpha density mean: {np.mean(original_stats['non_ca_density_values']):.4f}\n")
        f.write("\n![C-alpha Distribution](ca_atoms_distribution.png)\n")
        
        # Cached data section
        f.write("\n## Cached Data Analysis\n")
        f.write(f"- Average positive ratio: {np.mean(cached_stats['positive_ratio']):.4f}\n")
        f.write(f"- Positive density mean: {np.mean(cached_stats['positive_density_values']):.4f}\n")
        f.write(f"- Negative density mean: {np.mean(cached_stats['negative_density_values']):.4f}\n")
        f.write("\n![Cached Density](cached_density_distribution.png)\n")
        
        # Integrity section
        f.write("\n## Cache Integrity\n")
        f.write(f"- Corrupted files: {integrity_stats['corrupted_files']} ({integrity_stats['corrupted_files']/integrity_stats['total_files']*100:.1f}%)\n")
        f.write(f"- Inconsistent shapes: {integrity_stats['inconsistent_shapes']}\n")
        
        # New statistical comparison section
        f.write("\n## Statistical Comparisons\n")
        if 'cached_means_stats' in cached_stats:
            f.write("### Cached Data Means\n")
            f.write(f"- Overall mean: {cached_stats['cached_means_stats']['mean']:.4f}\n")
            f.write(f"- Standard deviation: {cached_stats['cached_means_stats']['std']:.4f}\n")
            f.write(f"- Range: {cached_stats['cached_means_stats']['min']:.4f} to {cached_stats['cached_means_stats']['max']:.4f}\n")
        
        if original_stats['ca_density_values'] and cached_stats['positive_density_values']:
            t_stat, p_value = stats.ttest_ind(
                original_stats['ca_density_values'],
                cached_stats['positive_density_values'],
                equal_var=False
            )
            f.write("\n### Original vs Cached Comparison\n")
            f.write(f"- T-test p-value: {p_value:.4f}\n")
            f.write(f"- Original mean: {np.mean(original_stats['ca_density_values']):.4f}\n")
            f.write(f"- Cached mean: {np.mean(cached_stats['positive_density_values']):.4f}\n")

# Visualization functions
def plot_ca_atom_distribution(ca_counts: List[int], output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(ca_counts, bins=20)
    plt.title('Distribution of C-alpha Atoms per Sample')
    plt.xlabel('Number of C-alpha Atoms')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'ca_atoms_distribution.png'))
    plt.close()

def plot_density_distribution(stats: Dict, output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    if stats['ca_density_values']:
        lower = np.percentile(stats['ca_density_values'], 0.05)
        upper = np.percentile(stats['ca_density_values'], 99.95)
        filtered_ca = [x for x in stats['ca_density_values'] if lower <= x <= upper]
        plt.hist(filtered_ca, bins=50, alpha=0.5, label='C-alpha')
    
    if stats['non_ca_density_values']:
        lower = np.percentile(stats['non_ca_density_values'], 0.05)
        upper = np.percentile(stats['non_ca_density_values'], 99.95)
        filtered_non_ca = [x for x in stats['non_ca_density_values'] if lower <= x <= upper]
        plt.hist(filtered_non_ca, bins=50, alpha=0.5, label='Non-C-alpha')
    plt.title('Density Values Distribution')
    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'density_distribution.png'))
    plt.close()

def plot_spatial_distances(distances: List[float], output_dir: str) -> None:
    if distances:
        plt.figure(figsize=(10, 6))
        lower = np.percentile(distances, 0.05)
        upper = np.percentile(distances, 99.95)
        filtered = [x for x in distances if lower <= x <= upper]
        plt.hist(filtered, bins=50)
        plt.title('C-alpha Atoms Spatial Distances')
        plt.xlabel('Distance (Å)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'ca_distances.png'))
        plt.close()

def plot_positive_ratio_distribution(ratios: List[float], output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=50)
    plt.title('Positive Ratio Distribution in Cached Samples')
    plt.xlabel('Positive Ratio')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'positive_ratio_distribution.png'))
    plt.close()

def plot_cached_density_distribution(stats: Dict, output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    if stats['positive_density_values']:
        lower = np.percentile(stats['positive_density_values'], 0.05)
        upper = np.percentile(stats['positive_density_values'], 99.95)
        filtered_pos = [x for x in stats['positive_density_values'] if lower <= x <= upper]
        plt.hist(filtered_pos, bins=50, alpha=0.5, label='Positive')
    
    if stats['negative_density_values']:
        lower = np.percentile(stats['negative_density_values'], 0.05)
        upper = np.percentile(stats['negative_density_values'], 99.95)
        filtered_neg = [x for x in stats['negative_density_values'] if lower <= x <= upper]
        plt.hist(filtered_neg, bins=50, alpha=0.5, label='Negative')
    plt.title('Density Values in Cached Data')
    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cached_density_distribution.png'))
    plt.close()

def plot_per_sample_densities(per_sample_densities: Dict[str, List[float]], output_dir: str) -> None:
    """Plot density distributions for each original sample separately."""
    plt.figure(figsize=(12, 8))
    has_data = False
    
    for sample_id, densities in list(per_sample_densities.items())[:SAMPLE_SIZE_VISUALIZE]:
        if len(densities) > 10:  # Need enough points for meaningful KDE
            has_data = True
            # Use less aggressive filtering for KDE plots
            lower = np.percentile(densities, 1)  # Changed from 0.05 to 1
            upper = np.percentile(densities, 99)  # Changed from 99.95 to 99
            filtered = [x for x in densities if lower <= x <= upper]
            sns.kdeplot(filtered, label=sample_id, bw_adjust=0.5)  # Added bandwidth adjustment
    
    if has_data:
        plt.title('C-alpha Density Distribution by Sample')
        plt.xlabel('Density Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'per_sample_densities.png'))
        plt.close()
    plt.title('C-alpha Density Distribution by Sample')
    plt.xlabel('Density Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'per_sample_densities.png'))
    plt.close()

def plot_cached_means_distribution(per_sample_means: List[float], output_dir: str) -> None:
    """Plot distribution of mean Calpha values across cached samples."""
    plt.figure(figsize=(10, 6))
    if len(per_sample_means) > 0:
        # Remove 0.05% from each tail
        lower = np.percentile(per_sample_means, 0.05)
        upper = np.percentile(per_sample_means, 99.95)
        filtered = [x for x in per_sample_means if lower <= x <= upper]
        plt.hist(filtered, bins=30, alpha=0.7)
    plt.axvline(np.mean(per_sample_means), color='r', linestyle='dashed', linewidth=1)
    plt.title('Distribution of Mean C-alpha Values in Cached Samples')
    plt.xlabel('Mean Density Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'cached_means_distribution.png'))
    plt.close()

def plot_spatial_heatmap(distribution: Dict, output_dir: str) -> None:
    if distribution:
        heatmap = np.zeros((10, 10, 10))
        for (x, y, z), count in distribution.items():
            if 0 <= x < 10 and 0 <= y < 10 and 0 <= z < 10:
                heatmap[x, y, z] = count
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY projection
        sns.heatmap(np.sum(heatmap, axis=2), ax=axes[0], cmap='viridis')
        axes[0].set_title('XY Projection')
        
        # XZ projection
        sns.heatmap(np.sum(heatmap, axis=1), ax=axes[1], cmap='viridis')
        axes[1].set_title('XZ Projection')
        
        # YZ projection
        sns.heatmap(np.sum(heatmap, axis=0), ax=axes[2], cmap='viridis')
        axes[2].set_title('YZ Projection')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'positive_spatial_distribution.png'))
        plt.close()

if __name__ == "__main__":
    config = load_config("/root/project/Calpha/Calpha/config/base.yaml")
    analyze_data(config)
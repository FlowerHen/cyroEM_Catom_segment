import os
import numpy as np
import torch
import yaml
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from typing import Dict, List
from torch.utils.data import DataLoader
from Calpha.dataset.dataset import CryoEMDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_model(config: Dict, model_path: str) -> None:
    """
    Main function to analyze model performance and generate debug outputs.
    Outputs are saved in checkpoints/.debug directory.
    """
    # Create output directory
    debug_dir = os.path.join(config['training']['checkpoint_dir'], '.debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load model state dictionary
    checkpoint = torch.load(model_path, map_location='cpu')
    
    from Calpha.model.segmentation_model import SegmentationModelResnet
    model = SegmentationModelResnet(config)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the saved weights
    model.eval()
    
    # Load dataset
    dataset = CryoEMDataset(config, mode='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Statistics containers
    stats = {
        'raw_outputs': [],
        'sigmoid_outputs': [],
        'threshold': config['training']['prediction_threshold'],
        'metrics_at_thresholds': []
    }
    
    # Check if cache exists
    if not os.path.exists(config['data']['cache_dir']):
        logging.warning(f"Cache directory {config['data']['cache_dir']} does not exist, skipping analysis")
        return
        
    # Sample a few batches for analysis
    sample_size = min(20, len(loader))
    logging.info(f"Analyzing model on {sample_size} samples")
    
    for i, batch in tqdm(enumerate(loader), total=sample_size, desc="Model analysis"):
        if i >= sample_size:
            break
            
        # Unpack batch according to dataset.py __getitem__ return values
        inputs, hard_labels, soft_labels, _, _ = batch
        with torch.no_grad():
            raw_outputs = model(inputs)
            sigmoid_outputs = torch.sigmoid(raw_outputs)
            
        stats['raw_outputs'].extend(raw_outputs.flatten().tolist())
        stats['sigmoid_outputs'].extend(sigmoid_outputs.flatten().tolist())
    
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for raw, label in zip(stats['sigmoid_outputs'], dataset.hard_labels):
            pred = 1 if raw > threshold else 0
            if label == 1:
                if pred == 1: tp += 1
                else: fn += 1
            else:
                if pred == 1: fp += 1
                else: tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        stats['metrics_at_thresholds'].append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Generate visualizations
    plot_output_distributions(stats, debug_dir)
    plot_metrics_vs_threshold(stats, debug_dir)
    
    # Generate summary report
    generate_model_report(stats, debug_dir)

def plot_output_distributions(stats: Dict, output_dir: str) -> None:
    """Plot distributions of raw and sigmoid outputs."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(stats['raw_outputs'], bins=50)
    plt.title('Raw Model Outputs Distribution')
    plt.xlabel('Output Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(stats['sigmoid_outputs'], bins=50)
    plt.axvline(x=stats['threshold'], color='r', linestyle='--', label=f'Threshold ({stats["threshold"]})')
    plt.title('Sigmoid Outputs Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'output_distributions.png'))
    plt.close()

def plot_metrics_vs_threshold(stats: Dict, output_dir: str) -> None:
    """Plot precision, recall and F1 vs threshold."""
    thresholds = [m['threshold'] for m in stats['metrics_at_thresholds']]
    precisions = [m['precision'] for m in stats['metrics_at_thresholds']]
    recalls = [m['recall'] for m in stats['metrics_at_thresholds']]
    f1s = [m['f1'] for m in stats['metrics_at_thresholds']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.axvline(x=stats['threshold'], color='r', linestyle='--', label=f'Config Threshold ({stats["threshold"]})')
    plt.title('Metrics vs Prediction Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_vs_threshold.png'))
    plt.close()

def generate_model_report(stats: Dict, output_dir: str) -> None:
    """Generate markdown report of model analysis."""
    report_path = os.path.join(output_dir, 'model_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Model Debug Report\n\n")
        f.write("## Output Distributions\n")
        f.write("![Output Distributions](output_distributions.png)\n\n")
        
        f.write("## Metrics vs Threshold\n")
        f.write("![Metrics vs Threshold](metrics_vs_threshold.png)\n\n")
        
        f.write("## Performance at Config Threshold\n")
        config_thresh = next(m for m in stats['metrics_at_thresholds'] 
                           if m['threshold'] == stats['threshold'])
        f.write(f"- **Threshold:** {config_thresh['threshold']:.2f}\n")
        f.write(f"- **Precision:** {config_thresh['precision']:.4f}\n")
        f.write(f"- **Recall:** {config_thresh['recall']:.4f}\n")
        f.write(f"- **F1 Score:** {config_thresh['f1']:.4f}\n")

if __name__ == "__main__":
    config = load_config("/root/project/Calpha/Calpha/config/base.yaml")
    # Get latest checkpoint
    checkpoint_dir = config['training']['checkpoint_dir']
    checkpoint_dir = "/root/project/Calpha/checkpoints_dice_weight2/SMres"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
        analyze_model(config, latest_checkpoint)
    else:
        logging.error("No checkpoints found in directory")
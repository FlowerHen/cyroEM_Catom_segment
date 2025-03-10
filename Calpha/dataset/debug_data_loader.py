"""
This is a debugging script for the CryoEMDataset. It uses the PyTorch DataLoader
to iterate over the dataset in the proper way (using __getitem__) rather than directly
indexing the internal samples array.
"""

import sys
import logging
import torch
from torch.utils.data import DataLoader
import yaml
import os
from .dataset import CryoEMDataset

def setup_logging():
    """Set up logging to file and console with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("debug_data_loader.log")
        ]
    )

def debug_main(config_file):
    setup_logging()
    # Load the YAML config file
    if not os.path.exists(config_file):
        logging.error(f"Config file {config_file} does not exist.")
        sys.exit(1)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    # Create dataset instance in 'train' mode
    dataset = CryoEMDataset(config, mode='train')
    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    for batch_idx, batch in enumerate(loader):
        volume, hard_label, soft_label, voxel_size, global_origin = batch
        logging.info(f"Batch {batch_idx} loaded with shapes:")
        logging.info(f" - Volume: {volume.shape}")
        logging.info(f" - Hard labels: {hard_label.shape}")
        logging.info(f" - Soft labels: {soft_label.shape}")
    logging.info("Finished iterating over the DataLoader.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_data_loader.py config.yaml")
        sys.exit(1)
        
    debug_main(sys.argv[1])
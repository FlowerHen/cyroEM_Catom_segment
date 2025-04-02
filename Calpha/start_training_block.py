import os
import torch
import numpy as np
import logging
from Calpha.dataset.block_dataset import get_block_classifier_data_loaders
from Calpha.model.block_classifier import CalphaBlockClassifier, CalphaBlockTrainer

def get_config():
    """Define configuration directly in code instead of using YAML"""
    config = {
        # Block classifier cache directory
        'data': {
            'root_dir': '/root/autodl-tmp/CryFold/Datasets',
            'cache_dir': '/root/project/Calpha/data_cache_with_neg',
            'split_ratio': {
                'train': 0.8,
                'val': 0.2
            },
            'num_workers': 8,
            'crop_size': 64,
            'overlap': 0.15,
            'd0': 3.0,
            'voxel_size': 1.6638,
            'batch_size': 16,
            'max_samples': 2000,
            'min_ca_atoms': 5,
            'max_empty_ratio': 1,
            'keep_empty_block': True
        },
        'training': {
            'lr': 3e-4,
            'weight_decay': 1e-5,
            'epochs': 100,
            'soft_weight': 0.5,
            'hard_weight': 1.0,
            'hard_dist_weight': 0,
            'dice_weight': 0.5,
            'focal_weight': 0.5,
            'pos_weight': 1.0,
            'checkpoint_dir': '/root/autodl-tmp/CryFold/block_model_checkpoints',
            'device': 'cuda',
            'grad_clip': 1.0,
            'seed': 1,
            'final_div_factor': 100
        },
        'block_classifier': {
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'epochs': 100,
            'batch_size': 32,  # Can use larger batch size for smaller blocks
            'checkpoint_dir': '/root/autodl-tmp/CryFold/block_classifier',
            'device': 'cuda',
            'seed': 1,
            'noise_std': 0.05,
            'grad_clip': 1.0,
            'pos_weight': 2.0,  # Weight for positive samples (blocks with C-alpha)
            'balance_classes': True,  # Balance positive and negative samples
            'max_negative_ratio': 3.0  # Maximum ratio of negative to positive samples
        },
        'augmentation': {
            'rotation': {
                'rotation_samples': 2
            },
            'resolution': {
                'use': true,
                'prob': 0.3,
                'sigma_range': [0.8, 1.0]
            },
            'noise': {
                'prob': 0.3,
                'ratio_range': [0.2, 0.5],
                'std': 0.05
            },
            'intensity_inverpsion': {
                'prob': 0.3
            },
            'gamma_correction': {
                'prob': 0.3,
                'gamma_range': [0.9, 1.1]
            }
        },
        'inference': {
            'overlap': 16,
            'threshold': 0.5,
            'use_block_classifier': True,
            'block_threshold': 0.1,
            'step_size': 64
        }
    }
    return config

def setup_logging(config):
    """Set up logging for the training process"""
    log_dir = config['block_classifier']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'block_classifier.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    # Get configuration directly from code
    config = get_config()
    
    # Set up logging
    setup_logging(config)
    
    # Set random seed for reproducibility
    seed = config['block_classifier']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Log configuration
    logging.info("C-alpha Block Classifier Training Configuration:")
    logging.info(f"Learning Rate: {config['block_classifier']['lr']}")
    logging.info(f"Weight Decay: {config['block_classifier']['weight_decay']}")
    logging.info(f"Epochs: {config['block_classifier']['epochs']}")
    logging.info(f"Batch Size: {config['block_classifier']['batch_size']}")
    logging.info(f"Checkpoint Directory: {config['block_classifier']['checkpoint_dir']}")
    logging.info(f"Device: {config['block_classifier']['device']}")
    
    # Initialize model
    device = torch.device(config['block_classifier']['device'])
    model = CalphaBlockClassifier(input_size=config['data']['crop_size'])
    trainer = CalphaBlockTrainer(model, config, device)
    
    # Check for existing checkpoints
    checkpoint_dir = config['block_classifier']['checkpoint_dir']
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir)
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoints:
            epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
            latest_epoch = max(epochs)
            latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pth')
    
    # Load checkpoint if available
    start_epoch = 1

    # Get data loaders
    train_loader, val_loader = get_block_classifier_data_loaders(config)
    if latest_checkpoint:
        start_epoch = trainer.load_checkpoint(latest_checkpoint)
        logging.info(f"Resuming training from epoch {start_epoch}")

    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    for epoch in range(start_epoch, config['block_classifier']['epochs'] + 1):
        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch(train_loader)
        logging.info(f"Epoch {epoch}/{config['block_classifier']['epochs']} - Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Validate
        if epoch % 5 == 0 or epoch == config['block_classifier']['epochs']:
            val_metrics = trainer.validate(val_loader)
            logging.info(f"Epoch {epoch} - Validation: Loss={val_metrics['loss']:.4f}, Accuracy={val_metrics['accuracy']:.4f}, "
                         f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
            
            # Check if this is the best model
            is_best = False
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                is_best = True
                logging.info(f"New best validation loss: {best_val_loss:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                is_best = True
                logging.info(f"New best validation F1 score: {best_f1:.4f}")
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, is_best=is_best)
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            trainer.save_checkpoint(epoch)
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best validation F1 score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
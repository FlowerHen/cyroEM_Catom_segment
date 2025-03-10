# all_model_test.py
import os
import time
import csv
from copy import deepcopy
from Calpha.dataset.dataset import get_data_loaders
from Calpha.training.trainer import CryoTrainer
from Calpha.model.segmentation_model import (
    SegmentationModelReduced,
    SegmentationModelResnet,
    SegmentationModelAttnReduced,
    SegmentationModel
)
from Calpha.model.hype_para_test import (
    load_config, set_random_seed,
    find_latest_checkpoint, early_stopping_check
)

model_variants = {
    "SM": SegmentationModel,
    "SMreduced": SegmentationModelReduced,
    "SMsimple": SegmentationModelReduced,
    "SMres": SegmentationModelResnet,
    "SMattn": SegmentationModelAttnReduced,
    "SMattnreduced": SegmentationModelAttnReduced
}

def train_model_variant(model_name, model_class, config):
    """Run training for a model variant using shared utilities from hype_para_test."""
    config_exp = deepcopy(config)
    config_exp['training']['checkpoint_dir'] = os.path.join(
        config['training']['checkpoint_dir'],
        model_name
    )
    os.makedirs(config_exp['training']['checkpoint_dir'], exist_ok=True)

    # Initialize model with variant-specific parameters
    model = model_class(use_simam=True)
    trainer = CryoTrainer(model, config_exp)
    
    # Resume logic using shared function
    start_epoch = find_latest_checkpoint(config_exp['training']['checkpoint_dir'])
    if start_epoch > 0:
        trainer.load_checkpoint(os.path.join(
            config_exp['training']['checkpoint_dir'],
            f"checkpoint_epoch_{start_epoch}.pth"
        ))

    # Training setup
    train_loader, val_loader = get_data_loaders(config_exp)
    csv_path = os.path.join(config_exp['training']['checkpoint_dir'], 'training_metrics.csv')
    fieldnames = ['Model', 'Epoch', 'Train Loss', 'Val Loss', 'Dice', 'Precision', 'Cumulative Time (s)']
    
    # Initialize CSV using shared pattern
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Training loop with shared early stopping
    train_ckpt_losses = []
    val_ckpt_losses = []
    cumulative_time = 0.0
    history = []

    for epoch in range(start_epoch + 1, config_exp['training']['epochs'] + 1):
        start_time = time.time()
        train_loss = trainer.train_epoch(train_loader)
        elapsed = time.time() - start_time
        cumulative_time += elapsed

        # Validation and logging
        if epoch == 1 or epoch % 5 == 0:
            val_loss, dice, precision = trainer.validate(val_loader)
            train_ckpt_losses.append(train_loss)
            val_ckpt_losses.append(val_loss)

            # Write metrics
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow({
                    'Model': model_name,
                    'Epoch': epoch,
                    'Train Loss': f"{train_loss:.4f}",
                    'Val Loss': f"{val_loss:.4f}",
                    'Dice': f"{dice:.4f}",
                    'Precision': f"{precision:.4f}",
                    'Cumulative Time (s)': f"{cumulative_time:.2f}"
                })

            # Checkpoint and early stopping
            if epoch != 1:
                trainer.save_checkpoint(epoch)
            stop, reason = early_stopping_check(train_ckpt_losses, val_ckpt_losses)
            if stop:
                print(f"Early stop: {model_name} @ epoch {epoch} - {reason}")
                break

    return history

if __name__ == "__main__":
    config = load_config("/root/project/Calpha/config/base.yaml")
    results = {}
    for name, variant in model_variants.items():
        print(f"\nTraining {name}")
        results[name] = train_model_variant(name, variant, config)
    
    # Print summary
    print("\nTraining Summary:")
    for name, hist in results.items():
        if hist:
            best = min(hist, key=lambda x: x[2])
            print            print(f"{name}: Best Val Loss {best[2]:.4f} @ epoch {best[0]}")

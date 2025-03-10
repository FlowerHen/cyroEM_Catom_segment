import os
import time
import csv
import yaml
import torch
from copy import deepcopy
from Calpha.dataset.dataset import get_data_loaders
from Calpha.training.trainer import CryoTrainer
from Calpha.model.segmentation_model import SegmentationModel, SegmentationModelResnet, SegmentationModelAttn, SegmentationModelMini, SegmentationModelResMini

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# def early_stopping_check(train_ckpt_losses, val_ckpt_losses):
#     """
#     Early stopping triggers if either:
#     1. Training loss shows two consecutive increases (current > previous > prior)
#     2. Validation loss increase exceeds training loss decrease for two consecutive check intervals
#     """
#     stop_reason = ""
#     stop = False

#     # Check training loss condition (requires at least 3 checkpoints)
#     if len(train_ckpt_losses) >= 3:
#         recent_train = train_ckpt_losses[-3:]
#         two_consec_increases = all(recent_train[i+1] > recent_train[i] for i in range(2))
#         if two_consec_increases:
#             stop = True
#             stop_reason = "Training loss increased two consecutive times"

#     # Check validation condition (requires at least 3 checkpoints)
#     if not stop and len(val_ckpt_losses) >= 3 and len(train_ckpt_losses) >= 3:
#         val_deltas = [val_ckpt_losses[i+1] - val_ckpt_losses[i] for i in range(-3, -1)]
#         train_deltas = [train_ckpt_losses[i+1] - train_ckpt_losses[i] for i in range(-3, -1)]
        
#         consecutive_val_failures = all(
#             val_delta > -train_delta 
#             for val_delta, train_delta in zip(val_deltas, train_deltas)
#         )
        
#         if consecutive_val_failures:
#             stop = True
#             stop_reason = "Validation loss rise exceeded training loss drop for two consecutive checks"

#     return stop, stop_reason

def early_stopping_check(train_ckpt_losses, val_ckpt_losses):
    """Early stopping check with new criteria based on validation trend."""
    if len(val_ckpt_losses) < 2:
        return False, ""

    delta_val = val_ckpt_losses[-1] - val_ckpt_losses[-2]
    reasons = []

    if delta_val < 0:
        if len(train_ckpt_losses) >= 3 and \
           train_ckpt_losses[-1] > train_ckpt_losses[-2] > train_ckpt_losses[-3]:
            reasons.append("Training loss increased consecutively for two checkpoints.")
            
        if len(train_ckpt_losses) >= 4 and \
           train_ckpt_losses[-1] > train_ckpt_losses[-4]:
            reasons.append("Current training loss exceeds value from three checkpoints back.")
    else:
        if len(val_ckpt_losses) >= 3 and len(train_ckpt_losses) >= 3:
            val_inc1 = val_ckpt_losses[-1] - val_ckpt_losses[-2]
            val_inc2 = val_ckpt_losses[-2] - val_ckpt_losses[-3]
            train_dec1 = train_ckpt_losses[-2] - train_ckpt_losses[-1]
            train_dec2 = train_ckpt_losses[-3] - train_ckpt_losses[-2]
            
            if val_inc1 > train_dec1 and val_inc2 > train_dec2:
                reasons.append("Validation increases exceed training decreases in consecutive checks.")

        if len(val_ckpt_losses) >= 4 and len(train_ckpt_losses) >= 4:
            total_val_inc = val_ckpt_losses[-1] - val_ckpt_losses[-4]
            total_train_dec = train_ckpt_losses[-4] - train_ckpt_losses[-1]
            
            if total_val_inc >= total_train_dec:
                reasons.append("Total validation increase over three checks exceeds training decrease.")

    return (bool(reasons), " | ".join(reasons)) if reasons else (False, "")

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return 0

    epoch_numbers = []
    for checkpoint in checkpoints:
        try:
            epoch = int(checkpoint.split('_epoch_')[1].split('.')[0])
            epoch_numbers.append(epoch)
        except Exception as e:
            print(f"Warning: cannot extract epoch number from {checkpoint}: {e}")
            continue

    return max(epoch_numbers) if epoch_numbers else 0

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def training_experiment(config, model_class, model_name, noise_std, use_sa):
    """Run training experiment with CSV logging."""
    config_exp = deepcopy(config)
    config_exp['augmentation']['noise_std'] = noise_std
    config_exp['model'] = {'use_sa': use_sa}
    
    seed_value = config_exp['training'].get('seed')
    if seed_value is not None:
        set_random_seed(seed_value)
        print(f"Random seed set to {seed_value}")
    
    model = model_class(use_sa=use_sa)
    exp_name = f"{model_class.__name__}_noise{noise_std}_sa{use_sa}"
    
    base_ckpt_dir = config_exp['training'].get('base_checkpoint_dir', config_exp['training']['checkpoint_dir'])
    exp_checkpoint_dir = os.path.join(base_ckpt_dir, exp_name)
    config_exp['training']['checkpoint_dir'] = exp_checkpoint_dir
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    # CSV logging setup
    csv_path = os.path.join(exp_checkpoint_dir, 'training_metrics.csv')
    cumulative_time = 0.0
    
    # Handle resume case
    start_epoch = find_latest_checkpoint(exp_checkpoint_dir)
    if start_epoch > 0 and os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader]
                if rows:
                    last_row = rows[-1]
                    cumulative_time = float(last_row['Cumulative Time (s)'])
        except Exception as e:
            print(f"Warning: Error reading CSV: {e}")

    # Initialize CSV file
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Model', 'Noise Intensity', 'SA', 'Epoch', 
                'Train Loss', 'Val Loss', 'Dice', 'Precision', 
                'Cumulative Time (s)'
            ])
            writer.writeheader()

    trainer = CryoTrainer(model, config_exp)
    if start_epoch > 0:
        checkpoint_path = os.path.join(exp_checkpoint_dir, f"checkpoint_epoch_{start_epoch}.pth")
        trainer.load_checkpoint(checkpoint_path)

    train_loader, val_loader = get_data_loaders(config_exp)

    train_ckpt_losses = []
    val_ckpt_losses = []
    total_epochs = config_exp['training']['epochs']
    
    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time()
        train_loss = trainer.train_epoch(train_loader)
        elapsed = time.time() - start
        cumulative_time += elapsed
        msg = f"Epoch {epoch}: Train Loss {train_loss:.4f} | Time: {elapsed:.2f}s"
        print(msg)
        trainer.logger.info(msg)

        if epoch == 1 or epoch % 5 == 0:
            val_loss, dice, precision = trainer.validate(val_loader)
            train_ckpt_losses.append(train_loss)
            val_ckpt_losses.append(val_loss)
            
            # Write to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'Model', 'Noise Intensity', 'SA', 'Epoch', 
                    'Train Loss', 'Val Loss', 'Dice', 'Precision', 
                    'Cumulative Time (s)'
                ])
                writer.writerow({
                    'Model': model_name,
                    'Noise Intensity': noise_std,
                    'SA': use_sa,
                    'Epoch': epoch,
                    'Train Loss': f"{train_loss:.4f}",
                    'Val Loss': f"{val_loss:.4f}",
                    'Dice': f"{dice:.4f}",
                    'Precision': f"{precision:.4f}",
                    'Cumulative Time (s)': f"{cumulative_time:.2f}"
                })

            msg = f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Time: {elapsed:.2f}s"
            print(msg)
            trainer.logger.info(msg)
            if epoch != 1:
                trainer.save_checkpoint(epoch)

            early_stop, stop_reason = early_stopping_check(train_ckpt_losses, val_ckpt_losses)
            if early_stop:
                msg = f"Early stopping triggered at epoch {epoch} due to: {stop_reason}"
                print(msg)
                trainer.logger.info(msg)
                break

    print("Training finished.")

if __name__ == "__main__":
    config = load_config("/root/project/Calpha/Calpha/config/para_trial2.yaml")
    noise_std_values = [0.05, 0.1]
    for noise_std in noise_std_values:
        for (model_class, model_name) in [(SegmentationModelMini, "SMmini"),
                                          (SegmentationModelAttn, "SMA"),
                                          (SegmentationModelResnet, "SMR"),
                                          (SegmentationModel, "SM")]:
            for use_sa in [True, False]:
                if model_class == SegmentationModelAttn and use_sa:
                    continue
                print(f"Running experiment with {model_name}, noise_std={noise_std}, sa={use_sa}")
                training_experiment(config, model_class, model_name, noise_std=noise_std, use_sa=use_sa)

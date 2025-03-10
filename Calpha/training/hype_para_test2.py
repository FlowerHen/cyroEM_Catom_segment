import os
import time
import csv
import yaml
import torch
from copy import deepcopy
from Calpha.dataset.dataset import get_data_loaders
from Calpha.training.trainer import CryoTrainer
from Calpha.model.segmentation_model import SegmentationModelMini, SegmentationModelResMini

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def early_stopping_check(train_ckpt_losses, val_ckpt_losses):
    """Simplified early stopping check."""
    if len(val_ckpt_losses) < 3 or len(train_ckpt_losses) < 3:
        return False, ""

    # Condition 1: Val loss increased twice consecutively
    val_inc_twice = val_ckpt_losses[-1] > val_ckpt_losses[-2] > val_ckpt_losses[-3]
    if val_inc_twice:
        return True, "Validation loss increased consecutively twice"

    # Condition 2: Train loss increased twice consecutively
    train_inc_twice = train_ckpt_losses[-1] > train_ckpt_losses[-2] > train_ckpt_losses[-3]
    if train_inc_twice:
        return True, "Training loss increased consecutively twice"

    # Condition 3: Val loss increase exceeds train loss decrease
    val_change = val_ckpt_losses[-1] - val_ckpt_losses[-2]
    train_change = train_ckpt_losses[-1] - train_ckpt_losses[-2]
    if val_change > -train_change and val_change > 0:
        return True, "Validation loss increase exceeds training loss decrease"

    return False, ""

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return 0
    epoch_numbers = [int(f.split('_epoch_')[1].split('.')[0]) for f in checkpoints]
    return max(epoch_numbers)

def delete_old_checkpoints(checkpoint_dir, current_epoch):
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split('_epoch_')[1].split('.')[0])
        if 5 <= epoch <= (current_epoch - 10):
            file_path = os.path.join(checkpoint_dir, checkpoint)
            os.remove(file_path)
            print(f"Deleted old checkpoint: {file_path}")

def set_random_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def training_experiment(config, model_class, model_name, lr, weight_decay):
    config_exp = deepcopy(config)
    config_exp['training']['lr'] = lr
    config_exp['training']['weight_decay'] = weight_decay
    
    seed_value = config_exp['training'].get('seed')
    if seed_value is not None:
        set_random_seed(seed_value)
        print(f"Random seed set to {seed_value}")
    
    model = model_class()
    exp_name = f"{model_name}_lr{lr:.0e}_wd{weight_decay:.0e}"
    
    base_ckpt_dir = config_exp['training'].get('base_checkpoint_dir', 
                                             config_exp['training']['checkpoint_dir'])
    exp_checkpoint_dir = os.path.join(base_ckpt_dir, exp_name)
    config_exp['training']['checkpoint_dir'] = exp_checkpoint_dir
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(exp_checkpoint_dir, 'training_metrics.csv')
    cumulative_time = 0.0
    train_ckpt_losses = []
    val_ckpt_losses = []
    
    # Resume handling with detailed logging
    start_epoch = find_latest_checkpoint(exp_checkpoint_dir)
    if start_epoch > 0 and os.path.exists(csv_path):
        print(f"\n{'-'*40}")
        print(f"Resuming training from epoch {start_epoch}")
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    print(f"Found {len(rows)} previous records in CSV")
                    for row in rows:
                        epoch = int(row['Epoch'])
                        if epoch % 5 == 0 or epoch == 1:
                            train_ckpt_losses.append(float(row['Train Loss']))
                            val_ckpt_losses.append(float(row['Val Loss']))
                    cumulative_time = float(rows[-1]['Cumulative Time (s)'])
                    print(f"Restored cumulative time: {cumulative_time:.2f}s")
                    print(f"Restored train losses: {train_ckpt_losses}")
                    print(f"Restored val losses: {val_ckpt_losses}")
                    # Immediate early stopping check after restoration
                    early_stop, reason = early_stopping_check(train_ckpt_losses, val_ckpt_losses)
                    if early_stop:
                        print(f"Early stopping triggered after restoration: {reason}")
                        return
        except Exception as e:
            print(f"Warning: Error reading CSV: {e}")
        print(f"{'-'*40}\n")
    else:
        print("Starting new training session")

    # Initialize CSV if not exists
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Model', 'Learning Rate', 'Weight Decay', 'Epoch', 
                'Train Loss', 'Val Loss', 'Dice', 'Precision', 
                'Cumulative Time (s)'
            ])
            writer.writeheader()

    trainer = CryoTrainer(model, config_exp)
    if start_epoch > 0:
        checkpoint_path = os.path.join(exp_checkpoint_dir, f"checkpoint_epoch_{start_epoch}.pth")
        trainer.load_checkpoint(checkpoint_path)

    train_loader, val_loader = get_data_loaders(config_exp)
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
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'Model', 'Learning Rate', 'Weight Decay', 'Epoch', 
                    'Train Loss', 'Val Loss', 'Dice', 'Precision', 
                    'Cumulative Time (s)'
                ])
                writer.writerow({
                    'Model': model_name,
                    'Learning Rate': f"{lr:.0e}",
                    'Weight Decay': f"{weight_decay:.0e}",
                    'Epoch': epoch,
                    'Train Loss': f"{train_loss:.6f}",
                    'Val Loss': f"{val_loss:.6f}",
                    'Dice': f"{dice:.4f}",
                    'Precision': f"{precision:.4f}",
                    'Cumulative Time (s)': f"{cumulative_time:.2f}"
                })

            msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s"
            print(msg)
            trainer.logger.info(msg)
            if epoch != 1:
                trainer.save_checkpoint(epoch)
                delete_old_checkpoints(exp_checkpoint_dir, epoch)

            early_stop, stop_reason = early_stopping_check(train_ckpt_losses, val_ckpt_losses)
            if early_stop:
                msg = f"Early stopping triggered at epoch {epoch} due to: {stop_reason}"
                print(msg)
                trainer.logger.info(msg)
                break

    print("Training finished.")

if __name__ == "__main__":
    config = load_config("/root/project/Calpha/Calpha/config/para_trial2.yaml")
    lr_values = [5e-4, 3e-4] 
    wd_values = [1e-3, 1e-4, 1e-5]
    
    for model_class, model_name in [(SegmentationModelResMini, "SMRmini"),
                                    (SegmentationModelMini, "SMmini")]:
        for lr in lr_values:
            for wd in wd_values:
                print(f"\n{'='*40}")
                print(f"Starting experiment: {model_name} | LR: {lr:.0e} | WD: {wd:.0e}")
                print(f"{'='*40}")
                training_experiment(config, model_class, model_name, lr=lr, weight_decay=wd)
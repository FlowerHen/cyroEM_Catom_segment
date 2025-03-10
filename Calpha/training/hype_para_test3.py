import os
import time
import csv
import yaml
import torch
from copy import deepcopy
from Calpha.dataset.dataset import get_data_loaders
from Calpha.training.trainer import CryoTrainer
from Calpha.model.segmentation_model_dropout import (
    SegmentationModel,
    SegmentationModelResnet,
    SegmentationModelAttn,
    SegmentationModelMini,
    SegmentationModelResMini,
    SegmentationAttnMini
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def early_stopping_check(train_ckpt_losses, val_ckpt_losses):
    if len(val_ckpt_losses) < 3 or len(train_ckpt_losses) < 3:
        return False, ""
    
    # Condition 1: Validation loss increases consecutively twice
    if val_ckpt_losses[-1] > val_ckpt_losses[-2] > val_ckpt_losses[-3]:
        return True, "Validation loss increased consecutively twice"
    
    # Condition 2: Training loss increases consecutively twice
    if train_ckpt_losses[-1] > train_ckpt_losses[-2] > train_ckpt_losses[-3]:
        return True, "Training loss increased consecutively twice"
    
    # Condition 3: Validation loss jump exceeds training loss decrease
    val_change = val_ckpt_losses[-1] - val_ckpt_losses[-2]
    train_change = train_ckpt_losses[-1] - train_ckpt_losses[-2]
    if val_change > -train_change and val_change > 0:
        return True, "Validation loss increase exceeds training loss decrease"
    
    return False, ""

def early_stopping_history_check(train_ckpt_losses, val_ckpt_losses):
    """
    Iteratively check the complete history of checkpointed losses for early stopping conditions.
    This function is used when restoring training from saved checkpoints.
    """
    if len(val_ckpt_losses) < 3 or len(train_ckpt_losses) < 3:
        return False, ""
    
    # Iterate over every consecutive triple of records in history.
    for i in range(2, len(val_ckpt_losses)):
        # Check validation loss consecutive increase
        if val_ckpt_losses[i] > val_ckpt_losses[i-1] > val_ckpt_losses[i-2]:
            return True, f"Validation loss increased consecutively at records {i-2} to {i}"
        # Check training loss consecutive increase
        if train_ckpt_losses[i] > train_ckpt_losses[i-1] > train_ckpt_losses[i-2]:
            return True, f"Training loss increased consecutively at records {i-2} to {i}"
        # Check if validation loss increase exceeds training loss decrease
        val_change = val_ckpt_losses[i] - val_ckpt_losses[i-1]
        train_change = train_ckpt_losses[i] - train_ckpt_losses[i-1]
        if val_change > -train_change and val_change > 0:
            return True, f"Validation loss increase exceeds training loss decrease at records {i-1} to {i}"
    
    return False, ""

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
    ]
    if not checkpoints:
        return 0
    epoch_numbers = [int(f.split('_epoch_')[1].split('.')[0]) for f in checkpoints]
    return max(epoch_numbers) if epoch_numbers else 0

def delete_old_checkpoints(checkpoint_dir, current_epoch):
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
    ]
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

def training_experiment(config, model_class, model_name, dropout_rate, final_div_factor):
    """
    This function supports restoration from checkpoints.
    """
    config_exp = deepcopy(config)
    config_exp['model'] = {'dropout_rate': dropout_rate}
    config_exp['optimizer'] = config_exp.get('optimizer', {})
    config_exp['optimizer']['final_div_factor'] = final_div_factor
    
    seed_value = config_exp['training'].get('seed')
    if seed_value is not None:
        set_random_seed(seed_value)
        print(f"Random seed set to {seed_value}")
    
    model = model_class(dropout_rate=dropout_rate)
    exp_name = f"{model_name}_dropout{dropout_rate}_divfactor{final_div_factor}"
    
    base_ckpt_dir = config_exp['training'].get(
        'base_checkpoint_dir', config_exp['training']['checkpoint_dir']
    )
    exp_checkpoint_dir = os.path.join(base_ckpt_dir, exp_name)
    config_exp['training']['checkpoint_dir'] = exp_checkpoint_dir
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(exp_checkpoint_dir, 'training_metrics.csv')
    cumulative_time = 0.0
    train_ckpt_losses = []
    val_ckpt_losses = []
    
    # Enhanced resume handling from test2
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
                        # Append only those records that were checkpointed (epoch == 1 or every 5 epochs)
                        if epoch % 5 == 0 or epoch == 1:
                            train_ckpt_losses.append(float(row['Train Loss']))
                            val_ckpt_losses.append(float(row['Val Loss']))
                    cumulative_time = float(rows[-1]['Cumulative Time (s)'])
                    print(f"Restored cumulative time: {cumulative_time:.2f}s")
                    
                    # For restoration: check the entire history for early stopping conditions.
                    early_stop, stop_reason = early_stopping_history_check(train_ckpt_losses, val_ckpt_losses)
                    if early_stop:
                        print(f"Early stopping triggered upon restoration: {stop_reason}")
                        return
        except Exception as e:
            print(f"Warning: Error reading CSV: {e}")
        print(f"{'-'*40}\n")
    else:
        print("Starting new training session")

    # Initialize CSV file if not present.
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'Model', 'Dropout Rate', 'Final Div Factor', 'Epoch',
                    'Train Loss', 'Val Loss', 'Dice', 'Precision',
                    'Cumulative Time (s)'
                ]
            )
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
        msg = f"[{exp_name}] Epoch {epoch}: Train Loss {train_loss:.4f} | Time: {elapsed:.2f}s"
        print(msg)
        trainer.logger.info(msg)

        if epoch == 1 or epoch % 5 == 0:
            val_loss, dice, precision = trainer.validate(val_loader)
            train_ckpt_losses.append(train_loss)
            val_ckpt_losses.append(val_loss)
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'Model', 'Dropout Rate', 'Final Div Factor', 'Epoch',
                        'Train Loss', 'Val Loss', 'Dice', 'Precision',
                        'Cumulative Time (s)'
                    ]
                )
                writer.writerow({
                    'Model': model_name,
                    'Dropout Rate': dropout_rate,
                    'Final Div Factor': final_div_factor,
                    'Epoch': epoch,
                    'Train Loss': f"{train_loss:.4f}",
                    'Val Loss': f"{val_loss:.4f}",
                    'Dice': f"{dice:.4f}",
                    'Precision': f"{precision:.4f}",
                    'Cumulative Time (s)': f"{cumulative_time:.2f}"
                })

            msg = f"[{exp_name}] Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s"
            print(msg)
            trainer.logger.info(msg)
            if epoch != 1:
                trainer.save_checkpoint(epoch)
                delete_old_checkpoints(exp_checkpoint_dir, epoch)

            # Normal training early stopping check (only last three entries)
            early_stop, stop_reason = early_stopping_check(train_ckpt_losses, val_ckpt_losses)
            if early_stop:
                msg = f"[{exp_name}] Early stopping triggered at epoch {epoch} due to: {stop_reason}"
                print(msg)
                trainer.logger.info(msg)
                break

    print(f"[{exp_name}] Training finished.")
    torch.cuda.empty_cache()
    return f"Training completed for {exp_name}"

def run_training():
    config = load_config("/root/project/Calpha/Calpha/config/base.yaml")
    dropout_rates = [0.0, 0.2, 0.4]
    final_div_factors = [100, 10000]
    model_classes = [
        (SegmentationAttnMini, "SMAmini"),
        (SegmentationModelMini, "SMmini"),
        (SegmentationModelResMini, "SMRmini")
    ]

    for final_div_factor in final_div_factors:
        for model_class, model_name in model_classes:
            for dropout_rate in dropout_rates:
                print(f"\n{'='*40}")
                print(f"Starting experiment: {model_name} | Dropout: {dropout_rate} | Div Factor: {final_div_factor}")
                print(f"{'='*40}")
                
                result = training_experiment(
                    config,
                    model_class,
                    model_name,
                    dropout_rate,
                    final_div_factor
                )
                print(f"Completed: {result}")

if __name__ == "__main__":
    run_training()

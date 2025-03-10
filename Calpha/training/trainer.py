import os
import time
import torch
import logging
from tqdm import tqdm

class CryoTrainer:
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device(self.config['training']['device'])
        self.model = model.to(self.device)
        self.lr = float(self.config['training']['lr'])
        weight_decay = float(self.config['training']['weight_decay'])
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(device=self.device.type)
        
        os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
        
        # Create a unique logger for each experiment by including the checkpoint_dir in the name.
        logger_name = f"CryoTrainer_{os.path.basename(os.path.normpath(self.config['training']['checkpoint_dir']))}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        # Clear old handlers if any (prevents logs going to the wrong file)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        log_file = os.path.join(self.config['training']['checkpoint_dir'], "training.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Enable anomaly detection to help locate NaN issues
        torch.autograd.set_detect_anomaly(True)

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch using the provided training data loader.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            inputs, hard_labels, soft_labels = batch[0], batch[1], batch[2]
            
            # Check each sample in the batch: skip if std is too low or contains NaN
            valid_indices = []
            batch_std = inputs.view(inputs.size(0), -1).std(dim=1)
            for i in range(inputs.size(0)):
                sample_std = batch_std[i].item()
                if sample_std < 1e-6:
                    self.logger.warning(f"Training: Batch {batch_idx} sample {i} skipped (std too low: {sample_std:.2e})")
                elif torch.isnan(inputs[i]).any():
                    self.logger.warning(f"Training: Batch {batch_idx} sample {i} skipped (contains NaN)")
                else:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                self.logger.warning(f"Training: No valid samples in batch {batch_idx}, skipping the batch.")
                continue
            
            inputs_valid = inputs[valid_indices].to(self.device, non_blocking=True)
            hard_labels_valid = hard_labels[valid_indices].to(self.device, non_blocking=True)
            soft_labels_valid = soft_labels[valid_indices].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Use autocast for mixed precision forward pass
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(inputs_valid)
                tensor_stats = {
                    "min": outputs.min().item(),
                    "max": outputs.max().item(),
                    "mean": outputs.mean().item()
                }
                self.logger.debug(f"Batch {batch_idx} output tensor stats: {tensor_stats}")
                loss = self._compute_loss(outputs, hard_labels_valid, soft_labels_valid)
            
            self.scaler.scale(loss).backward()
            # Unscale before clipping gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Scheduler step: ensure optimizer.step() happens before scheduler.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        self.logger.info(f"Epoch completed in {epoch_time:.2f} seconds, Average Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_samples = 0
        eps = 1e-6
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                inputs, hard_labels, soft_labels = batch[0], batch[1], batch[2]
                
                # Check each sample: skip if std is too low or contains NaN
                valid_indices = []
                batch_std = inputs.view(inputs.size(0), -1).std(dim=1)
                for i in range(inputs.size(0)):
                    sample_std = batch_std[i].item()
                    if sample_std < eps:
                        self.logger.warning(f"Validation: Batch {batch_idx} sample {i} skipped (std too low: {sample_std:.2e})")
                    elif torch.isnan(inputs[i]).any():
                        self.logger.warning(f"Validation: Batch {batch_idx} sample {i} skipped (contains NaN)")
                    else:
                        valid_indices.append(i)
                if len(valid_indices) == 0:
                    self.logger.warning(f"Validation: No valid samples in batch {batch_idx}, skipping the batch.")
                    continue
    
                inputs_valid = inputs[valid_indices].to(self.device, non_blocking=True)
                hard_labels_valid = hard_labels[valid_indices].to(self.device, non_blocking=True)
                soft_labels_valid = soft_labels[valid_indices].to(self.device, non_blocking=True)
                
                outputs = self.model(inputs_valid)
                loss = self._compute_loss(outputs, hard_labels_valid, soft_labels_valid)
                
                batch_size = inputs_valid.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
    
                y_prob = torch.sigmoid(outputs)
                y_pred = (y_prob > 0.5).float()
                
                y_pred = y_pred.squeeze(1)
                truth = hard_labels_valid.squeeze(1)
    
                batch_dice = 0.0
                batch_precision = 0.0
                for i in range(y_pred.size(0)):
                    pred_i = y_pred[i]
                    true_i = truth[i]
                    intersection = (pred_i * true_i).sum()
                    dice_i = (2. * intersection + eps) / (pred_i.sum() + true_i.sum() + eps)
                    precision_i = (intersection + eps) / (pred_i.sum() + eps)
                    batch_dice += dice_i
                    batch_precision += precision_i
                batch_dice = batch_dice / y_pred.size(0)
                batch_precision = batch_precision / y_pred.size(0)
                
                total_dice += batch_dice.item() * batch_size
                total_precision += batch_precision.item() * batch_size
    
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        avg_dice = total_dice / total_samples if total_samples > 0 else float('nan')
        avg_precision = total_precision / total_samples if total_samples > 0 else float('nan')
    
        self.logger.info(f"Validation - Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | Precision_hard: {avg_precision:.4f}")
        return avg_loss, avg_dice, avg_precision

    def _compute_loss(self, pred, hard, soft):
        import torch.nn.functional as F
    
        ce_loss = F.binary_cross_entropy_with_logits(pred, hard)
        # Temporary Debugging Section: Print statistics of soft_labels
        soft_stats = {
            "min": soft.min().item(),
            "max": soft.max().item(),
            "mean": soft.mean().item()
        }
        if soft_stats["min"] < 0 or soft_stats["max"] > 1:
            print("DEBUG: Soft labels statistics:", soft_stats)
        probs = torch.sigmoid(pred.float())
        with torch.amp.autocast(enabled=False, device_type=self.device.type):
            mse_loss = F.mse_loss(probs.to(torch.float32), soft.to(torch.float32))
        return ce_loss + self.config['training']['soft_weight'] * mse_loss

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else {},
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch

    def train(self, train_loader, val_loader=None, resume_epoch=1, seed=None):
        """
        The main training loop. Sets the random seed (if provided), initializes the scheduler with a
        low final_div_factor according to the configuration, and then executes the training loop.
        """
        if seed is not None:
            # Fix randomness for reproducibility of data splits and training process.
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            # Ensure deterministic behavior for cudnn (at the expense of performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize the scheduler if it is not already set, using a low final_div_factor from config.
        if self.scheduler is None:
            total_steps = int(self.config['training']['epochs'] * len(train_loader))
            final_div_factor = float(self.config['training'].get('final_div_factor', 1e4))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                final_div_factor=final_div_factor
            )
        
        for epoch in range(resume_epoch, self.config['training']['epochs'] + 1):
            train_loss = self.train_epoch(train_loader)
            msg = f"Epoch {epoch}/{self.config['training']['epochs']} - Train Loss: {train_loss:.4f}"
            
            if val_loader is not None and (epoch == resume_epoch or epoch % 5 == 0):
                val_loss, val_dice, val_precision = self.validate(val_loader)
                msg += f", Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | Precision_hard: {val_precision:.4f}"
            print(msg)
            self.logger.info(msg)
            
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
                
    def lr_forward_search(self, train_loader, start_lr=1e-7, end_lr=1, num_steps=100, beta=0.98):
        """
        Perform a learning rate forward search (LR Finder) to identify a promising learning rate range.
        This function gradually increases the learning rate from start_lr to end_lr over a specified
        number of steps, tracking the training loss in an exponential moving average fashion (controlled
        by beta). 
        """
        # Save the original model and optimizer state for later restoration.
        original_state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        self.model.train()
        optimizer = self.optimizer

        # Initialize the learning rate and compute the multiplier factor.
        lr = start_lr
        mult = (end_lr / start_lr) ** (1 / num_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        lrs = []
        losses = []
        avg_loss = 0.0
        best_loss = float('inf')
        batch_iter = iter(train_loader)

        # Loop over a fixed number of steps to update the learning rate and track loss.
        for i in range(num_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break  # End of data loader reached early
            inputs, hard_labels, soft_labels = batch[0], batch[1], batch[2]
            inputs = inputs.to(self.device)
            hard_labels = hard_labels.to(self.device)
            soft_labels = soft_labels.to(self.device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, hard_labels, soft_labels)
            
            # Update the smoothed loss and record it
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (i + 1))
            lrs.append(lr)
            losses.append(smoothed_loss)

            # If the loss diverges dramatically, quit early.
            if i > 0 and smoothed_loss > 4 * best_loss:
                break

            if smoothed_loss < best_loss or i == 0:
                best_loss = smoothed_loss

            # Backpropagate and update parameters (these updates are temporary).
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # Increase the learning rate exponentially for the next iteration.
            lr *= mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Restore the original model and optimizer state to avoid disrupting ongoing training.
        self.model.load_state_dict(original_state['model_state'])
        self.optimizer.load_state_dict(original_state['optimizer_state'])
        
        return lrs, losses

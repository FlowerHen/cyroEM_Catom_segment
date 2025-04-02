import os
import time
import torch
import logging
import csv
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random
from . import visualization

class CryoTrainer:
    def __init__(self, model, config):
        """Initialize the CryoTrainer with model and configuration."""
        self.config = config
        self.initial_checkpoint_path = self.config['training'].get('initial_checkpoint_path')
        self.device = torch.device(self.config['training']['device'])
        self.model = model.to(self.device)

        logger_name = f"CryoTrainer_{os.path.basename(os.path.normpath(self.config['training']['checkpoint_dir']))}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            log_file = os.path.join(self.config['training']['checkpoint_dir'], "training.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            if self.config['training'].get('enhanced_logging', False):
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

        self.lr = float(self.config['training']['lr'])
        weight_decay = float(self.config['training']['weight_decay'])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay
        )

        self.scheduler = None
        self.amp_enabled = self.config['training'].get('amp', True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)

        os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)

        torch.autograd.set_detect_anomaly(self.config['training'].get('detect_anomaly', False))

        self.enable_csv_logging = self.config['training'].get('csv_logging', False)
        if self.enable_csv_logging:
            self.csv_path = os.path.join(self.config['training']['checkpoint_dir'], 'training_metrics.csv')
            self.fieldnames = [
                'Epoch', 'Cumulative Time (s)',
                'Train Loss', 'Train CE', 'Train MSE', 'Train Hard Dist', 'Train Dice', 'Train Focal',
                'Train Precision', 'Train Recall', 'Train F1', 'Train IoU',
                'Val Loss', 'Val CE', 'Val MSE', 'Val Hard Dist', 'Val Dice', 'Val Focal',
                'Val Precision', 'Val Recall', 'Val F1', 'Val IoU'
            ]
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_val_dice_score = 0.0
        self.best_epoch = 0
        self.cumulative_time = 0.0
        self.nan_error_count = 0
        self.last_validation_metrics = None

        training_config = self.config['training']
        self.enable_early_stopping = training_config.get('early_stopping', False)
        if self.enable_early_stopping:
            self.early_stopping_patience = training_config.get('early_stopping_patience', 3)
            self.early_stopping_delta = max(training_config.get('early_stopping_delta', 0.0), 1e-4)

        self.delete_old_checkpoints = training_config.get('delete_old_checkpoints', False)
        if self.delete_old_checkpoints:
            self.keep_checkpoint_every = training_config.get('keep_checkpoint_every', 10)

        self.validation_interval = training_config.get('validation_interval', 5)
        self.checkpoint_interval = training_config.get('checkpoint_interval', 5)
        self.current_losses = {}

        self.enable_validation_visualization = training_config.get('enable_validation_visualization', False)
        self.viz_count = training_config.get('visualization_count', 5)  # Number of visualizations to generate

        if self.enable_validation_visualization:
            if visualization is None:
                self.logger.warning("Visualization enabled in config, but 'visualization' module not imported/available. Disabling.")
                self.enable_validation_visualization = False
            else:
                self.visualization_dir = os.path.join(self.config['training']['checkpoint_dir'], 'validation_visualizations')
                os.makedirs(self.visualization_dir, exist_ok=True)

        self.prediction_threshold = self.config['training'].get('prediction_threshold', 0.5)
        self.grad_clip_value = self.config['training'].get('grad_clip', 1.0)

    def _compute_performance_metrics(self, y_pred_label, truth_hard, eps=1e-6):
        """Compute performance metrics using thresholded predictions."""
        intersection = (y_pred_label * truth_hard).sum(dim=(1, 2, 3))
        pred_sum = y_pred_label.sum(dim=(1, 2, 3))
        true_sum = truth_hard.sum(dim=(1, 2, 3))

        precision = (intersection + eps) / (pred_sum + eps)
        recall = (intersection + eps) / (true_sum + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        union = pred_sum + true_sum - intersection
        iou = (intersection + eps) / (union + eps)

        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'iou': iou.mean().item()
        }

    def train_epoch(self, train_loader):
        """Train the model for one epoch with gradient accumulation."""
        self.model.train()
        accumulation_steps = self.config['training'].get('accumulation_steps', 1)
        total_metrics = {
            'loss': 0.0, 'ce': 0.0, 'mse': 0.0, 'hard_dist': 0.0, 'dice': 0.0, 'focal': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0
        }
        total_samples = 0
        valid_batches = 0
        eps = 1e-6
        metric_keys = list(total_metrics.keys())

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            if not isinstance(batch, (list, tuple)) or len(batch) < 3:
                self.logger.warning(f"Skipping malformed batch {batch_idx}")
                continue
            inputs, hard_labels, soft_labels = batch[0], batch[1], batch[2]

            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                self.logger.warning(f"Input NaN/Inf detected in batch {batch_idx}, skipping.")
                continue

            valid_indices = [i for i in range(inputs.size(0)) if inputs[i].view(-1).std(unbiased=False) >= eps]
            if not valid_indices:
                continue

            inputs_valid = inputs[valid_indices].to(self.device, non_blocking=True)
            hard_labels_valid = hard_labels[valid_indices].to(self.device, non_blocking=True)
            soft_labels_valid = soft_labels[valid_indices].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                pred = self.model(inputs_valid)
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    self.logger.warning(f"Output NaN/Inf detected in batch {batch_idx}, skipping.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                loss = self._compute_loss(pred, hard_labels_valid, soft_labels_valid)
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Loss is NaN/Inf in batch {batch_idx}, skipping.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                scaled_loss = loss / accumulation_steps

            self.scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if not hasattr(self, '_grads_unscaled'):
                    self.scaler.unscale_(self.optimizer)
                    self._grads_unscaled = True
                
                nan_found = any(param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                              for param in self.model.parameters())

                if nan_found:
                    self.nan_error_count += 1
                    self.logger.warning(f"NaN/Inf in gradients at batch {batch_idx}; skipping step. (Total: {self.nan_error_count})")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    delattr(self, '_grads_unscaled')
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                delattr(self, '_grads_unscaled')

            if self.scheduler and not nan_found and (batch_idx + 1) % accumulation_steps == 0:
                 self.scheduler.step()

            if not nan_found and not (torch.isnan(loss) or torch.isinf(loss)):
                 with torch.no_grad():
                    prob = torch.sigmoid(pred.float())
                    pred_label = (prob > self.prediction_threshold).float()
                    truth_hard = hard_labels_valid.float()
                    if pred_label.dim() == truth_hard.dim() - 1:
                         truth_hard = truth_hard.squeeze(1)
                    elif pred_label.dim() == truth_hard.dim() and pred_label.size(1) == 1:
                         pred_label = pred_label.squeeze(1)
                         truth_hard = truth_hard.squeeze(1)

                    perf_metrics = self._compute_performance_metrics(pred_label, truth_hard, eps)

                 batch_size = inputs_valid.size(0)
                 total_samples += batch_size
                 valid_batches += 1

                 total_metrics['loss'] += loss.item() * batch_size
                 for loss_name, loss_value in self.current_losses.items():
                     if loss_name != 'loss' and loss_name in total_metrics:
                         total_metrics[loss_name] += loss_value * batch_size
                 for metric_name, metric_value in perf_metrics.items():
                     if metric_name in total_metrics:
                         total_metrics[metric_name] += metric_value * batch_size

                 pbar.set_postfix({
                     'Loss': f"{loss.item():.4f}",
                     'CE': f"{self.current_losses.get('ce', 0):.4f}",
                     'Dice': f"{self.current_losses.get('dice', 0):.4f}",
                     'F1': f"{perf_metrics.get('f1', 0):.4f}",
                     'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                 })

        if len(train_loader) % accumulation_steps != 0:
            try:
                self.scaler.unscale_(self.optimizer)
                nan_found = any(param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                                for param in self.model.parameters())
                if not nan_found:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                     self.logger.warning(f"NaN/Inf in gradients during final accumulation step; skipping.")
            except Exception as e:
                 self.logger.error(f"Error during final accumulation step: {e}")
            finally:
                 self.optimizer.zero_grad(set_to_none=True)

        if valid_batches == 0 or total_samples == 0:
            self.logger.warning("No valid batches processed in training epoch.")
            return {metric: float('nan') for metric in metric_keys}

        avg_metrics = {metric: total_metrics[metric] / total_samples for metric in metric_keys}
        for loss_comp in ['ce', 'mse', 'hard_dist', 'dice', 'focal']:
             if loss_comp in avg_metrics:
                  avg_metrics[loss_comp] = total_metrics[loss_comp] / total_samples

        avg_metrics['loss'] = total_metrics['loss'] / total_samples

        self.logger.info(f"Train Epoch Summary - Samples: {total_samples}, "
                         f"Avg Loss: {avg_metrics['loss']:.4f}, CE: {avg_metrics['ce']:.4f}, "
                         f"Dice: {avg_metrics['dice']:.4f}, Focal: {avg_metrics['focal']:.4f}, "
                         f"MSE: {avg_metrics['mse']:.4f}, "
                         f"F1: {avg_metrics['f1']:.4f}, Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}")
        return avg_metrics

    def validate(self, val_loader, epoch=None, visualize=False):
        """Validate the model on the validation set."""
        self.model.eval()
        total_metrics = {
            'loss': 0.0, 'ce': 0.0, 'mse': 0.0, 'hard_dist': 0.0, 'dice': 0.0, 'focal': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0
        }
        total_samples = 0
        valid_batches = 0
        eps = 1e-6
        metric_keys = list(total_metrics.keys())

        all_hard_labels_viz = []
        all_outputs_viz = []

        # Initialize lists for advanced plots if enabled and visualization is requested
        if visualize and self.config['training'].get('enable_advanced_plots', False):
            std_diffs = []
            proportion_positives = []
            f1_scores = []
            fn_soft_labels = []
            fp_soft_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if not isinstance(batch, (list, tuple)) or len(batch) < 3:
                    self.logger.warning(f"Skipping malformed validation batch {batch_idx}")
                    continue
                inputs, hard_labels, soft_labels = batch[0], batch[1], batch[2]

                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    self.logger.warning(f"Input NaN/Inf detected in validation batch {batch_idx}, skipping.")
                    continue

                valid_indices = [i for i in range(inputs.size(0)) if inputs[i].view(-1).std(unbiased=False) >= eps]
                if not valid_indices:
                    continue

                inputs_valid = inputs[valid_indices].to(self.device, non_blocking=True)
                hard_labels_valid = hard_labels[valid_indices].to(self.device, non_blocking=True)
                soft_labels_valid = soft_labels[valid_indices].to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                    pred = self.model(inputs_valid)
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        self.logger.warning(f"Output NaN/Inf detected in validation batch {batch_idx}, skipping.")
                        continue
                    loss = self._compute_loss(pred, hard_labels_valid, soft_labels_valid)
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"Loss is NaN/Inf in validation batch {batch_idx}, skipping.")
                        continue

                prob = torch.sigmoid(pred.float())
                pred_label = (prob > self.prediction_threshold).float()
                truth_hard = hard_labels_valid.float()
                if pred_label.dim() == truth_hard.dim() - 1:
                     truth_hard = truth_hard.squeeze(1)
                elif pred_label.dim() == truth_hard.dim() and pred_label.size(1) == 1:
                     pred_label = pred_label.squeeze(1)
                     truth_hard = truth_hard.squeeze(1)

                perf_metrics = self._compute_performance_metrics(pred_label, truth_hard, eps)

                batch_size = inputs_valid.size(0)
                total_samples += batch_size
                valid_batches += 1

                total_metrics['loss'] += loss.item() * batch_size
                for loss_name, loss_value in self.current_losses.items():
                     if loss_name != 'loss' and loss_name in total_metrics:
                          total_metrics[loss_name] += loss_value * batch_size
                for metric_name, metric_value in perf_metrics.items():
                     if metric_name in total_metrics:
                          total_metrics[metric_name] += metric_value * batch_size

                if visualize and self.enable_validation_visualization:
                    all_hard_labels_viz.append(hard_labels_valid.cpu())
                    all_outputs_viz.append(pred.cpu())

                # Collect data for advanced plots
                if visualize and self.config['training'].get('enable_advanced_plots', False):
                    for i in range(inputs_valid.size(0)):
                        sample_prob = prob[i].float()
                        sample_hard = hard_labels_valid[i].float()
                        sample_soft = soft_labels_valid[i].float()

                        pos_mask = sample_hard > 0.5
                        neg_mask = sample_hard <= 0.5

                        std_pos = sample_prob[pos_mask].std(unbiased=False).item() if pos_mask.sum() > 1 else 0.0
                        std_neg = sample_prob[neg_mask].std(unbiased=False).item() if neg_mask.sum() > 1 else 0.0
                        std_diff = std_pos - std_neg if pos_mask.any() and neg_mask.any() else 0.0

                        proportion_positive = pos_mask.float().mean().item()

                        sample_y_pred = (sample_prob > self.prediction_threshold).float()
                        tp = (sample_y_pred * sample_hard).sum().item()
                        fp = (sample_y_pred * (1 - sample_hard)).sum().item()
                        fn = ((1 - sample_y_pred) * sample_hard).sum().item()
                        precision = tp / (tp + fp + 1e-6)
                        recall = tp / (tp + fn + 1e-6)
                        f1 = 2 * precision * recall / (precision + recall + 1e-6)

                        std_diffs.append(std_diff)
                        proportion_positives.append(proportion_positive)
                        f1_scores.append(f1)

                        fn_mask = (sample_hard > 0.5) & (sample_y_pred <= 0.5)
                        fp_mask = (sample_hard <= 0.5) & (sample_y_pred > 0.5)
                        if fn_mask.any():
                            fn_soft_labels.extend(sample_soft[fn_mask].cpu().numpy().flatten())
                        if fp_mask.any():
                            fp_soft_labels.extend(sample_soft[fp_mask].cpu().numpy().flatten())

        if valid_batches == 0 or total_samples == 0:
            self.logger.warning("No valid batches processed during validation.")
            return {metric: float('nan') for metric in metric_keys}

        avg_metrics = {metric: total_metrics[metric] / total_samples for metric in metric_keys}
        for loss_comp in ['ce', 'mse', 'hard_dist', 'dice', 'focal']:
            if loss_comp in avg_metrics:
                 avg_metrics[loss_comp] = total_metrics[loss_comp] / total_samples
        avg_metrics['loss'] = total_metrics['loss'] / total_samples

        self.logger.info(f"Validation Results ({valid_batches} batches, {total_samples} samples) - "
                         f"Avg Loss: {avg_metrics['loss']:.4f}, CE: {avg_metrics['ce']:.4f}, "
                         f"Dice: {avg_metrics['dice']:.4f}, Focal: {avg_metrics['focal']:.4f}, "
                         f"MSE: {avg_metrics['mse']:.4f}, "
                         f"F1: {avg_metrics['f1']:.4f}, Precision: {avg_metrics['precision']:.4f}, "
                         f"Recall: {avg_metrics['recall']:.4f}, IoU: {avg_metrics['iou']:.4f}")

        if visualize and self.enable_validation_visualization and all_outputs_viz:
            all_hard_labels_viz = torch.cat(all_hard_labels_viz)
            all_outputs_viz = torch.cat(all_outputs_viz)
            self._generate_visualizations(all_hard_labels_viz, all_outputs_viz, epoch)
            self.logger.info(f"Saved visualizations for epoch {epoch} in: {self.visualization_dir}")

        # Generate advanced plots if enabled
        if visualize and self.config['training'].get('enable_advanced_plots', False):
            self._plot_std_diff_vs_f1(std_diffs, f1_scores, epoch)
            self._plot_proportion_vs_f1(proportion_positives, f1_scores, epoch)
            self._plot_misclassified_soft_labels(fn_soft_labels, fp_soft_labels, epoch)

        self.last_validation_metrics = avg_metrics
        return avg_metrics

    def _compute_loss(self, pred, hard, soft):
        """Compute the combined loss."""
        hard_weight = self.config['training'].get('hard_weight', 1.0)
        soft_weight = self.config['training'].get('soft_weight', 1.0)
        dice_weight = self.config['training'].get('dice_weight', 1.0)
        focal_weight = self.config['training'].get('focal_weight', 1.0)
        hard_dist_weight = self.config['training'].get('hard_dist_weight', 0.0)

        total_weight = hard_weight + soft_weight + hard_dist_weight + dice_weight + focal_weight
        if total_weight > 0:
            hard_weight /= total_weight
            soft_weight /= total_weight
            hard_dist_weight /= total_weight
            dice_weight /= total_weight
            focal_weight /= total_weight
        else:
            hard_weight = soft_weight = hard_dist_weight = dice_weight = focal_weight = 0.0

        prob = torch.sigmoid(pred.float())

        pos_weight_config = self.config['training'].get('pos_weight', 1000.0)
        if isinstance(pos_weight_config, str) and pos_weight_config.lower() == 'auto':
            positives = hard.sum().float()
            negatives = float(hard.numel()) - positives
            pos_weight_val = negatives / (positives + 1e-6) if positives > 0 else 1.0
            pos_weight = torch.tensor(pos_weight_val, device=self.device)
        else:
            pos_weight = torch.tensor(float(pos_weight_config), device=self.device)
        
        ce_loss = F.binary_cross_entropy_with_logits(pred, hard.float(), pos_weight=pos_weight)
        mse_loss = self._compute_mse_loss(prob, soft.float())
        hard_dist_loss = self._compute_mse_loss(prob, hard.float())
        dice_loss = self._soft_dice_loss(prob, hard.float())
        focal_loss = self._focal_loss(pred, hard.float()) 

        loss = (hard_weight * ce_loss +
                soft_weight * mse_loss +
                hard_dist_weight * hard_dist_loss +
                dice_weight * dice_loss +
                focal_weight * focal_loss)

        self.current_losses = {
            'loss': loss.item() if not torch.isnan(loss) else float('nan'),
            'ce': ce_loss.item() if not torch.isnan(ce_loss) else float('nan'),
            'mse': mse_loss.item() if not torch.isnan(mse_loss) else float('nan'),
            'hard_dist': hard_dist_loss.item() if not torch.isnan(hard_dist_loss) else float('nan'),
            'dice': dice_loss.item() if not torch.isnan(dice_loss) else float('nan'),
            'focal': focal_loss.item() if not torch.isnan(focal_loss) else float('nan'),
        }

        return loss

    def _soft_dice_loss(self, prob, target, smooth=1e-6):
        """Compute soft Dice loss using probabilities."""
        dims = tuple(range(2, prob.dim()))
        intersection = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice_coeff = (2. * intersection + smooth) / (denom + smooth)
        return (1.0 - dice_coeff).mean()

    def _compute_mse_loss(self, pred_prob, target_soft_or_hard):
        """Compute MSE loss."""
        return F.mse_loss(pred_prob, target_soft_or_hard)

    def _focal_loss(self, pred_logits, target_hard, gamma=None, alpha=None):
        """Compute focal loss using logits."""
        gamma = gamma if gamma is not None else self.config['training'].get('focal_gamma', 2.5)
        alpha = alpha if alpha is not None else self.config['training'].get('focal_alpha', 0.9)

        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_hard, reduction='none')
        probs = torch.sigmoid(pred_logits)
        p_t = probs * target_hard + (1 - probs) * (1 - target_hard)
        modulating_factor = (1.0 - p_t)**gamma
        alpha_t = alpha * target_hard + (1 - alpha) * (1 - target_hard)
        focal_loss = alpha_t * modulating_factor * bce_loss
        return focal_loss.mean()

    def _generate_visualizations(self, hard_labels_cpu, outputs_logits_cpu, epoch):
        """Generate validation visualizations using the visualization module."""
        if visualization is None:
            self.logger.warning("Visualization module not available, cannot generate plots.")
            return
        threshold = self.prediction_threshold
        
        try:
            hard_labels_tensor = hard_labels_cpu
            outputs_logits_tensor = outputs_logits_cpu
        except Exception as e:
            self.logger.error(f"Error preparing tensors for visualization: {e}")
            return

        for plot_type, func in [
            ('prediction_densities', visualization.plot_label_densities),
            ('confusion_matrix', visualization.plot_confusion_matrix),
            ('metrics_vs_threshold', visualization.plot_metrics_vs_threshold)
        ]:
            try:
                path = os.path.join(self.visualization_dir, f'{plot_type}_epoch_{epoch}.png')
                if plot_type == 'metrics_vs_threshold':
                    func(hard_labels_tensor, outputs_logits_tensor, 
                         np.linspace(0.1, 0.9, 9), path, epoch)
                else:
                    func(hard_labels_tensor, outputs_logits_tensor, 
                         threshold, path, epoch)
            except Exception as e:
                self.logger.warning(f"Failed to generate {plot_type} for epoch {epoch}: {e}", exc_info=True)

    def _plot_std_diff_vs_f1(self, std_diffs, f1_scores, epoch):
        """Generate plot for std difference vs F1 score."""
        save_path = os.path.join(self.visualization_dir, f'std_diff_vs_f1_epoch_{epoch}.png')
        visualization.plot_std_diff_vs_f1(std_diffs, f1_scores, save_path, epoch)
        self.logger.info(f"Saved std_diff_vs_f1 plot for epoch {epoch} at {save_path}")

    def _plot_proportion_vs_f1(self, proportion_positives, f1_scores, epoch):
        """Generate plot for proportion positive vs F1 score."""
        save_path = os.path.join(self.visualization_dir, f'proportion_vs_f1_epoch_{epoch}.png')
        visualization.plot_proportion_vs_f1(proportion_positives, f1_scores, save_path, epoch)
        self.logger.info(f"Saved proportion_vs_f1 plot for epoch {epoch} at {save_path}")

    def _plot_misclassified_soft_labels(self, fn_soft_labels, fp_soft_labels, epoch):
        """Generate density plot for misclassified soft labels."""
        save_path = os.path.join(self.visualization_dir, f'misclassified_soft_labels_epoch_{epoch}.png')
        visualization.plot_misclassified_soft_labels(fn_soft_labels, fp_soft_labels, save_path, epoch)
        self.logger.info(f"Saved misclassified_soft_labels plot for epoch {epoch} at {save_path}")

    def save_checkpoint(self, epoch, is_best=False, val_loss=None):
        """Save model checkpoint."""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.amp_enabled else None,
            'cumulative_time': self.cumulative_time,
            'best_val_loss': self.best_val_loss,
            'best_val_dice_score': self.best_val_dice_score,
            'best_epoch': self.best_epoch,
            'config': self.config
        }

        if epoch == 1 or epoch % self.checkpoint_interval == 0 or epoch == self.config['training']['epochs']:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            try:
                torch.save(checkpoint, path)
                self.logger.info(f"Checkpoint saved: {path}")
            except Exception as e:
                 self.logger.error(f"Failed to save checkpoint {path}: {e}")

        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            try:
                 torch.save(checkpoint, best_path)
                 self.logger.info(f"New best model saved: {best_path} (Epoch: {epoch}, Val Loss: {val_loss:.6f})")
            except Exception as e:
                 self.logger.error(f"Failed to save best model checkpoint {best_path}: {e}")

        if self.delete_old_checkpoints:
            self.delete_old_checkpoints_func(epoch)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting fresh.")
            return 1, 0.0

        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_to_load = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                     self.logger.warning(f"Could not load scheduler state: {e}")
            if self.amp_enabled and checkpoint.get('scaler_state_dict'):
                 try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 except Exception as e:
                      self.logger.warning(f"Could not load GradScaler state: {e}")
            start_epoch = checkpoint.get('epoch', 0) + 1
            self.cumulative_time = checkpoint.get('cumulative_time', 0.0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_dice_score = checkpoint.get('best_val_dice_score', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.logger.info(f"Resumed from epoch {start_epoch}. Best val loss so far: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
            return start_epoch, self.cumulative_time
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return 1, 0.0

    def update_csv(self, epoch, train_metrics, val_metrics=None):
        """Update CSV log with training and validation metrics."""
        if not self.enable_csv_logging:
            return

        row_data = {'Epoch': epoch, 'Cumulative Time (s)': f"{self.cumulative_time:.2f}"}
        for prefix, metrics in [('Train', train_metrics), ('Val', val_metrics)]:
            if metrics:
                row_data[f'{prefix} Loss'] = f"{metrics.get('loss', float('nan')):.6f}"
                row_data[f'{prefix} CE'] = f"{metrics.get('ce', float('nan')):.6f}"
                row_data[f'{prefix} MSE'] = f"{metrics.get('mse', float('nan')):.6f}"
                row_data[f'{prefix} Hard Dist'] = f"{metrics.get('hard_dist', float('nan')):.6f}"
                row_data[f'{prefix} Dice'] = f"{metrics.get('dice', float('nan')):.6f}"
                row_data[f'{prefix} Focal'] = f"{metrics.get('focal', float('nan')):.6f}"
                row_data[f'{prefix} Precision'] = f"{metrics.get('precision', float('nan')):.6f}"
                row_data[f'{prefix} Recall'] = f"{metrics.get('recall', float('nan')):.6f}"
                row_data[f'{prefix} F1'] = f"{metrics.get('f1', float('nan')):.6f}"
                row_data[f'{prefix} IoU'] = f"{metrics.get('iou', float('nan')):.6f}"
            else:
                if prefix == 'Val':
                    for field in self.fieldnames:
                        if field.startswith('Val '):
                            row_data[field] = ''

        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            full_row_data = {field: row_data.get(field, '') for field in self.fieldnames}
            writer.writerow(full_row_data)

    def early_stopping_check(self):
        """Check if early stopping criteria are met."""
        if not self.enable_early_stopping or len(self.val_loss_history) <= self.early_stopping_patience:
            return False, ""
        current_loss = self.val_loss_history[-1]
        reference_loss = self.val_loss_history[-(self.early_stopping_patience + 1)]
        if current_loss >= reference_loss - self.early_stopping_delta:
             reason = (f"Validation loss ({current_loss:.6f}) has not improved by more than delta="
                       f"{self.early_stopping_delta:.2e} compared to {self.early_stopping_patience} epochs ago "
                       f"({reference_loss:.6f}).")
             return True, reason
        return False, ""

    def should_stop_early_after_resume(self, current_epoch):
        """Check if training should stop early based on best epoch and patience after resuming."""
        if not self.enable_early_stopping or self.best_epoch == 0:
            return False, ""
        
        patience_in_epochs = self.early_stopping_patience * self.validation_interval
        if current_epoch > self.best_epoch + patience_in_epochs:
            reason = (f"Current epoch {current_epoch} exceeds best epoch {self.best_epoch} "
                      f"plus patience {self.early_stopping_patience} * validation_interval {self.validation_interval}")
            return True, reason
        
        return False, ""

    def find_latest_checkpoint(self):
        """Find the latest checkpoint file."""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        latest_epoch = -1
        latest_checkpoint = None
        if not os.path.isdir(checkpoint_dir):
            return None
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pth'):
                try:
                    epoch = int(filename.split('_epoch_')[1].split('.')[0])
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_checkpoint = os.path.join(checkpoint_dir, filename)
                except (ValueError, IndexError):
                    continue
        return latest_checkpoint

    def delete_old_checkpoints_func(self, current_epoch):
        """Delete old checkpoints based on retention policy."""
        if not self.delete_old_checkpoints:
            return
        checkpoint_dir = self.config['training']['checkpoint_dir']
        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
             if filename.startswith('checkpoint_epoch_') and filename.endswith('.pth'):
                  try:
                       epoch = int(filename.split('_epoch_')[1].split('.')[0])
                       checkpoints.append({'path': os.path.join(checkpoint_dir, filename), 'epoch': epoch})
                  except (ValueError, IndexError):
                       continue
        if not checkpoints:
            return
        checkpoints.sort(key=lambda x: x['epoch'])
        keep_epochs = set()
        if checkpoints:
             keep_epochs.add(checkpoints[-1]['epoch'])
        keep_epochs.add(self.best_epoch)
        keep_epochs.add(1)
        if self.keep_checkpoint_every > 0:
             keep_epochs.update(cp['epoch'] for cp in checkpoints if cp['epoch'] % self.keep_checkpoint_every == 0)
        keep_epochs.add(current_epoch)
        deleted_count = 0
        for cp in checkpoints:
            if cp['epoch'] not in keep_epochs:
                try:
                    os.remove(cp['path'])
                    deleted_count += 1
                except OSError as e:
                    self.logger.warning(f"Failed to delete checkpoint {cp['path']}: {e}")
        if deleted_count > 0:
             self.logger.info(f"Deleted {deleted_count} old checkpoints.")

    def train(self, train_loader, val_loader=None, resume_epoch=None, seed=None):
        """Train the model over multiple epochs."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.logger.info(f"Using fixed random seed: {seed}")

        start_epoch, self.cumulative_time = 1, 0.0
        checkpoint_to_load = None
        if resume_epoch is not None:
             path = os.path.join(self.config['training']['checkpoint_dir'], f"checkpoint_epoch_{resume_epoch}.pth")
             if os.path.exists(path):
                  checkpoint_to_load = path
             else:
                  self.logger.warning(f"Specified resume epoch {resume_epoch} checkpoint not found at {path}")
        if checkpoint_to_load is None and self.config['training'].get('auto_resume', True):
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_to_load = latest_checkpoint
            elif self.initial_checkpoint_path and os.path.exists(self.initial_checkpoint_path):
                 self.logger.info(f"No existing checkpoints found, using initial checkpoint: {self.initial_checkpoint_path}")
                 checkpoint_to_load = self.initial_checkpoint_path
                 self.best_val_loss = float('inf')
                 self.best_epoch = 0
                 self.cumulative_time = 0.0
        if checkpoint_to_load:
            start_epoch, self.cumulative_time = self.load_checkpoint(checkpoint_to_load)
            
            # Check if we should stop early after resuming
            if self.enable_early_stopping and self.best_epoch > 0:
                should_stop, reason = self.should_stop_early_after_resume(start_epoch)
                if should_stop:
                    self.logger.info(f"EARLY STOPPING triggered after resume: {reason}")
                    return self.best_val_loss, self.best_val_dice_score
        else:
             self.logger.info("No checkpoint found or specified. Starting training from scratch.")

        try:
             total_epochs = self.config['training']['epochs']
             steps_per_epoch = len(train_loader) // self.config['training'].get('accumulation_steps', 1)
             total_steps = total_epochs * steps_per_epoch
             self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                 self.optimizer,
                 max_lr=self.lr,
                 total_steps=total_steps,
                 pct_start=self.config['training'].get('pct_start', 0.3),
                 div_factor=self.config['training'].get('div_factor', 25),
                 final_div_factor=self.config['training'].get('final_div_factor', 1e4),
             )
             if checkpoint_to_load:
                  self.logger.info("Scheduler state loaded from checkpoint.")
             elif start_epoch > 1:
                  initial_step = (start_epoch - 1) * steps_per_epoch
                  self.logger.warning(f"Manually stepping scheduler to step {initial_step} corresponding to resumed epoch {start_epoch}")
        except Exception as e:
             self.logger.error(f"Failed to initialize scheduler: {e}")
             self.scheduler = None

        if start_epoch == 1:
             batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'N/A'
             accum_steps = self.config['training'].get('accumulation_steps', 1)
             eff_batch_size = batch_size * accum_steps if isinstance(batch_size, int) else 'N/A'
             training_params = (
                 f"Training Params --- LR: {self.lr:.1e}, WeightDecay: {self.config['training']['weight_decay']:.1e}, "
                 f"Epochs: {total_epochs}, BatchSize: {batch_size}, AccumSteps: {accum_steps}, "
                 f"EffBatchSize: {eff_batch_size}, Device: {self.device}, AMP: {self.amp_enabled}"
             )
             loss_weights = (
                 f"Loss Weights (Normalized) --- CE: {self.config['training'].get('hard_weight', 1.0):.2f}, "
                 f"MSE: {self.config['training'].get('soft_weight', 1.0):.2f}, "
                 f"Dice: {self.config['training'].get('dice_weight', 1.0):.2f}, "
                 f"Focal: {self.config['training'].get('focal_weight', 1.0):.2f}, "
                 f"HardDist: {self.config['training'].get('hard_dist_weight', 0.0):.2f}"
             )
             self.logger.info(training_params)
             self.logger.info(loss_weights)

        self.logger.info(f"Starting training loop from epoch {start_epoch} to {total_epochs}")
        
        validation_epochs = []
        for epoch in range(start_epoch, total_epochs + 1):
            if epoch % self.validation_interval == 0 or epoch == total_epochs:
                validation_epochs.append(epoch)
        
        visualization_epochs = []
        if self.enable_validation_visualization and self.viz_count > 0 and validation_epochs:
            n_val = len(validation_epochs)
            viz_count = min(self.viz_count, n_val)
            if viz_count > 0:
                intervals = n_val // viz_count
                for i in range(viz_count):
                    idx = i * intervals
                    if idx < n_val:
                        visualization_epochs.append(validation_epochs[idx])
                if validation_epochs[-1] not in visualization_epochs:
                    visualization_epochs.append(validation_epochs[-1])
            self.logger.info(f"Will create visualizations at epochs: {visualization_epochs}")

        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - epoch_start_time
            self.cumulative_time += epoch_time

            log_msg_parts = [
                f"Epoch {epoch}/{total_epochs}",
                f"Time: {epoch_time:.2f}s",
                f"CumTime: {self.cumulative_time/3600:.2f}h",
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}",
                f"TrainLoss: {train_metrics.get('loss', float('nan')):.4f}",
                f"TrainF1: {train_metrics.get('f1', float('nan')):.4f}"
            ]

            val_metrics = None
            is_best = False
            if val_loader and epoch in validation_epochs:
                do_visualize = epoch in visualization_epochs
                val_metrics = self.validate(val_loader, epoch, visualize=do_visualize)
                val_loss = val_metrics.get('loss', float('inf'))
                if not np.isnan(val_loss):
                     self.val_loss_history.append(val_loss)
                     if val_loss < self.best_val_loss:
                         self.best_val_loss = val_loss
                         self.best_epoch = epoch
                         is_best = True
                         log_msg_parts.append(f"ValLoss: {val_loss:.4f} *Best*")
                     else:
                          log_msg_parts.append(f"ValLoss: {val_loss:.4f}")
                     log_msg_parts.append(f"ValF1: {val_metrics.get('f1', float('nan')):.4f}")
                     if self.enable_early_stopping:
                         should_stop, reason = self.early_stopping_check()
                         if should_stop:
                             self.logger.info(f"EARLY STOPPING triggered at epoch {epoch}: {reason}")
                             self.save_checkpoint(epoch, is_best=is_best, val_loss=val_loss)
                             self.update_csv(epoch, train_metrics, val_metrics)
                             break
                else:
                     log_msg_parts.append("ValLoss: NaN")

            if epoch % self.checkpoint_interval == 0 or epoch == total_epochs or is_best:
                self.save_checkpoint(epoch, is_best=is_best, val_loss=self.last_validation_metrics.get('loss') if self.last_validation_metrics else None)

            self.logger.info(" | ".join(log_msg_parts))
            self.update_csv(epoch, train_metrics, val_metrics)

        self.logger.info(f"Training finished after epoch {epoch}. Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        if self.nan_error_count > 0:
            self.logger.warning(f"Total NaN/Inf gradient occurrences during training: {self.nan_error_count}")
        return self.best_val_loss, self.best_val_dice_score

    def lr_forward_search(self, train_loader, start_lr=1e-7, end_lr=1, num_steps=100, beta=0.98):
        """Perform learning rate range test."""
        self.logger.info(f"Starting LR range test from {start_lr:.2e} to {end_lr:.2e} over {num_steps} steps.")
        original_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.amp_enabled else None
        }
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        temp_optimizer = torch.optim.AdamW(self.model.parameters(), lr=start_lr)
        lr_mult = (end_lr / start_lr) ** (1 / (num_steps - 1)) if num_steps > 1 else 1.0
        current_lr = start_lr
        lrs = []
        losses = []
        avg_loss = 0.0
        best_loss = float('inf')
        data_iter = iter(train_loader)

        pbar = tqdm(range(num_steps), desc="LR Search")
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            if not isinstance(batch, (list, tuple)) or len(batch) < 3: continue
            inputs, hard_labels, soft_labels = batch[0], batch[1], batch[2]
            if inputs.numel() == 0: continue
            inputs = inputs.to(self.device, non_blocking=True)
            hard_labels = hard_labels.to(self.device, non_blocking=True)
            soft_labels = soft_labels.to(self.device, non_blocking=True)
            for param_group in temp_optimizer.param_groups:
                param_group['lr'] = current_lr
            lrs.append(current_lr)
            temp_optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, hard_labels, soft_labels)
            if torch.isnan(loss) or torch.isinf(loss):
                 self.logger.warning(f"LR Search: Loss is NaN/Inf at LR={current_lr:.2e}, stopping search early.")
                 break
            if step == 0:
                 avg_loss = loss.item()
            else:
                 avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**(step + 1))
            losses.append(smoothed_loss)
            if smoothed_loss < best_loss:
                 best_loss = smoothed_loss
            if step > 10 and smoothed_loss > 4 * best_loss:
                 self.logger.info(f"LR Search: Loss ({smoothed_loss:.4f}) diverged significantly from best ({best_loss:.4f}) at LR={current_lr:.2e}")
                 break
            self.scaler.scale(loss).backward()
            self.scaler.step(temp_optimizer)
            self.scaler.update()
            current_lr *= lr_mult
            current_lr = min(current_lr, end_lr)
            pbar.set_postfix({'LR': f"{current_lr:.2e}", 'Loss': f"{smoothed_loss:.4f}"})
            
        self.logger.info("Restoring model and optimizer state after LR search.")
        self.model.load_state_dict(original_state['model'])
        self.optimizer.load_state_dict(original_state['optimizer'])
        if self.amp_enabled and original_state['scaler']:
            self.scaler.load_state_dict(original_state['scaler'])
        for state in self.optimizer.state.values():
             for k, v in state.items():
                  if isinstance(v, torch.Tensor):
                       state[k] = v.to(self.device)
        self.logger.info("LR range test finished.")
        return lrs, losses

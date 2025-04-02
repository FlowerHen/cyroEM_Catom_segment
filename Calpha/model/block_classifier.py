import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBlock, DownsampleLayer, UpsampleAdd
from ..dataset.cube_rotation import sample_rotation_params, apply_rotation

class CalphaBlockClassifier(nn.Module):
    """
    A lightweight 3D CNN model for block classification of 64³ blocks.
    This model determines whether a 64³ block contains C-alpha atoms or not.
    """
    def __init__(self, input_size=64, base_channels=16):
        super(CalphaBlockClassifier, self).__init__()
        self.input_size = input_size
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block: 64³ -> 32³
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Second conv block: 32³ -> 16³
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Third conv block: 16³ -> 8³
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Fourth conv block: 8³ -> 4³
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Calculate the size after convolutions and pooling
        # Input: 64³ -> After 4 pooling layers with stride 2: 4³
        feature_size = input_size // (2**4)
        flattened_size = base_channels * 8 * (feature_size**3)
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return self.sigmoid(out).squeeze(-1)
    
    def compute_background_noise(self, volume, threshold=0.1):
        """
        Compute background noise intensity from blocks without C_alpha.
        
        Args:
            volume: Input volume tensor [B, 1, D, H, W]
            threshold: Probability threshold to consider a region as background
            
        Returns:
            Estimated background noise level for each sample in the batch
        """
        with torch.no_grad():
            pred = self.forward(volume)
            block_mask = (torch.sigmoid(pred) < threshold).float()  # Areas with low probability of C_alpha
            
        # Calculate statistics in background regions
        batch_size = volume.size(0)
        noise_levels = []
        
        for i in range(batch_size):
            vol = volume[i, 0]  # [D, H, W]
            mask = block_mask[i, 0]  # [D, H, W]
            
            # If there are background regions
            if mask.sum() > 0:
                # Extract background regions
                background = vol[mask > 0]
                # Calculate standard deviation as noise level
                noise_level = background.std().item()
            else:
                # If no background regions, use overall standard deviation
                noise_level = vol.std().item()
                
            noise_levels.append(noise_level)
            
        return torch.tensor(noise_levels, device=volume.device)
    
import os
import numpy as np
import torch
import torch.nn.functional as F

class CalphaBlockTrainer:
    """
    Trainer class for the C-alpha block classifier model.
    This trainer works with 64³ blocks to classify whether they contain C-alpha atoms.
    """
    def __init__(self, model, config, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['block_classifier']['lr']),
            weight_decay=float(config['block_classifier']['weight_decay'])
        )
        os.makedirs(config['block_classifier']['checkpoint_dir'], exist_ok=True)
        
        # Set up loss function with class weighting if needed
        pos_weight = config['block_classifier'].get('pos_weight', 1.0) # changed to 1 since with_calpha is boolean
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))

        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            volumes, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(volumes)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping if configured
            if self.config['block_classifier'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['block_classifier']['grad_clip']
                )
                
            self.optimizer.step()
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
        accuracy = correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # For calculating precision, recall, F1
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for batch in dataloader:
                volumes, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(volumes)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Calculate metrics
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Update precision/recall counters
                true_positives += ((predictions == 1) & (labels == 1)).sum().item()
                false_positives += ((predictions == 1) & (labels == 0)).sum().item()
                false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_path = os.path.join(
            self.config['block_classifier']['checkpoint_dir'],
            f"checkpoint_epoch_{epoch}.pth"
        )
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(
                self.config['block_classifier']['checkpoint_dir'],
                "best_model.pth"
            )
            torch.save(checkpoint, best_path)
            logging.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1

# Alias for backward compatibility
blockClassifierTrainer = CalphaBlockTrainer
SimpleblockClassifier = CalphaBlockClassifier

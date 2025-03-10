import os
import torch
import yaml
from Calpha.dataset.dataset import get_data_loaders
from Calpha.training.trainer import CryoTrainer
from Calpha.model.segmentation_model import SegmentationModel,SegmentationModelMini,SegmentationModelResnet,SegmentationModelAttn

config_path = "/root/project/Calpha/Calpha/config/base.yaml"

def load_config(config_path = ""):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config(config_path)
    model = SegmentationModelResnet()
    trainer = CryoTrainer(model, config)
    
    checkpoint_dir = config['training']['checkpoint_dir']
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir)
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoints:
            epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
            latest_epoch = max(epochs)
            latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pth')
    
    start_epoch = 1
    if latest_checkpoint:
        start_epoch = trainer.load_checkpoint(latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")
    
    # get_data_loaders returns training and validation data loaders
    train_loader, val_loader = get_data_loaders(config)
    trainer.train(train_loader, val_loader=val_loader, resume_epoch=start_epoch)

if __name__ == "__main__":
    main()
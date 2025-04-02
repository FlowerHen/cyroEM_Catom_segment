import os
from copy import deepcopy
from Calpha.dataset.dataset import get_data_loaders
from Calpha.training.trainer import CryoTrainer
from Calpha.model.segmentation_model import SegmentationModelResnet

def load_config(config_path):
    import yaml 
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_with_hyperparams(dice_weight, focal_weight, pos_weight, config):
    """
    Train model with specific hyperparameters
    """
    # Create experiment-specific config
    config_exp = deepcopy(config)
    exp_name = f"DW{dice_weight}_FW{focal_weight}_PW{pos_weight}"
    config_exp['training']['checkpoint_dir'] = os.path.join(
        config['training']['checkpoint_dir'],
        exp_name
    )
    os.makedirs(config_exp['training']['checkpoint_dir'], exist_ok=True)
    
    # Update loss weights in config
    config_exp['training']['loss_weights'] = {
        'dice_weight': dice_weight,
        'focal_weight': focal_weight,
        'pos_weight': pos_weight
    }
    
    # Initialize model and trainer
    model = SegmentationModelResnet(use_sa=False)
    trainer = CryoTrainer(model, config_exp)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(config_exp)
    
    # Train the model
    best_val_loss, best_val_dice = trainer.train(train_loader, val_loader)
    
    return best_val_loss, best_val_dice

if __name__ == "__main__":
    # Load base configuration
    config = load_config("/root/project/Calpha/Calpha/config/base.yaml")
    results = {}
    
    # Define hyperparameter combinations to test
    hyperparam_combinations = [
        # Standard cases (dice_weight + focal_weight = 1)
        # (1, 0, 100),  # Dice only, lower pos_weight
        # (1, 0, 300),  # Dice only, higher pos_weight
        # (0, 1, 100),  # Focal only, lower pos_weight
        # (0, 1, 300),  # Focal only, higher pos_weight
        
        # Special cases with both weights zero
        (0, 0, 100),  # Neither loss, original pos_weight
        (0, 0, 300),  # Neither loss, higher pos_weight
        (0, 0, 500),  # Neither loss, even higher pos_weight
        
        # Cases with one weight >1
        (3, 0, 10),   # Strong dice weighting
        (0, 3, 10),   # Strong focal weighting
        
        # Special case with equal weights
        (2, 2, 1),    # Balanced but non-standard weights
    ]
    
    # Run training for each hyperparameter combination
    for dw, fw, pw in hyperparam_combinations:
        print(f"\nTraining with dice_weight={dw}, focal_weight={fw}, pos_weight={pw}")
        try:
            best_val_loss, best_val_dice = train_with_hyperparams(dw, fw, pw, config)
            results[(dw, fw, pw)] = (best_val_loss, best_val_dice)
        except Exception as e:
            print(f"Error occurred during training: {e}")
            results[(dw, fw, pw)] = (None, None)
    
    # Print summary of results
    print("\nTraining Summary:")
    for params, (best_val_loss, best_val_dice) in results.items():
        dw, fw, pw = params
        if best_val_loss is not None:
            print(f"DW{dw}_FW{fw}_PW{pw}: Val Loss {best_val_loss:.4f}, Dice {best_val_dice:.4f}")
        else:
            print(f"DW{dw}_FW{fw}_PW{pw}: Training Failed")

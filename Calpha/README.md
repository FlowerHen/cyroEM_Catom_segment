# Cryo-EM Protein Structure Segmentation

## Project Overview
This project provides a deep learning framework for 3D protein structure segmentation from Cryo-Electron Microscopy (Cryo-EM) data. The system includes:

- Multiple 3D UNet variants with Res2Net blocks and attention mechanisms
- Comprehensive training pipeline with advanced features
- Data loading and preprocessing for Cryo-EM volumes
- Inference and postprocessing capabilities

## Architecture

### Model Input/Output Specifications
- **Input**: 3D volume tensors of shape (1, D, D, D) where D is the crop size (default 64)
- **Output**: Segmentation masks of shape (1, D, D, D) with values between 0-1
- Reference: [model/segmentation_model.py](model/segmentation_model.py)

### Model Variants
1. **SegmentationModel**: Basic 3D UNet with bottleneck blocks
2. **SegmentationModelResnet**: Res2Net-based UNet variant
3. **SegmentationModelAttn**: UNet with attention gates
4. **SegmentationModelMini**: Lightweight 2-layer UNet
5. **SegmentationModelResMini**: Lightweight Res2Net UNet
6. **SegmentationAttnMini**: Lightweight UNet with attention

Key architectural features:
- 3D convolutional operations
- Skip connections for feature fusion
- Dropout for regularization (default rate 0.2)
- Multiple loss functions (CE, MSE, Dice, Focal)

## Data Preprocessing
The preprocessing pipeline includes:

1. **Volume Loading**:
   - Load NPZ files containing 3D electron density maps
   - Extract voxel size and global origin metadata

2. **Structure Parsing**:
   - Parse CIF files to extract C-alpha atom coordinates
   - Filter invalid residues and atoms

3. **Label Generation**:
   - Create hard labels (binary masks at C-alpha positions)
   - Create soft labels (Gaussian-smoothed distance transforms)

4. **Data Augmentation**:
   - Random rotations
   - Resolution changes (Gaussian blur)
   - Additive noise
   - Intensity inversion
   - Gamma correction

## Training Process
Key training features:
- Mixed precision training
- Gradient clipping (default 1.0)
- OneCycle learning rate scheduling
- Early stopping
- Checkpointing and model saving
- Extensive logging and visualization

## Inference
The system provides:
- Binary inference with configurable thresholds
- Postprocessing for segmentation results
- Evaluation metrics calculation

## Requirements
Key dependencies:
- Python 3.x
- PyTorch (with CUDA if available)
- NumPy
- tqdm
- Matplotlib
- Seaborn
- Biopython (for CIF parsing)

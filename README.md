# Cryo-EM C-alpha Detection

A reproducible 3D heatmap pipeline for locating protein C-alpha atoms in cryo-EM density maps.

The maintained implementation lives in `src/cryo_calpha`. The original research prototype is preserved in `Calpha` for reference only; it is not used by the new CLI.

## Core contracts

- Density arrays use `ZYX` axis order.
- Atomic and map metadata use world `XYZ` coordinates in angstroms.
- Dataset splits are made by source protein before crops or augmentations are generated.
- Segmentation models output logits with shape `[B, 1, D, H, W]`.
- Sliding-window predictions are fused with Gaussian importance weights.
- C-alpha coordinates are extracted as local heatmap maxima and evaluated with one-to-one matching.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install a PyTorch build appropriate for the target CUDA version when the default wheel is not suitable. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Workflow

```bash
cryo-calpha validate-config --config configs/default.yaml
cryo-calpha build-manifest --config configs/default.yaml
cryo-calpha build-cache --config configs/default.yaml
cryo-calpha train --config configs/default.yaml
cryo-calpha infer --config configs/default.yaml --map input.npz --checkpoint runs/best.pt --output predictions.npz
cryo-calpha evaluate --config configs/default.yaml --predictions predictions.npz --structure truth.cif
```

All commands support `--help`. Paths in configuration files are resolved relative to the configuration file.

## Input data

Each source sample directory must contain exactly one `.npz` density map and one `.cif`/`.mmcif` structure. An NPZ map must contain:

- `grid`: 3D density array in `ZYX` order
- `voxel_size`: three values in `XYZ` order, in angstroms
- `global_origin`: three values in `XYZ` order, in angstroms

MRC/MAP files are supported for standalone inference. Training manifests currently use NPZ maps so their spatial metadata is explicit and testable.

## Development

```bash
python -m pytest
python -m ruff check .
```

The detailed audit and reconstruction rationale is in [CALPHA_PROBLEM_ANALYSIS_REPORT.md](CALPHA_PROBLEM_ANALYSIS_REPORT.md).

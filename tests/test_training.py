from pathlib import Path

import pytest


def test_tiny_training_run_steps_optimizer_and_saves_checkpoints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from cryo_calpha.config import (
        AppConfig,
        AugmentationConfig,
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from cryo_calpha.trainer import Trainer

    class TinyDataset(Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, object]:
            volume = torch.zeros(1, 4, 4, 4)
            volume[:, index, index, index] = 1
            return {"volume": volume, "heatmap": volume.clone()}

    config = AppConfig(
        data=DataConfig(tmp_path, tmp_path / "manifest", tmp_path / "cache"),
        augmentation=AugmentationConfig(enabled=False),
        model=ModelConfig(base_channels=4, depth=2, dropout=0),
        training=TrainingConfig(
            output_dir=tmp_path / "run",
            device="cpu",
            epochs=1,
            batch_size=1,
            num_workers=0,
            accumulation_steps=2,
            amp=False,
            early_stopping_patience=0,
        ),
        inference=InferenceConfig(),
    )
    loader = DataLoader(TinyDataset(), batch_size=1)
    trainer = Trainer(nn.Conv3d(1, 1, 1), config, train_loader=loader, val_loader=loader)
    result = trainer.fit()

    assert trainer.global_step == 2
    assert result["best_epoch"] == 1
    assert (tmp_path / "run" / "last.pt").is_file()
    assert (tmp_path / "run" / "best.pt").is_file()

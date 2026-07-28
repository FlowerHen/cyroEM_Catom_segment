from pathlib import Path

import numpy as np
import pytest


def test_cif_xyz_coordinate_is_cached_at_correct_zyx_index(tmp_path: Path) -> None:
    pytest.importorskip("Bio")
    from Bio.PDB import MMCIFIO
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    from cryo_calpha.cache import build_cache, read_cache_manifest
    from cryo_calpha.config import (
        AppConfig,
        AugmentationConfig,
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from cryo_calpha.manifest import SampleRecord, write_manifest

    sample_dir = tmp_path / "raw" / "sample-1"
    sample_dir.mkdir(parents=True)
    map_path = sample_dir / "density.npz"
    structure_path = sample_dir / "structure.cif"
    rng = np.random.default_rng(7)
    grid = rng.normal(0, 0.01, (8, 8, 8)).astype(np.float32)
    grid[5, 4, 3] = 1.0
    np.savez(
        map_path,
        grid=grid,
        voxel_size=np.ones(3),
        global_origin=np.zeros(3),
    )

    structure = Structure("sample-1")
    model = Model(0)
    chain = Chain("A")
    residue = Residue((" ", 1, " "), "ALA", " ")
    residue.add(Atom("CA", np.array([3.0, 4.0, 5.0]), 1.0, 1.0, " ", " CA ", 1, "C"))
    chain.add(residue)
    model.add(chain)
    structure.add(model)
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(structure_path))

    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [SampleRecord("sample-1", str(map_path), str(structure_path), "train")],
        manifest_path,
    )
    config = AppConfig(
        data=DataConfig(
            root_dir=tmp_path / "raw",
            manifest_path=manifest_path,
            cache_dir=tmp_path / "cache",
            crop_size_zyx=(8, 8, 8),
            overlap_fraction=0,
            empty_to_positive_ratio=0,
        ),
        augmentation=AugmentationConfig(enabled=False),
        model=ModelConfig(),
        training=TrainingConfig(output_dir=tmp_path / "run", device="cpu"),
        inference=InferenceConfig(),
    )
    cache_dir = build_cache(config)
    records = read_cache_manifest(cache_dir)

    assert len(records) == 1
    with np.load(cache_dir / records[0].cache_file, allow_pickle=False) as crop:
        assert crop["hard_label"][5, 4, 3] == 1
        assert np.unravel_index(np.argmax(crop["heatmap"]), (8, 8, 8)) == (5, 4, 3)
        np.testing.assert_allclose(crop["crop_origin_xyz"], [0, 0, 0])

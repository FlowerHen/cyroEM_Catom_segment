from pathlib import Path

import pytest

from cryo_calpha.config import load_config


def test_default_config_resolves_paths_from_config_directory() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "default.yaml")
    assert config.data.root_dir == (root / "data" / "raw").resolve()
    assert config.training.output_dir == (root / "runs" / "baseline").resolve()


def test_unknown_config_key_is_rejected(tmp_path) -> None:
    source = Path(__file__).parents[1] / "configs" / "default.yaml"
    text = source.read_text().replace(
        "  seed: 42\n\naugmentation:", "  seed: 42\n  typo: true\n\naugmentation:"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(path)

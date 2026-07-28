import pytest


def test_unet_returns_logits_with_input_spatial_shape() -> None:
    torch = pytest.importorskip("torch")
    from cryo_calpha.models import UNet3D

    model = UNet3D(base_channels=4, depth=2, dropout=0)
    inputs = torch.randn(1, 1, 17, 19, 21)
    outputs = model(inputs)
    assert outputs.shape == inputs.shape

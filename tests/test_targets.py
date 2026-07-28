import numpy as np

from cryo_calpha.targets import create_calpha_targets


def test_target_peak_uses_zyx_array_order() -> None:
    shape = (7, 11, 13)
    origin = np.array([10.0, 20.0, 30.0])
    voxel = np.array([1.0, 2.0, 3.0])
    coordinate_xyz = np.array([[16.0, 28.0, 36.0]])
    hard, heatmap = create_calpha_targets(shape, coordinate_xyz, origin, voxel, sigma_angstrom=1.5)
    assert hard[2, 4, 6] == 1
    assert np.unravel_index(np.argmax(heatmap), heatmap.shape) == (2, 4, 6)


def test_empty_coordinate_set_returns_empty_targets() -> None:
    hard, heatmap = create_calpha_targets(
        (3, 4, 5), np.empty((0, 3)), [0, 0, 0], [1, 1, 1], sigma_angstrom=1
    )
    assert not hard.any()
    assert not heatmap.any()

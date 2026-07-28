import numpy as np

from cryo_calpha.sliding_window import (
    gaussian_importance_map,
    optimizer_steps_per_epoch,
    sliding_window_starts,
)


def test_sliding_windows_cover_odd_volume() -> None:
    shape = (71, 83, 97)
    window = (32, 32, 32)
    starts = sliding_window_starts(shape, window, 0.25)
    count = np.zeros(shape, dtype=np.int16)
    for z, y, x in starts:
        count[z : z + 32, y : y + 32, x : x + 32] += 1
    assert np.all(count > 0)


def test_small_volume_has_single_padded_window_start() -> None:
    assert sliding_window_starts((10, 20, 30), 64, 0.5) == [(0, 0, 0)]


def test_gaussian_importance_is_positive_and_center_weighted() -> None:
    weights = gaussian_importance_map((9, 9, 9))
    assert weights.min() > 0
    assert weights[4, 4, 4] == weights.max()
    assert weights[4, 4, 4] > weights[0, 0, 0]


def test_optimizer_steps_use_ceiling_division() -> None:
    assert optimizer_steps_per_epoch(10, 4) == 3

import numpy as np

from cryo_calpha.spatial import (
    crop_origin_xyz,
    index_zyx_to_world_xyz,
    world_xyz_to_index_zyx,
)


def test_xyz_zyx_round_trip_with_anisotropic_voxels() -> None:
    origin = np.array([10.0, -5.0, 2.0])
    voxel_size = np.array([1.1, 1.3, 1.7])
    indices = np.array([[2, 4, 6], [5, 1, 9]])
    world = index_zyx_to_world_xyz(indices, origin, voxel_size)
    recovered = world_xyz_to_index_zyx(world, origin, voxel_size)
    np.testing.assert_array_equal(recovered, indices)


def test_crop_origin_reverses_start_axes() -> None:
    actual = crop_origin_xyz([10, 20, 30], [2, 4, 6], [1, 2, 3])
    np.testing.assert_allclose(actual, [16, 28, 36])


def test_nearest_rounding_does_not_truncate_negative_values_to_zero() -> None:
    result = world_xyz_to_index_zyx([[-0.9, 0, 0]], [0, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(result, [[0, 0, -1]])

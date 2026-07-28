import numpy as np

from cryo_calpha.io import load_npz_map


def test_npz_map_requires_explicit_spatial_metadata(tmp_path) -> None:
    path = tmp_path / "map.npz"
    np.savez(
        path,
        grid=np.zeros((3, 4, 5), dtype=np.float32),
        voxel_size=np.array([1.0, 1.5, 2.0]),
        global_origin=np.array([10.0, 20.0, 30.0]),
    )
    density = load_npz_map(path)
    assert density.grid_zyx.shape == (3, 4, 5)
    np.testing.assert_allclose(density.voxel_size_xyz, [1, 1.5, 2])


def test_default_mrc_axes_are_loaded_as_zyx(tmp_path) -> None:
    import pytest

    mrcfile = pytest.importorskip("mrcfile")
    from cryo_calpha.io import load_mrc_map

    path = tmp_path / "map.mrc"
    grid = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(grid)
        mrc.voxel_size = (1.0, 1.5, 2.0)
        mrc.header.origin.x = 10
        mrc.header.origin.y = 20
        mrc.header.origin.z = 30
    density = load_mrc_map(path)

    np.testing.assert_array_equal(density.grid_zyx, grid)
    np.testing.assert_allclose(density.voxel_size_xyz, [1, 1.5, 2])
    np.testing.assert_allclose(density.global_origin_xyz, [10, 20, 30])

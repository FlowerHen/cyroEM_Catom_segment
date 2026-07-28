import numpy as np

from cryo_calpha.transforms import cube_rotations


def test_cube_rotations_are_unique_and_orientation_preserving() -> None:
    rotations = cube_rotations()
    assert len(rotations) == 24
    assert all(rotation.determinant == 1 for rotation in rotations)
    marker = np.arange(27).reshape(3, 3, 3)
    signatures = {rotation.apply(marker).tobytes() for rotation in rotations}
    assert len(signatures) == 24

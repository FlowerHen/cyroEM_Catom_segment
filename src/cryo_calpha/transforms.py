from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _permutation_parity(permutation: tuple[int, int, int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right] for left in range(3) for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True)
class CubeRotation:
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]

    @property
    def determinant(self) -> int:
        return _permutation_parity(self.permutation) * int(np.prod(self.signs))

    def apply(self, array: NDArray[np.generic]) -> NDArray[np.generic]:
        if array.ndim != 3:
            raise ValueError("cube rotations require a three-dimensional array")
        result = np.transpose(array, self.permutation)
        for axis, sign in enumerate(self.signs):
            if sign < 0:
                result = np.flip(result, axis=axis)
        return np.ascontiguousarray(result)


def cube_rotations() -> tuple[CubeRotation, ...]:
    rotations = tuple(
        CubeRotation(permutation, signs)
        for permutation in itertools.permutations(range(3))
        for signs in itertools.product((-1, 1), repeat=3)
        if _permutation_parity(permutation) * int(np.prod(signs)) == 1
    )
    if len(rotations) != 24 or any(rotation.determinant != 1 for rotation in rotations):
        raise RuntimeError("failed to construct the 24 orientation-preserving cube rotations")
    return rotations


def random_cube_rotation(rng: np.random.Generator) -> CubeRotation:
    rotations = cube_rotations()
    return rotations[int(rng.integers(len(rotations)))]

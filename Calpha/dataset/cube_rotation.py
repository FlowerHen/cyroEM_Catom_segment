import numpy as np
import itertools
import random

def sample_rotation_params(n):
    """
    Sample n unique rotation parameters from the 24 possible rotations.
    Returns a list of tuples: (face_up, z_rot)
    """
    if n > 24:
        raise ValueError("n cannot be greater than 24, as there are only 24 unique rotations.")
    face_up_directions = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)
    ]
    z_rotations = [0, 1, 2, 3] # 90*z_rotations
    all_rotations = list(itertools.product(face_up_directions, z_rotations))
    sampled_rotations = random.sample(all_rotations, n)
    return sampled_rotations

def apply_rotation(matrix, rotation_param):
    """
    Apply a rotation to a 3D matrix given rotation parameters.
    rotation_param is a tuple: (face_up, z_rot)
    """
    face_up, z_rot = rotation_param
    rotated = np.transpose(matrix, face_up)
    rotated = np.rot90(rotated, k=z_rot, axes=(1, 2))
    return rotated

def sampling_cube_rotations(matrix, n):
    """
    Generate n random rotations of a 3D numpy matrix based on the 24 possible rotations.
    """
    sampled_rotations = sample_rotation_params(n)
    rotated_matrices = [apply_rotation(matrix, rotation_param) for rotation_param in sampled_rotations]
    return rotated_matrices


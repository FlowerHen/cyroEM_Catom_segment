import numpy as np

from cryo_calpha.evaluator import evaluate_probability_map
from cryo_calpha.targets import create_calpha_targets


def test_synthetic_target_to_evaluation_round_trip() -> None:
    truth = np.array([[4.0, 5.0, 6.0], [10.0, 9.0, 8.0]])
    _, heatmap = create_calpha_targets(
        (16, 16, 16), truth, [0, 0, 0], [1, 1, 1], sigma_angstrom=0.6
    )
    result = evaluate_probability_map(
        heatmap,
        truth,
        origin_xyz=[0, 0, 0],
        voxel_size_xyz=[1, 1, 1],
        probability_threshold=0.5,
        nms_radius_angstrom=2.0,
        match_radius_angstrom=1.0,
    )
    assert result.metrics["true_positives"] == 2
    assert result.metrics["false_positives"] == 0
    assert result.metrics["false_negatives"] == 0

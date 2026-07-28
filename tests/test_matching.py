import numpy as np

from cryo_calpha.matching import match_points_one_to_one


def test_duplicate_predictions_match_truth_only_once() -> None:
    predictions = np.array([[0.0, 0, 0], [0.1, 0, 0], [4.0, 0, 0]])
    truth = np.array([[0.0, 0, 0], [4.0, 0, 0]])
    result = match_points_one_to_one(predictions, truth, radius_angstrom=0.5)
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.metrics()["precision"] == 2 / 3


def test_empty_point_sets_have_well_defined_counts() -> None:
    result = match_points_one_to_one(np.empty((0, 3)), np.array([[0, 0, 0]]), radius_angstrom=1)
    assert result.true_positives == 0
    assert result.false_negatives == 1
    assert result.metrics()["recall"] == 0

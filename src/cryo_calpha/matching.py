from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class MatchResult:
    matched_prediction_indices: NDArray[np.int64]
    matched_truth_indices: NDArray[np.int64]
    distances_angstrom: NDArray[np.float64]
    prediction_count: int
    truth_count: int

    @property
    def true_positives(self) -> int:
        return len(self.distances_angstrom)

    @property
    def false_positives(self) -> int:
        return self.prediction_count - self.true_positives

    @property
    def false_negatives(self) -> int:
        return self.truth_count - self.true_positives

    def metrics(self) -> dict[str, float | int]:
        tp, fp, fn = self.true_positives, self.false_positives, self.false_negatives
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        distances = self.distances_angstrom
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_distance_angstrom": float(distances.mean()) if len(distances) else float("nan"),
            "median_distance_angstrom": float(np.median(distances))
            if len(distances)
            else float("nan"),
            "rmse_angstrom": float(np.sqrt(np.mean(distances**2)))
            if len(distances)
            else float("nan"),
            "p90_distance_angstrom": float(np.percentile(distances, 90))
            if len(distances)
            else float("nan"),
        }


def _point_array(points: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape [N, 3]")
    return array


def match_points_one_to_one(
    predictions_xyz: ArrayLike,
    truth_xyz: ArrayLike,
    *,
    radius_angstrom: float,
) -> MatchResult:
    predictions = _point_array(predictions_xyz, "predictions_xyz")
    truth = _point_array(truth_xyz, "truth_xyz")
    if radius_angstrom <= 0:
        raise ValueError("radius_angstrom must be positive")
    prediction_count, truth_count = len(predictions), len(truth)
    if not prediction_count or not truth_count:
        return MatchResult(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            prediction_count,
            truth_count,
        )

    distances = cdist(predictions, truth)
    size = prediction_count + truth_count
    invalid_cost = radius_angstrom * 4.0 + 1.0
    unmatched_cost = radius_angstrom / 2.0 + np.finfo(np.float64).eps * 16
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:prediction_count, :truth_count] = np.where(
        distances <= radius_angstrom, distances, invalid_cost
    )
    cost[:prediction_count, truth_count:] = unmatched_cost
    cost[prediction_count:, :truth_count] = unmatched_cost
    rows, columns = linear_sum_assignment(cost)
    valid = (
        (rows < prediction_count)
        & (columns < truth_count)
        & (
            distances[rows.clip(max=prediction_count - 1), columns.clip(max=truth_count - 1)]
            <= radius_angstrom
        )
    )
    matched_predictions = rows[valid].astype(np.int64)
    matched_truth = columns[valid].astype(np.int64)
    matched_distances = distances[matched_predictions, matched_truth]
    return MatchResult(
        matched_predictions,
        matched_truth,
        matched_distances,
        prediction_count,
        truth_count,
    )

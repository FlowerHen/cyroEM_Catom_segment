from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from .matching import MatchResult, match_points_one_to_one
from .peaks import PeakSet, extract_calpha_peaks


@dataclass(frozen=True)
class EvaluationResult:
    peaks: PeakSet
    matches: MatchResult

    @property
    def metrics(self) -> dict[str, float | int]:
        metrics = self.matches.metrics()
        metrics["prediction_count"] = len(self.peaks.coordinates_xyz)
        metrics["truth_count"] = self.matches.truth_count
        return metrics


def evaluate_probability_map(
    probability_zyx: ArrayLike,
    truth_xyz: ArrayLike,
    *,
    origin_xyz: ArrayLike,
    voxel_size_xyz: ArrayLike,
    probability_threshold: float,
    nms_radius_angstrom: float,
    match_radius_angstrom: float,
) -> EvaluationResult:
    peaks = extract_calpha_peaks(
        probability_zyx,
        origin_xyz=origin_xyz,
        voxel_size_xyz=voxel_size_xyz,
        threshold=probability_threshold,
        nms_radius_angstrom=nms_radius_angstrom,
    )
    matches = match_points_one_to_one(
        peaks.coordinates_xyz, truth_xyz, radius_angstrom=match_radius_angstrom
    )
    return EvaluationResult(peaks, matches)

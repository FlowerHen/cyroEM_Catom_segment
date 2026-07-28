import numpy as np

from cryo_calpha.peaks import extract_calpha_peaks


def test_peak_extraction_and_nms_use_world_distance() -> None:
    probability = np.zeros((9, 9, 9), dtype=np.float32)
    probability[2, 3, 4] = 0.9
    probability[2, 3, 5] = 0.8
    probability[7, 7, 7] = 0.95
    peaks = extract_calpha_peaks(
        probability,
        origin_xyz=[10, 20, 30],
        voxel_size_xyz=[1, 1, 1],
        threshold=0.5,
        nms_radius_angstrom=1.5,
    )
    assert len(peaks.scores) == 2
    assert peaks.scores[0] == np.float32(0.95)

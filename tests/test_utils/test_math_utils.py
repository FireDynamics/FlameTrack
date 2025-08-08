import pytest

from flametrack.utils.math_utils import compute_target_ratio


def test_compute_target_ratio_normal():
    # height = 2.0, width = 4.0 → height / width = 0.5
    assert compute_target_ratio(4.0, 2.0) == pytest.approx(0.5)


def test_compute_target_ratio_zero_width():
    # width = 0.0 → Fallback auf 1.0
    assert compute_target_ratio(0.0, 3.0) == 1.0


import numpy as np
import pytest

from flametrack.utils.math_utils import estimate_resolution_from_points


def test_estimate_resolution_from_points_typical_case():
    # Rechteckige Struktur: 100 Pixel Breite, 200 Pixel Höhe
    p0 = np.array([0, 0])  # Top-Left
    p1 = np.array([100, 0])  # Top-Right
    p3 = np.array([0, 200])  # Bottom-Left

    plate_width_mm = 1000.0
    plate_height_mm = 2000.0
    result = estimate_resolution_from_points(
        p0, p1, p3, plate_width_mm, plate_height_mm
    )

    assert pytest.approx(result["mm_per_px_width"], rel=1e-6) == 10.0
    assert pytest.approx(result["mm_per_px_height"], rel=1e-6) == 10.0
    assert pytest.approx(result["error_mm_width"], rel=1e-6) == 5.0
    assert pytest.approx(result["error_mm_height"], rel=1e-6) == 5.0
    assert result["assumed_pixel_error"] == 0.5


def test_estimate_resolution_from_points_invalid_zero_dimension():
    p0 = np.array([0, 0])
    p1 = np.array([0, 0])  # Gleicher Punkt → Breite = 0
    p3 = np.array([0, 100])

    with pytest.raises(ValueError, match="Degenerate rectangle"):
        estimate_resolution_from_points(p0, p1, p3, 1000, 1000)

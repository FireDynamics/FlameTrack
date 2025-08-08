import pytest

from flametrack.analysis.IR_analysis import get_dewarp_parameters


def test_get_dewarp_parameters_valid():
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    target_pixels_width = 200
    target_pixels_height = 200
    result = get_dewarp_parameters(corners, target_pixels_width, target_pixels_height)
    assert result is not None
    assert isinstance(result, dict)
    assert "transformation_matrix" in result
    assert "target_pixels_width" in result
    assert "target_pixels_height" in result
    assert "target_ratio" in result
    assert result["target_pixels_width"] == target_pixels_width
    assert result["target_pixels_height"] == target_pixels_height
    assert result["target_ratio"] == target_pixels_height / target_pixels_width


def test_get_dewarp_parameters_target_ratio():
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    target_ratio = 1.0
    result = get_dewarp_parameters(corners, target_ratio=target_ratio)
    assert result is not None
    assert isinstance(result, dict)
    assert result["target_pixels_width"] == result["target_pixels_height"]
    assert result["target_ratio"] == 1.0


def test_get_dewarp_parameters_missing_arguments():
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    with pytest.raises(
        ValueError, match="Either plate dimensions or target ratio must be provided"
    ):
        get_dewarp_parameters(corners)


def test_get_dewarp_parameters_with_real_plate_dimensions():
    corners = [[0, 0], [0.2, 0], [0.2, 0.1], [0, 0.1]]
    plate_width_m = 0.2
    plate_height_m = 0.1
    pixels_per_mm = 1000  # 1000 px/m = 1 px/mm
    result = get_dewarp_parameters(
        corners,
        plate_width_m=plate_width_m,
        plate_height_m=plate_height_m,
        pixels_per_millimeter=pixels_per_mm,
    )
    assert isinstance(result, dict)
    assert result["target_pixels_width"] == int(plate_width_m * pixels_per_mm)
    assert result["target_pixels_height"] == int(plate_height_m * pixels_per_mm)
    assert "transformation_matrix" in result

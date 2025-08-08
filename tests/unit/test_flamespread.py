import numpy as np
import pytest

from flametrack.analysis import flamespread


def test_find_peaks_in_gradient_detects_peak():
    y = np.array([0, 1, 3, 7, 4, 2, 1])
    peaks = flamespread.find_peaks_in_gradient(
        y, min_height=0.5, min_distance=None, min_width=None
    )
    assert peaks.tolist() == [4]  # negativer Gradient-Peak bei Index 4


def test_right_most_point_over_threshold():
    y = np.array([0, 1, 2, 3, 4, 5])
    result = flamespread.right_most_point_over_threshold(y, threshold=2)
    assert result == 5


def test_left_most_point_over_threshold():
    y = np.array([0, 0, 1, 2, 3])
    result = flamespread.left_most_point_over_threshold(y, threshold=1)
    assert result == 3


def test_right_most_peak_returns_last_peak():
    y = np.array([0, 2, 0, 1, 3, 0])
    result = flamespread.right_most_peak(
        y, min_height=None, min_distance=10, min_width=None
    )
    assert isinstance(result, (int, np.integer))
    assert result == 2  # letzter Peak im negativen Gradienten


def test_highest_peak_returns_largest_gradient_peak():
    y = np.array([0, 10, 30, 10, 15, 0, 4])
    result = flamespread.highest_peak(y, min_height=0.5)
    assert isinstance(result, (int, np.integer))
    assert result == 3  # stärkste Steigung bei Index 2


def test_highest_peak_to_lowest_value_finds_expected_peak():
    y = np.array([10, 15, 30, 8, 3, 2, 1])
    result = flamespread.highest_peak_to_lowest_value(
        y,
        min_distance=1,
        min_height=0.5,
        min_width=1,
        high_val=10,
        low_val=5,
    )
    assert isinstance(result, (int, np.integer))
    assert result == 3


def test_highest_peak_to_lowest_value_with_direction_weighting():
    y = np.array([10, 15, 30, 8, 3, 2, 1])
    result = flamespread.highest_peak_to_lowest_value(
        y,
        min_distance=1,
        min_height=0.5,
        min_width=1,
        high_val=10,
        low_val=5,
        direction_weighting=1.0,
        previous_peak=2,
        previous_velocity=6,
    )
    assert isinstance(result, (int, np.integer))
    assert result == 3


def test_highest_peak_to_lowest_value_without_previous_peak():
    y = np.array([10, 15, 30, 8, 3, 2, 1])
    result = flamespread.highest_peak_to_lowest_value(
        y,
        min_distance=1,
        min_height=0.5,
        min_width=1,
        high_val=10,
        low_val=5,
        previous_peak=None,
    )
    assert isinstance(result, (int, np.integer))
    assert result == 3


def test_highest_peak_to_lowest_value_with_low_velocity():
    y = np.array([10, 15, 30, 8, 3, 2, 1])
    result = flamespread.highest_peak_to_lowest_value(
        y,
        min_distance=1,
        min_height=0.5,
        min_width=1,
        high_val=10,
        low_val=5,
        direction_weighting=1.0,
        previous_peak=2,
        previous_velocity=2,
    )
    assert isinstance(result, (int, np.integer))
    assert result == 3


def test_highest_peak_to_lowest_value_direction_factor_zero():
    y = np.array([10, 15, 30, 8, 3, 2, 1])
    result = flamespread.highest_peak_to_lowest_value(
        y,
        min_distance=1,
        min_height=0.5,
        min_width=1,
        high_val=10,
        low_val=5,
        direction_weighting=1000.0,
        previous_peak=2,
        previous_velocity=6,
    )
    assert isinstance(result, (int, np.integer))
    assert result == 3


def test_band_filter_clips_correctly():
    arr = np.array([[50, 100], [200, 300]])
    result = flamespread.band_filter(arr, low=75, high=250)
    expected = np.array([[75, 100], [200, 250]])
    assert np.array_equal(result, expected)


def test_calculate_edge_data_runs():
    data = np.zeros((5, 5, 3))
    for i in range(3):
        data[:, :, i] = i * 50 + np.eye(5)

    result = flamespread.calculate_edge_data(
        data,
        flamespread.right_most_point_over_threshold,
        custom_filter=lambda x: x,
    )

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(row, list) for row in result)
    assert all(len(row) == 5 for row in result)

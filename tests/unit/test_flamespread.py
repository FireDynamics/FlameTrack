import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from flametrack.analysis import flamespread

matplotlib.use("Agg")


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


# ---------------------------------------------------------------------------
# Cluster-based edge functions (lines 86–113)
# ---------------------------------------------------------------------------


def test_left_edge_of_rightmost_cluster_no_match():
    """Returns len(y) when no pixel exceeds threshold."""
    y = np.zeros(10)
    result = flamespread.left_edge_of_rightmost_cluster(y, threshold=1)
    assert result == len(y)


def test_left_edge_of_rightmost_cluster_single_pixel():
    y = np.array([0, 0, 5, 0, 0])
    result = flamespread.left_edge_of_rightmost_cluster(y, threshold=1)
    assert result == 2


def test_left_edge_of_rightmost_cluster_cluster():
    """Leftmost pixel of rightmost hot cluster."""
    y = np.array([0, 5, 5, 5, 0, 0])
    result = flamespread.left_edge_of_rightmost_cluster(y, threshold=1)
    assert result == 1


def test_right_edge_of_leftmost_cluster_no_match():
    """Returns 0 when no pixel exceeds threshold."""
    y = np.zeros(10)
    result = flamespread.right_edge_of_leftmost_cluster(y, threshold=1)
    assert result == 0


def test_right_edge_of_leftmost_cluster_cluster():
    """Rightmost pixel of leftmost hot cluster."""
    y = np.array([0, 5, 5, 5, 0, 0])
    result = flamespread.right_edge_of_leftmost_cluster(y, threshold=1)
    assert result == 3


def test_right_edge_of_leftmost_cluster_single_pixel():
    y = np.array([0, 0, 7, 0, 0])
    result = flamespread.right_edge_of_leftmost_cluster(y, threshold=1)
    assert result == 2


# ---------------------------------------------------------------------------
# highest_peak_to_lowest_value → no-match branch (line 212)
# ---------------------------------------------------------------------------


def test_highest_peak_to_lowest_value_no_peaks_returns_zero():
    """When no valid peak passes the high_val/low_val filter → return 0."""
    # Flat signal: no peaks at all
    y = np.ones(20) * 10
    result = flamespread.highest_peak_to_lowest_value(
        y,
        min_distance=1,
        min_height=0.5,
        min_width=1,
        high_val=50,  # require val >= 50 before peak → never satisfied
        low_val=100,
    )
    assert result == 0


# ---------------------------------------------------------------------------
# band_filter edge cases (lines 488, 490)
# ---------------------------------------------------------------------------


def test_band_filter_low_none():
    arr = np.array([1.0, 5.0, 10.0])
    result = flamespread.band_filter(arr, low=None, high=7.0)
    assert result[2] == pytest.approx(7.0)
    assert result[0] == pytest.approx(1.0)


def test_band_filter_high_none():
    arr = np.array([1.0, 5.0, 10.0])
    result = flamespread.band_filter(arr, low=3.0, high=None)
    assert result[0] == pytest.approx(3.0)
    assert result[2] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# calculate_edge_data with Otsu masking (lines 271–294)
# ---------------------------------------------------------------------------


def test_calculate_edge_data_with_otsu():
    """Runs calculate_edge_data with use_otsu_masking=True, no error."""
    rng = np.random.default_rng(42)
    data = rng.uniform(100, 400, (8, 8, 4)).astype(np.float32)

    result = flamespread.calculate_edge_data(
        data,
        flamespread.right_most_point_over_threshold,
        custom_filter=lambda x: x,
        use_otsu_masking=True,
    )
    assert len(result) == 4
    assert all(len(row) == 8 for row in result)


# ---------------------------------------------------------------------------
# Visualization functions (lines 506–582)
# ---------------------------------------------------------------------------


def test_plot_edge_runs(monkeypatch):
    frame = np.random.randint(0, 255, (5, 10)).astype(np.float32)
    flamespread.plot_edge(frame, find_edge_point=lambda y: 0)


def test_show_flame_spread_returns_fig_ax():
    edge_results = np.random.randint(0, 10, (20, 5))
    fig, ax = flamespread.show_flame_spread(edge_results, y_coord=2)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_show_flame_contour_returns_fig_ax():
    data = np.random.rand(5, 5, 10).astype(np.float32)
    edge_results = np.random.randint(0, 5, (10, 5))
    fig, ax = flamespread.show_flame_contour(data, edge_results, frame=3)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_show_flame_spread_velocity_returns_fig_ax():
    edge_results = np.tile(np.arange(20), (5, 1)).T  # shape (20, 5)
    fig, ax = flamespread.show_flame_spread_velocity(edge_results, y_coord=1)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

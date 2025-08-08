import pytest

from flametrack.gui.plotting_utils import sort_corner_points  # <- ggf. anpassen


def test_sort_corner_points_with_valid_input():
    input_points = [
        (200, 310),  # unten rechts
        (150, 100),  # oben mitte
        (100, 110),  # oben links
        (200, 110),  # oben rechts
        (150, 300),  # unten mitte
        (100, 310),  # unten links
    ]

    sorted_points = sort_corner_points(input_points, direction="clockwise")

    # Test: Es werden exakt 6 Punkte zurückgegeben
    assert len(sorted_points) == 6

    # Test: Keine Punkte verloren oder doppelt
    assert set(sorted_points) == set(input_points)

    # Test: Der erste Punkt ist der mit kleinstem X (links), bei Gleichstand kleinstem Y
    expected_first = min(input_points, key=lambda p: (p[0], p[1]))
    assert sorted_points[0] == expected_first


def test_sort_corner_points_with_invalid_input():
    # Weniger als 6 Punkte
    with pytest.raises(ValueError):
        sort_corner_points([(0, 0), (1, 1), (2, 2)])

    # Mehr als 6 Punkte
    with pytest.raises(ValueError):
        sort_corner_points([(x, x) for x in range(7)])

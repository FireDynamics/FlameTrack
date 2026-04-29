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


def test_sort_corner_points_lfs_upright():
    """LFS with a roughly upright rectangle — must return [TL, TR, BR, BL]."""
    tl = (50, 50)
    tr = (250, 50)
    br = (250, 150)
    bl = (50, 150)
    # Pass in scrambled order
    result = sort_corner_points(
        [br, tl, bl, tr], experiment_type="Lateral Flame Spread"
    )
    assert result[0] == tl, f"Expected TL={tl}, got {result[0]}"
    assert result[1] == tr, f"Expected TR={tr}, got {result[1]}"
    assert result[2] == br, f"Expected BR={br}, got {result[2]}"
    assert result[3] == bl, f"Expected BL={bl}, got {result[3]}"


def test_sort_corner_points_lfs_elongated_tilted():
    """
    LFS with a highly elongated, steeply tilted plate — the case where
    angle-based sorting fails.  Plate goes from lower-left to upper-right,
    like the IR image of a lateral flame spread experiment.
    """
    tl = (100, 265)  # left edge, upper point
    bl = (130, 285)  # left edge, lower point
    tr = (1000, 105)  # right edge, upper point
    br = (1060, 130)  # right edge, lower point
    # Pass in scrambled order
    result = sort_corner_points(
        [br, tl, bl, tr], experiment_type="Lateral Flame Spread"
    )
    assert result[0] == tl, f"Expected TL={tl}, got {result[0]}"
    assert result[1] == tr, f"Expected TR={tr}, got {result[1]}"
    assert result[2] == br, f"Expected BR={br}, got {result[2]}"
    assert result[3] == bl, f"Expected BL={bl}, got {result[3]}"


def test_sort_corner_points_with_invalid_input():
    # Weniger als 6 Punkte
    with pytest.raises(ValueError):
        sort_corner_points([(0, 0), (1, 1), (2, 2)])

    # Mehr als 6 Punkte
    with pytest.raises(ValueError):
        sort_corner_points([(x, x) for x in range(7)])

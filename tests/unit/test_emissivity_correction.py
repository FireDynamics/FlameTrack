import numpy as np
import pytest

from flametrack.analysis.emissivity_correction import (
    apply_region_corrections,
    correct_temperature,
    region_mask,
)
from flametrack.analysis.region import Region, RegionShape


def test_emissivity_one_is_identity():
    apparent = np.array([20.0, 50.0, 100.0])
    corrected = correct_temperature(apparent, emissivity=1.0)
    np.testing.assert_allclose(corrected, apparent, atol=1e-6)


def test_lower_emissivity_increases_corrected_temperature():
    apparent = np.array([50.0])
    corrected = correct_temperature(apparent, emissivity=0.8)
    print(f"apparent={apparent[0]}, corrected={corrected[0]}")
    assert corrected[0] > apparent[0]


def test_region_mask_rectangle_covers_expected_area():
    points = [(2, 2), (8, 2), (8, 8), (2, 8)]
    mask = region_mask(points, (10, 10))
    assert mask.shape == (10, 10)
    assert mask[5, 5]
    assert not mask[0, 0]
    assert not mask[9, 9]


def test_region_mask_polygon():
    points = [(1, 1), (5, 1), (3, 5)]
    mask = region_mask(points, (10, 10))
    assert mask.any()
    assert not mask[9, 9]


def test_apply_region_corrections_only_touches_masked_pixels():
    frame = np.full((10, 10), 50.0)
    region = Region(
        region_id="r1",
        name="hot spot",
        shape=RegionShape.RECTANGLE,
        points=[(2, 2), (5, 2), (5, 5), (2, 5)],
        emissivity=0.7,
    )
    corrected = apply_region_corrections(frame, [region])
    mask = region_mask(region.points, frame.shape)

    print("inside region: ", corrected[mask][:5])
    print("outside region: ", corrected[~mask][:5])

    assert np.all(corrected[mask] != frame[mask])
    assert np.all(corrected[~mask] == frame[~mask])


def test_apply_region_corrections_multiple_non_overlapping_regions():
    frame = np.full((20, 20), 40.0)
    region_a = Region(
        "a",
        "A",
        RegionShape.RECTANGLE,
        [(0, 0), (5, 0), (5, 5), (0, 5)],
        emissivity=0.5,
    )
    region_b = Region(
        "b",
        "B",
        RegionShape.RECTANGLE,
        [(10, 10), (15, 10), (15, 15), (10, 15)],
        emissivity=0.9,
    )

    corrected = apply_region_corrections(frame, [region_a, region_b])

    mask_a = region_mask(region_a.points, frame.shape)
    mask_b = region_mask(region_b.points, frame.shape)
    outside = ~(mask_a | mask_b)

    assert np.all(corrected[mask_a] != frame[mask_a])
    assert np.all(corrected[mask_b] != frame[mask_b])
    assert np.all(corrected[outside] == frame[outside])


def test_apply_region_corrections_no_regions_returns_unchanged_copy():
    frame = np.full((5, 5), 30.0)
    corrected = apply_region_corrections(frame, [])
    np.testing.assert_array_equal(corrected, frame)
    assert corrected is not frame


def test_apply_region_corrections_empty_mask_is_skipped():
    frame = np.full((5, 5), 30.0)
    region = Region(
        "r",
        "outside",
        RegionShape.RECTANGLE,
        [(-10, -10), (-8, -10), (-8, -8), (-10, -8)],
        emissivity=0.5,
    )
    corrected = apply_region_corrections(frame, [region])
    np.testing.assert_array_equal(corrected, frame)

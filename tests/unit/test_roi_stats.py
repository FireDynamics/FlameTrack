import h5py
import numpy as np
import pytest

from flametrack.analysis.region import Region, RegionShape
from flametrack.analysis.roi_stats import (
    export_time_series_csv,
    extract_time_series,
    region_stats,
    region_stats_all,
    save_time_series,
)


@pytest.fixture
def region():
    return Region(
        region_id="r1",
        name="test",
        shape=RegionShape.RECTANGLE,
        points=[(2, 2), (5, 2), (5, 5), (2, 5)],
    )


def test_region_stats_basic(region):
    frame = np.zeros((10, 10))
    frame[2:6, 2] = 1.0
    frame[2:6, 3] = 2.0
    frame[2:6, 4] = 3.0
    frame[2:6, 5] = 4.0
    stats = region_stats(frame, region)

    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)


def test_region_stats_empty_mask_returns_nan():
    region = Region(
        "r",
        "outside",
        RegionShape.RECTANGLE,
        [(-10, -10), (-8, -10), (-8, -8), (-10, -8)],
    )
    frame = np.zeros((5, 5))
    stats = region_stats(frame, region)
    assert np.isnan(stats["min"])
    assert np.isnan(stats["max"])
    assert np.isnan(stats["mean"])


def test_region_stats_all_multiple_regions():
    region_a = Region("a", "A", RegionShape.RECTANGLE, [(0, 0), (3, 0), (3, 3), (0, 3)])
    region_b = Region("b", "B", RegionShape.RECTANGLE, [(5, 5), (8, 5), (8, 8), (5, 8)])
    frame = np.full((10, 10), 5.0)
    stats = region_stats_all(frame, [region_a, region_b])
    assert set(stats.keys()) == {"a", "b"}


def test_extract_time_series_uses_raw_data_when_no_corrected(region):
    with h5py.File("test_raw.h5", "w", driver="core", backing_store=False) as f:
        grp = f.create_group("dewarped_data")
        data = np.random.default_rng(0).random((10, 10, 4))
        grp.create_dataset("data", data=data)

        series = extract_time_series(grp, [region])

        assert len(series["r1"]["mean"]) == 4
        expected_mean_0 = data[2:6, 2:6, 0].mean()
        assert series["r1"]["mean"][0] == pytest.approx(expected_mean_0, rel=1e-4)


def test_extract_time_series_prefers_corrected_data(region):
    with h5py.File("test_corrected.h5", "w", driver="core", backing_store=False) as f:
        grp = f.create_group("dewarped_data")
        raw = np.zeros((10, 10, 2))
        corrected = np.ones((10, 10, 2)) * 99.0
        grp.create_dataset("data", data=raw)
        grp.create_dataset("corrected_data", data=corrected)

        series = extract_time_series(grp, [region])

        assert series["r1"]["mean"][0] == pytest.approx(99.0)


def test_save_time_series_writes_hdf5_groups(region):
    with h5py.File("test_save.h5", "w", driver="core", backing_store=False) as f:
        grp = f.create_group("dewarped_data")
        data = np.full((10, 10, 3), 7.0)
        grp.create_dataset("data", data=data)

        series = extract_time_series(grp, [region])
        save_time_series(grp, series)

        assert "roi_stats" in grp
        assert "r1" in grp["roi_stats"]
        assert grp["roi_stats"]["r1"]["mean"].shape == (3,)


def test_export_time_series_csv(tmp_path):
    series = {
        "r1": {"min": [1.0, 2.0], "max": [3.0, 4.0], "mean": [2.0, 3.0]},
    }
    path = tmp_path / "out.csv"
    export_time_series_csv(series, str(path))

    content = path.read_text()
    lines = content.strip().split("\n")
    assert lines[0] == "frame,region_id,min,max,mean"
    assert len(lines) == 3

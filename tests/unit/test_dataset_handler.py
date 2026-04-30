"""Tests for flametrack.analysis.dataset_handler."""

from __future__ import annotations

import configparser
import os

import h5py
import numpy as np
import pytest

import flametrack.analysis.dataset_handler as dh
import flametrack.analysis.user_config as uc

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(tmp_path, monkeypatch):
    """Point user_config and dataset_handler at a temporary directory tree."""
    ini_path = str(tmp_path / "config.ini")
    cfg = configparser.ConfigParser()
    cfg["DEFAULT"] = {
        "experiment_folder": str(tmp_path),
        "IR_folder": "exported_data/",
        "processed_data": "processed_data/",
    }
    with open(ini_path, "w") as f:
        cfg.write(f)
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "DEFAULT")


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    """Ensure module-level globals are reset between tests."""
    monkeypatch.setattr(dh, "HDF_FILE", None)
    monkeypatch.setattr(dh, "LOADED_FILE_PATH", None)
    yield
    # Close any open file handle to avoid ResourceWarning
    if dh.HDF_FILE is not None:
        try:
            dh.HDF_FILE.close()
        except Exception:
            pass
    dh.HDF_FILE = None
    dh.LOADED_FILE_PATH = None


# ---------------------------------------------------------------------------
# create_h5_file
# ---------------------------------------------------------------------------


def test_create_h5_file_with_filename(tmp_path):
    """create_h5_file(filename=...) creates the file and sets global state."""
    h5_path = str(tmp_path / "test.h5")
    f = dh.create_h5_file(filename=h5_path)
    try:
        assert os.path.exists(h5_path)
        assert f.attrs["file_version"] == "1.0"
        assert "flametrack_version" in f.attrs
        assert "flametrack_commit" in f.attrs
        assert dh.HDF_FILE is f
        assert dh.LOADED_FILE_PATH == h5_path
    finally:
        f.close()
        dh.HDF_FILE = None


def test_create_h5_file_no_args_raises():
    """create_h5_file() without exp_name and filename raises ValueError."""
    with pytest.raises(ValueError, match="exp_name"):
        dh.create_h5_file()


def test_create_h5_file_creates_parent_dirs(tmp_path, monkeypatch):
    """Parent directories are created automatically."""
    _make_config(tmp_path, monkeypatch)
    exp_name = "test_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    assert not proc_dir.exists()

    f = dh.create_h5_file(exp_name=exp_name)
    try:
        assert proc_dir.exists()
        assert f is not None
    finally:
        f.close()
        dh.HDF_FILE = None


# ---------------------------------------------------------------------------
# init_h5_for_experiment / assert_h5_schema
# ---------------------------------------------------------------------------


def test_init_h5_lateral_flame_spread(tmp_path):
    h5_path = str(tmp_path / "lfs.h5")
    with h5py.File(h5_path, "w") as f:
        dh.init_h5_for_experiment(f, "Lateral Flame Spread")
        assert "dewarped_data" in f
        assert "edge_results" in f


def test_init_h5_room_corner(tmp_path):
    h5_path = str(tmp_path / "rce.h5")
    with h5py.File(h5_path, "w") as f:
        dh.init_h5_for_experiment(f, "Room Corner")
        assert "dewarped_data_left" in f
        assert "dewarped_data_right" in f


def test_init_h5_unknown_type_raises(tmp_path):
    h5_path = str(tmp_path / "bad.h5")
    with h5py.File(h5_path, "w") as f:
        with pytest.raises(ValueError, match="Unknown experiment type"):
            dh.init_h5_for_experiment(f, "Unknown")


def test_assert_h5_schema_passes_when_groups_exist(tmp_path):
    h5_path = str(tmp_path / "ok.h5")
    with h5py.File(h5_path, "w") as f:
        f.require_group("dewarped_data")
        dh.assert_h5_schema(f, "Lateral Flame Spread")  # should not raise


def test_assert_h5_schema_raises_when_group_missing(tmp_path):
    h5_path = str(tmp_path / "missing.h5")
    with h5py.File(h5_path, "w") as f:
        with pytest.raises(RuntimeError, match="Missing groups"):
            dh.assert_h5_schema(f, "Lateral Flame Spread")


def test_assert_h5_schema_room_corner_raises_when_incomplete(tmp_path):
    h5_path = str(tmp_path / "partial.h5")
    with h5py.File(h5_path, "w") as f:
        f.require_group("dewarped_data_left")  # only left, no right
        with pytest.raises(RuntimeError, match="Missing groups"):
            dh.assert_h5_schema(f, "Room Corner")


# ---------------------------------------------------------------------------
# get_h5_file_path
# ---------------------------------------------------------------------------


def test_get_h5_file_path_without_left(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    path = dh.get_h5_file_path("my_exp")
    assert path.endswith("my_exp_results.h5")
    assert "processed_data" in path


def test_get_h5_file_path_with_left(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    path = dh.get_h5_file_path("my_exp", left=True)
    assert path.endswith("my_exp_results_left.h5")


# ---------------------------------------------------------------------------
# get_file / close_file
# ---------------------------------------------------------------------------


def test_get_file_opens_and_reuses(tmp_path, monkeypatch):
    """get_file reuses an open handle when the same path is requested."""
    _make_config(tmp_path, monkeypatch)
    exp_name = "reuse_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    proc_dir.mkdir(parents=True)
    h5_path = str(proc_dir / f"{exp_name}_results.h5")

    # Pre-create the file so get_file can open it in read mode
    with h5py.File(h5_path, "w") as f:
        f.attrs["test"] = "hello"

    f1 = dh.get_file(exp_name)
    f2 = dh.get_file(exp_name)
    assert f1 is f2
    f1.close()
    dh.HDF_FILE = None


def test_get_file_reopens_for_different_path(tmp_path, monkeypatch):
    """get_file opens a new handle when a different exp_name is requested."""
    _make_config(tmp_path, monkeypatch)
    for name in ("exp_a", "exp_b"):
        proc_dir = tmp_path / name / "processed_data"
        proc_dir.mkdir(parents=True)
        with h5py.File(str(proc_dir / f"{name}_results.h5"), "w") as f:
            f.attrs["name"] = name

    fa = dh.get_file("exp_a")
    assert dh.LOADED_FILE_PATH is not None and "exp_a" in dh.LOADED_FILE_PATH

    fb = dh.get_file("exp_b")
    assert dh.LOADED_FILE_PATH is not None and "exp_b" in dh.LOADED_FILE_PATH
    assert fa is not fb
    fb.close()
    dh.HDF_FILE = None


def test_get_file_mode_w_raises():
    with pytest.raises(ValueError, match="create_h5_file"):
        dh.get_file("any_exp", mode="w")  # type: ignore[arg-type]


def test_close_file_clears_globals(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    exp_name = "close_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    proc_dir.mkdir(parents=True)
    h5_path = str(proc_dir / f"{exp_name}_results.h5")
    with h5py.File(h5_path, "w") as f:
        f.attrs["x"] = 1

    dh.get_file(exp_name)
    assert dh.HDF_FILE is not None
    dh.close_file()
    assert dh.HDF_FILE is None
    assert dh.LOADED_FILE_PATH is None


def test_close_file_when_already_none():
    """close_file() is a no-op when HDF_FILE is already None."""
    assert dh.HDF_FILE is None
    dh.close_file()  # should not raise


# ---------------------------------------------------------------------------
# save_edge_results / get_data / get_edge_results
# ---------------------------------------------------------------------------


def test_save_and_get_edge_results(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    exp_name = "edge_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    proc_dir.mkdir(parents=True)
    h5_path = str(proc_dir / f"{exp_name}_results.h5")

    # Create the HDF5 file with the edge_results group
    edge_data = np.arange(12, dtype=np.int32).reshape(3, 4)
    with h5py.File(h5_path, "w") as f:
        grp = f.require_group("edge_results")
        grp.create_dataset("data", data=edge_data)

    # save_edge_results should overwrite
    new_edge = np.zeros((3, 4), dtype=np.int32)
    dh.save_edge_results(exp_name, new_edge)

    result = dh.get_edge_results(exp_name)
    assert np.array_equal(result[:], new_edge)
    dh.close_file()


def test_save_edge_results_creates_dataset_when_absent(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    exp_name = "new_edge_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    proc_dir.mkdir(parents=True)
    h5_path = str(proc_dir / f"{exp_name}_results.h5")

    # Start with empty file
    with h5py.File(h5_path, "w") as _:
        pass

    edge_data = np.ones((5, 3), dtype=np.int32)
    dh.save_edge_results(exp_name, edge_data)

    with h5py.File(h5_path, "r") as f:
        assert "edge_results" in f
        assert np.array_equal(f["edge_results"]["data"][:], edge_data)


# ---------------------------------------------------------------------------
# get_dewarped_data / get_dewarped_metadata
# ---------------------------------------------------------------------------


def test_get_dewarped_data(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    exp_name = "dw_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    proc_dir.mkdir(parents=True)
    h5_path = str(proc_dir / f"{exp_name}_results.h5")

    dw_data = np.random.rand(4, 4, 3).astype(np.float32)
    with h5py.File(h5_path, "w") as f:
        grp = f.require_group("dewarped_data")
        grp.create_dataset("data", data=dw_data)
        grp.attrs["plate_width_mm"] = 500.0

    result = dh.get_dewarped_data(exp_name)
    assert result.shape == (4, 4, 3)
    dh.close_file()


def test_get_dewarped_metadata(tmp_path, monkeypatch):
    _make_config(tmp_path, monkeypatch)
    exp_name = "meta_exp"
    proc_dir = tmp_path / exp_name / "processed_data"
    proc_dir.mkdir(parents=True)
    h5_path = str(proc_dir / f"{exp_name}_results.h5")

    with h5py.File(h5_path, "w") as f:
        grp = f.require_group("dewarped_data")
        grp.attrs["plate_width_mm"] = 600.0
        grp.attrs["plate_height_mm"] = 400.0

    meta = dh.get_dewarped_metadata(exp_name)
    assert meta["plate_width_mm"] == pytest.approx(600.0)
    assert meta["plate_height_mm"] == pytest.approx(400.0)
    dh.close_file()

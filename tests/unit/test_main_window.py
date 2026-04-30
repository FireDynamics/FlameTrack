from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from pytest import approx

from flametrack.analysis.flamespread import EDGE_METHOD_CATALOG
from flametrack.gui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Existing tests (keep)
# ---------------------------------------------------------------------------


def test_update_target_ratio_normal(mainwindow):
    mainwindow.ui.doubleSpinBox_plate_width.setValue(3.0)
    mainwindow.ui.doubleSpinBox_plate_height.setValue(2.0)
    mainwindow.update_target_ratio()
    expected_ratio = 2.0 / 3.0  # height / width
    assert mainwindow.target_ratio == approx(expected_ratio)


def test_update_target_ratio_zero_width(mainwindow):
    mainwindow.ui.doubleSpinBox_plate_width.setValue(0.0)
    mainwindow.ui.doubleSpinBox_plate_height.setValue(2.0)
    mainwindow.update_target_ratio()
    assert mainwindow.target_ratio == 1.0  # fallback


# ---------------------------------------------------------------------------
# _detect_datatype (static, no Qt needed)
# ---------------------------------------------------------------------------


def test_detect_datatype_ir(tmp_path):
    (tmp_path / "frame_0001.csv").write_text("data")

    assert MainWindow._detect_datatype(str(tmp_path)) == "IR"


def test_detect_datatype_video(tmp_path):
    (tmp_path / "recording.mp4").write_bytes(b"data")

    assert MainWindow._detect_datatype(str(tmp_path)) == "video"


def test_detect_datatype_picture(tmp_path):
    (tmp_path / "frame_001.jpg").write_bytes(b"data")

    assert MainWindow._detect_datatype(str(tmp_path)) == "picture"


def test_detect_datatype_subdirectory(tmp_path):
    """CSV in a subdirectory is found (one level deep scan)."""
    sub = tmp_path / "exported_data"
    sub.mkdir()
    (sub / "frame.csv").write_text("data")

    assert MainWindow._detect_datatype(str(tmp_path)) == "IR"


def test_detect_datatype_fallback(tmp_path):
    """Empty folder falls back to IR."""

    assert MainWindow._detect_datatype(str(tmp_path)) == "IR"


def test_detect_datatype_ir_wins_over_video(tmp_path):
    """CSV has higher priority than mp4."""
    (tmp_path / "frame.csv").write_text("data")
    (tmp_path / "video.mp4").write_bytes(b"data")

    assert MainWindow._detect_datatype(str(tmp_path)) == "IR"


# ---------------------------------------------------------------------------
# _read_plate_mm_from_h5_root_first (static)
# ---------------------------------------------------------------------------


def test_read_plate_mm_lfs_root(tmp_path):
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["plate_width_mm"] = 600.0
        f.attrs["plate_height_mm"] = 400.0

    with h5py.File(h5_path, "r") as f:
        w, h = MainWindow._read_plate_mm_from_h5_root_first(f)
    assert w == approx(600.0)
    assert h == approx(400.0)


def test_read_plate_mm_room_corner_root(tmp_path):
    h5_path = tmp_path / "rce.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["plate_width_mm_left"] = 500.0
        f.attrs["plate_height_mm_left"] = 300.0

    with h5py.File(h5_path, "r") as f:
        w, h = MainWindow._read_plate_mm_from_h5_root_first(f)
    assert w == approx(500.0)
    assert h == approx(300.0)


def test_read_plate_mm_room_corner_group(tmp_path):
    h5_path = tmp_path / "rce_grp.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.require_group("dewarped_data_left")
        grp.attrs["plate_width_mm"] = 450.0
        grp.attrs["plate_height_mm"] = 250.0

    with h5py.File(h5_path, "r") as f:
        w, h = MainWindow._read_plate_mm_from_h5_root_first(f)
    assert w == approx(450.0)
    assert h == approx(250.0)


def test_read_plate_mm_lfs_group(tmp_path):
    h5_path = tmp_path / "lfs_grp.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.require_group("dewarped_data")
        grp.attrs["plate_width_mm"] = 700.0
        grp.attrs["plate_height_mm"] = 350.0

    with h5py.File(h5_path, "r") as f:
        w, h = MainWindow._read_plate_mm_from_h5_root_first(f)
    assert w == approx(700.0)
    assert h == approx(350.0)


def test_read_plate_mm_nothing_returns_none(tmp_path):
    h5_path = tmp_path / "empty.h5"
    with h5py.File(h5_path, "w") as f:
        pass

    with h5py.File(h5_path, "r") as f:
        w, h = MainWindow._read_plate_mm_from_h5_root_first(f)
    assert w is None
    assert h is None


def test_read_plate_mm_right_fallback(tmp_path):
    """When only _right attributes exist, they should be used as fallback."""
    h5_path = tmp_path / "right_only.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["plate_width_mm_right"] = 550.0
        f.attrs["plate_height_mm_right"] = 320.0

    with h5py.File(h5_path, "r") as f:
        w, h = MainWindow._read_plate_mm_from_h5_root_first(f)
    assert w == approx(550.0)
    assert h == approx(320.0)


# ---------------------------------------------------------------------------
# _read_plate_mm (instance method, same logic via mainwindow fixture)
# ---------------------------------------------------------------------------


def test_read_plate_mm_instance_lfs_root(mainwindow, tmp_path):
    h5_path = tmp_path / "lfs.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["plate_width_mm"] = 800.0
        f.attrs["plate_height_mm"] = 500.0

    with h5py.File(h5_path, "r") as f:
        w, h = mainwindow._read_plate_mm(f)
    assert w == approx(800.0)
    assert h == approx(500.0)


def test_read_plate_mm_instance_nothing(mainwindow, tmp_path):
    h5_path = tmp_path / "empty.h5"
    with h5py.File(h5_path, "w") as f:
        pass

    with h5py.File(h5_path, "r") as f:
        w, h = mainwindow._read_plate_mm(f)
    assert w is None
    assert h is None


# ---------------------------------------------------------------------------
# _detect_experiment_type_from_h5
# ---------------------------------------------------------------------------


def test_detect_experiment_type_room_corner(mainwindow, tmp_path):
    h5_path = tmp_path / "rce.h5"
    with h5py.File(h5_path, "w") as f:
        f.require_group("dewarped_data_left")
        f.require_group("dewarped_data_right")

    with h5py.File(h5_path, "r") as f:
        mainwindow._detect_experiment_type_from_h5(f)

    assert mainwindow.experiment_type == "Room Corner"
    assert mainwindow.ui.comboBox_experiment_type.currentText() == "Room Corner"


def test_detect_experiment_type_lfs(mainwindow, tmp_path):
    h5_path = tmp_path / "lfs.h5"
    with h5py.File(h5_path, "w") as f:
        f.require_group("dewarped_data")

    with h5py.File(h5_path, "r") as f:
        mainwindow._detect_experiment_type_from_h5(f)

    assert mainwindow.experiment_type == "Lateral Flame Spread"


def test_detect_experiment_type_unknown_unchanged(mainwindow, tmp_path):
    """Empty HDF5 leaves experiment_type unchanged."""
    original_type = mainwindow.experiment_type
    h5_path = tmp_path / "empty.h5"
    with h5py.File(h5_path, "w") as f:
        pass

    with h5py.File(h5_path, "r") as f:
        mainwindow._detect_experiment_type_from_h5(f)

    assert mainwindow.experiment_type == original_type


# ---------------------------------------------------------------------------
# _apply_plate_mm_to_spinboxes
# ---------------------------------------------------------------------------


def test_apply_plate_mm_sets_spinboxes(mainwindow):
    mainwindow._apply_plate_mm_to_spinboxes(900.0, 600.0)
    assert mainwindow.ui.doubleSpinBox_plate_width.value() == approx(900.0)
    assert mainwindow.ui.doubleSpinBox_plate_height.value() == approx(600.0)


def test_apply_plate_mm_updates_ratio(mainwindow):
    mainwindow._apply_plate_mm_to_spinboxes(400.0, 200.0)
    assert mainwindow.target_ratio == approx(200.0 / 400.0)


def test_apply_plate_mm_partial_none(mainwindow):
    """Only width provided — height spinbox should remain unchanged."""
    mainwindow.ui.doubleSpinBox_plate_height.setValue(123.0)
    mainwindow._apply_plate_mm_to_spinboxes(500.0, None)
    assert mainwindow.ui.doubleSpinBox_plate_width.value() == approx(500.0)
    assert mainwindow.ui.doubleSpinBox_plate_height.value() == approx(123.0)


def test_apply_plate_mm_both_none(mainwindow):
    """Neither value → spinboxes unchanged, no exception."""
    mainwindow.ui.doubleSpinBox_plate_width.setValue(50.0)
    mainwindow._apply_plate_mm_to_spinboxes(None, None)
    assert mainwindow.ui.doubleSpinBox_plate_width.value() == approx(50.0)


# ---------------------------------------------------------------------------
# set_experiment_type
# ---------------------------------------------------------------------------


def test_set_experiment_type_room_corner(mainwindow):
    mainwindow.set_experiment_type("Room Corner")
    assert mainwindow.experiment_type == "Room Corner"
    assert mainwindow.required_points == 6


def test_set_experiment_type_lfs(mainwindow):
    mainwindow.set_experiment_type("Lateral Flame Spread")
    assert mainwindow.experiment_type == "Lateral Flame Spread"
    assert mainwindow.required_points == 4


def test_set_experiment_type_updates_experiment_object(mainwindow):
    """When an experiment is loaded, its type is updated too."""
    mock_exp = MagicMock()
    mainwindow.experiment = mock_exp
    mainwindow.set_experiment_type("Room Corner")
    assert mock_exp.experiment_type == "Room Corner"
    mainwindow.experiment = None  # cleanup


# ---------------------------------------------------------------------------
# update_flame_direction_visibility
# ---------------------------------------------------------------------------


def test_flame_direction_visible_for_lfs(mainwindow):
    # isVisible() requires the parent window to be shown too;
    # isHidden() reliably reflects setVisible() calls on the widget itself.
    mainwindow.ui.comboBox_experiment_type.setCurrentText("Lateral Flame Spread")
    mainwindow.update_flame_direction_visibility()
    assert not mainwindow.ui.comboBox_flame_direction.isHidden()


def test_flame_direction_hidden_for_room_corner(mainwindow):
    mainwindow.ui.comboBox_experiment_type.setCurrentText("Room Corner")
    mainwindow.update_flame_direction_visibility()
    assert mainwindow.ui.comboBox_flame_direction.isHidden()


# ---------------------------------------------------------------------------
# _edge_spec_for / _edge_method_for / _edge_threshold
# ---------------------------------------------------------------------------


def test_edge_spec_for_right_to_left(mainwindow):
    spec = mainwindow._edge_spec_for("right_to_left")
    assert spec.short_id == "leftmost_threshold"


def test_edge_spec_for_left_to_right(mainwindow):
    spec = mainwindow._edge_spec_for("left_to_right")
    assert spec.short_id == "rightmost_threshold"


def test_edge_spec_for_respects_manual_key(mainwindow):
    """When edge_method_key is set explicitly, it overrides direction."""
    mainwindow.edge_method_key = "rightmost_threshold"
    spec = mainwindow._edge_spec_for("right_to_left")
    assert spec.short_id == "rightmost_threshold"
    mainwindow.edge_method_key = None  # cleanup


def test_edge_method_for_returns_callable(mainwindow):
    fn = mainwindow._edge_method_for("right_to_left")
    assert callable(fn)
    # Should work on a real array
    y = np.array([0, 10, 200, 300, 100, 0], dtype=np.float32)
    result = fn(y)
    assert isinstance(result, (int, np.integer))


def test_edge_threshold_ir(mainwindow):
    mainwindow.datatype = "IR"
    thr = mainwindow._edge_threshold()
    spec = EDGE_METHOD_CATALOG["leftmost_threshold"]
    assert thr == approx(spec.default_threshold_ir)


def test_edge_threshold_image(mainwindow):
    mainwindow.datatype = "picture"
    thr = mainwindow._edge_threshold()
    spec = EDGE_METHOD_CATALOG["leftmost_threshold"]
    assert thr == approx(spec.default_threshold_image)


# ---------------------------------------------------------------------------
# update_edge_progress_lfs / left / right
# ---------------------------------------------------------------------------


def test_update_edge_progress_lfs(mainwindow):
    mainwindow.update_edge_progress_lfs(42)
    assert mainwindow.ui.progress_edge_finding_plate1.value() == 42


def test_update_edge_progress_left(mainwindow):
    mainwindow.update_edge_progress_left(75)
    assert mainwindow.ui.progress_edge_finding_plate1.value() == 75


def test_update_edge_progress_right(mainwindow):
    mainwindow.update_edge_progress_right(88)
    assert mainwindow.ui.progress_edge_finding_plate2.value() == 88


def test_update_edge_progress_lfs_with_progressbar(mainwindow):
    """Console progressbar is started lazily on first update."""
    mock_bar = MagicMock()
    mainwindow.console_bar = mock_bar
    mainwindow.console_bar_started = False

    mainwindow.update_edge_progress_lfs(10)
    mock_bar.start.assert_called_once()
    mock_bar.update.assert_called_once_with(10)
    assert mainwindow.console_bar_started is True

    # Second call: start should NOT be called again
    mainwindow.update_edge_progress_lfs(20)
    mock_bar.start.assert_called_once()  # still only once
    mainwindow.console_bar = None
    mainwindow.console_bar_started = False


def test_update_edge_progress_right_with_progressbar(mainwindow):
    """Right console progressbar starts lazily."""
    mock_bar = MagicMock()
    mainwindow.console_bar_right = mock_bar
    mainwindow.console_bar_right_started = False

    mainwindow.update_edge_progress_right(50)
    mock_bar.start.assert_called_once()
    mock_bar.update.assert_called_once_with(50)
    mainwindow.console_bar_right = None
    mainwindow.console_bar_right_started = False

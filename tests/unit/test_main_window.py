from pytest import approx


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

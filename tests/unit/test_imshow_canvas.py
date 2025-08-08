import numpy as np
import pytest

from flametrack.gui.imshow_canvas import ImshowCanvas


def test_initialization(qtbot):
    widget = ImshowCanvas()
    qtbot.addWidget(widget)
    assert widget.data is None
    assert widget.x_max == 0
    assert widget.y_max == 0
    assert widget.image_item is not None
    assert widget.plot_widget is not None


def test_plot_basic(qtbot):
    widget = ImshowCanvas()
    qtbot.addWidget(widget)

    data = np.ones((10, 20))
    widget.plot(data, cmin=0.0, cmax=1.0)

    assert widget.data is data
    rect = widget.image_item.boundingRect()
    assert rect.width() == 20
    assert rect.height() == 10

    levels = widget.image_item.levels  # als Property ohne ()
    assert levels[0] == 0.0
    assert levels[1] == 1.0


def test_plot_with_small_image(qtbot):
    widget = ImshowCanvas()
    qtbot.addWidget(widget)

    data = np.array([[0]])
    widget.plot(data, cmin=0, cmax=1)

    assert widget.data is data
    rect = widget.image_item.boundingRect()
    assert rect.width() == 1
    assert rect.height() == 1


def test_update_colormap_changes_levels(qtbot):
    widget = ImshowCanvas()
    qtbot.addWidget(widget)

    data = np.ones((5, 5)) * 10
    widget.data = data
    widget.image_item.setLevels([0, 10])

    widget.update_colormap(0.2, 0.8)
    levels = widget.image_item.levels  # Property

    assert levels[0] == pytest.approx(2)
    assert levels[1] == pytest.approx(8)


def test_update_colormap_with_no_image_item(qtbot):
    widget = ImshowCanvas()
    qtbot.addWidget(widget)

    widget.image_item = None
    # Should not raise any exception
    widget.update_colormap(0, 1)

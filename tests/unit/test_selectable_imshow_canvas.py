from unittest.mock import patch

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent

from flametrack.gui.draggable_point import DraggablePoint
from flametrack.gui.selectable_imshow_canvas import SelectableImshowCanvas


class DummyParent:
    required_points = 6
    experiment_type = "Room Corner"


class DummyPlotDataItem(pg.PlotDataItem):
    def __init__(self):
        super().__init__()
        self._setData_called = False
        orig_setData = self.setData

        def setData_wrapper(*args, **kwargs):
            self._setData_called = True
            return orig_setData(*args, **kwargs)

        self.setData = setData_wrapper


class DummyVB:
    def __init__(self):
        self.addedItems = []

    def addItem(self, item):
        self.addedItems.append(item)

    def removeItem(self, item):
        if item in self.addedItems:
            self.addedItems.remove(item)

    def setLimits(self, **kwargs):
        pass

    def setRange(self, **kwargs):
        pass

    def mapSceneToView(self, pos):
        return QPointF(5, 5)

    def mapViewToScene(self, pos):
        return pos

    def zValue(self):
        return 0


@pytest.fixture
def canvas(qtbot):
    widget = SelectableImshowCanvas()
    qtbot.addWidget(widget)
    yield widget
    try:
        widget.deleteLater()
    except RuntimeError:
        pass


def test_update_lines_creates_closed_polygon(canvas):
    canvas.parent = DummyParent()
    canvas.draggable_points = [
        DraggablePoint(0, 0, parent=canvas),
        DraggablePoint(1, 0, parent=canvas),
        DraggablePoint(1, 1, parent=canvas),
    ]
    canvas.lines = DummyPlotDataItem()

    with patch(
        "flametrack.gui.selectable_imshow_canvas.sort_corner_points"
    ) as mock_sort:
        mock_sort.return_value = [(0, 0), (1, 0), (1, 1)]
        canvas.updateLines()

    assert canvas.lines._setData_called


def test_plot_calls_update_lines_and_adds_points(canvas):
    canvas.parent = DummyParent()
    canvas.data = np.ones((1, 1))
    p1 = DraggablePoint(0, 0, parent=canvas)
    canvas.draggable_points = [p1]
    canvas.lines = DummyPlotDataItem()
    canvas.plot_widget.plotItem.vb = DummyVB()

    canvas.plot(canvas.data, cmin=0, cmax=1)

    vb = canvas.plot_widget.plotItem.vb
    assert p1 in vb.addedItems
    assert canvas.lines in vb.addedItems


def test_clear_points_removes_all(canvas):
    p1 = DraggablePoint(0, 0, parent=canvas)
    p2 = DraggablePoint(1, 1, parent=canvas)
    canvas.draggable_points = [p1, p2]
    canvas.plot_widget.addItem(p1)
    canvas.plot_widget.addItem(p2)

    canvas.lines = DummyPlotDataItem()
    canvas.plot_widget.addItem(canvas.lines)

    canvas.clear_points()

    assert not canvas.draggable_points
    items = canvas.plot_widget.listDataItems()
    assert p1 not in items
    assert p2 not in items
    assert canvas.lines not in items


def test_key_press_event_deletes_and_clears(canvas):
    canvas.parent = DummyParent()
    p1 = DraggablePoint(0, 0, parent=canvas)
    p2 = DraggablePoint(1, 1, parent=canvas)
    canvas.draggable_points = [p1, p2]
    canvas.plot_widget.addItem(p1)
    canvas.plot_widget.addItem(p2)

    canvas.lines = DummyPlotDataItem()
    canvas.plot_widget.addItem(canvas.lines)
    canvas.plot_widget.plotItem.vb = DummyVB()

    event_d = QKeyEvent(QEvent.KeyPress, Qt.Key_D, Qt.NoModifier)
    canvas.keyPressEvent(event_d)
    assert len(canvas.draggable_points) < 2

    event_c = QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.NoModifier)
    canvas.keyPressEvent(event_c)
    assert len(canvas.draggable_points) == 0


def test_on_click_adds_points_and_lines(canvas):
    canvas.parent = DummyParent()
    canvas.data = np.ones((10, 10))
    canvas.plot_widget.plotItem.vb = DummyVB()

    class DummyEvent:
        def button(self):
            return Qt.MouseButton.LeftButton

        def scenePos(self):
            class Pos:
                def x(self):
                    return 5

                def y(self):
                    return 5

            return Pos()

    event = DummyEvent()
    canvas.on_click(event)

    assert len(canvas.draggable_points) == 1
    assert canvas.lines is not None


def test_on_click_ignores_when_too_many_points(canvas):
    canvas.parent = DummyParent()
    canvas.parent.required_points = 1
    canvas.data = np.ones((10, 10))
    canvas.plot_widget.plotItem.vb = DummyVB()

    class DummyEvent:
        def button(self):
            return Qt.MouseButton.LeftButton

        def scenePos(self):
            class Pos:
                def x(self):
                    return 5

                def y(self):
                    return 5

            return Pos()

    event = DummyEvent()
    canvas.draggable_points = [DraggablePoint(0, 0, parent=canvas)]
    canvas.on_click(event)
    assert len(canvas.draggable_points) == 1  # No additional points added

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from flametrack.gui.draggable_point import DraggablePoint


class DummyEvent:
    def __init__(self, x, y):
        self._pos = QPointF(x, y)
        self._ignored = False

    def pos(self):
        return self._pos

    def ignore(self):
        self._ignored = True


def test_initialization_sets_properties():
    dp = DraggablePoint(10, 20, size=15, color=(100, 150, 200))
    assert dp.size == 15
    assert dp.color == (100, 150, 200)
    assert isinstance(dp.scatter_points[0], QPointF)
    assert dp.scatter_points[0].x() == 10
    assert dp.scatter_points[0].y() == 20
    assert dp.dragging is False
    assert dp.zValue() == 1000


def test_updateGraph_updates_data(monkeypatch):
    dp = DraggablePoint(0, 0)
    called = {}

    def fake_setData(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(dp, "setData", fake_setData)

    dp.updateGraph()

    # Check if setData was called with expected keys
    assert "pos" in called
    assert "symbolBrush" in called
    assert "symbolPen" in called
    assert "size" in called
    assert "symbol" in called


def test_mousePressEvent_starts_dragging_on_point():
    dp = DraggablePoint(5, 5, size=10)
    event = DummyEvent(7, 7)  # Close enough for dragging (manhattanLength < size)
    dp.mousePressEvent(event)
    assert dp.dragging is True
    assert dp.dragged_index == 0
    assert not event._ignored


def test_mousePressEvent_ignores_when_not_on_point():
    dp = DraggablePoint(0, 0, size=5)
    event = DummyEvent(100, 100)  # Far away
    dp.mousePressEvent(event)
    assert dp.dragging is False
    assert event._ignored is True


def test_mouseMoveEvent_moves_point_when_dragging(monkeypatch):
    dp = DraggablePoint(0, 0, size=10)
    dp.dragging = True
    dp.dragged_index = 0
    new_pos = QPointF(20, 20)
    event = DummyEvent(20, 20)

    called_updateLines = False

    class DummyParent:
        def updateLines(self):
            nonlocal called_updateLines
            called_updateLines = True

    dp.parent = DummyParent()
    monkeypatch.setattr(dp, "updateGraph", lambda: None)

    dp.mouseMoveEvent(event)
    # Point should be updated to new pos
    assert dp.scatter_points[0].x() == 20
    assert dp.scatter_points[0].y() == 20
    # updateLines() of parent should have been called
    assert called_updateLines is True


def test_mouseMoveEvent_does_nothing_when_not_dragging(monkeypatch):
    dp = DraggablePoint(0, 0, size=10)
    dp.dragging = False
    dp.dragged_index = None
    event = DummyEvent(20, 20)

    called_updateGraph = False
    monkeypatch.setattr(dp, "updateGraph", lambda: nonlocal_set())

    def nonlocal_set():
        nonlocal called_updateGraph
        called_updateGraph = True

    dp.mouseMoveEvent(event)
    # updateGraph should NOT have been called
    assert called_updateGraph is False


def test_mouseReleaseEvent_stops_dragging():
    dp = DraggablePoint(0, 0)
    dp.dragging = True
    dp.dragged_index = 0
    event = DummyEvent(0, 0)
    dp.mouseReleaseEvent(event)
    assert dp.dragging is False
    assert dp.dragged_index is None


def test_deletePoint_deletes_point_when_dragging(monkeypatch):
    dp = DraggablePoint(0, 0)
    dp.scatter_points.append(QPointF(1, 1))
    dp.dragging = True
    dp.dragged_index = 1

    called_updateGraph = False
    monkeypatch.setattr(dp, "updateGraph", lambda: nonlocal_set())

    def nonlocal_set():
        nonlocal called_updateGraph
        called_updateGraph = True

    dp.deletePoint()
    assert len(dp.scatter_points) == 1
    assert dp.dragging is False
    assert dp.dragged_index is None
    assert called_updateGraph is True


def test_deletePoint_does_nothing_when_not_dragging(monkeypatch):
    dp = DraggablePoint(0, 0)
    dp.dragging = False
    dp.dragged_index = 0

    called_updateGraph = False
    monkeypatch.setattr(dp, "updateGraph", lambda: nonlocal_set())

    def nonlocal_set():
        nonlocal called_updateGraph
        called_updateGraph = True

    dp.deletePoint()
    # Point count should remain unchanged
    assert len(dp.scatter_points) == 1
    # updateGraph should NOT have been called
    assert called_updateGraph is False

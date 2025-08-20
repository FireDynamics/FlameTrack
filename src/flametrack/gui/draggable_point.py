# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF


class DraggablePoint(pg.GraphItem):
    """
    PyQtGraph GraphItem representing a single draggable point.

    Attributes:
        parent: Optional parent object, expected to have update_lines() method.
        size: Diameter of the point symbol.
        color: RGB tuple for point color.
        scatter_points: List of QPointF representing the point(s).
        dragging: Flag indicating if the point is currently being dragged.
        dragged_index: Index of the point being dragged.
    """

    def __init__(
        self,
        x: float,
        y: float,
        size: int = 10,
        color: tuple[int, int, int] = (255, 0, 0),
        *,
        parent=None,
    ):
        """
        Initialize the draggable point at (x, y).

        Args:
            x: Initial x coordinate.
            y: Initial y coordinate.
            size: Size of the point symbol (diameter).
            color: RGB color tuple.
            parent: Optional parent for callbacks.
        """
        super().__init__()
        self.parent = parent
        self.size = size
        self.color = color
        self.scatter_points = [QPointF(x, y)]
        self.dragging = False
        self.dragged_index = None
        self.setZValue(1000)  # ensure point is drawn on top
        self.update_graph()

    def update_graph(self) -> None:
        """
        Update the graphical representation of the point(s).
        Converts QPointF positions to numpy array and sets GraphItem data.
        """
        pos = np.array([[p.x(), p.y()] for p in self.scatter_points])
        adj = np.array([])  # no edges since only single points
        size = np.full(len(pos), self.size)
        symbolBrush = [pg.mkBrush(self.color)] * len(pos)
        symbolPen = [pg.mkPen("k")] * len(pos)
        self.setData(
            pos=pos,
            adj=adj,
            size=size,
            symbolBrush=symbolBrush,
            symbol="o",
            symbolPen=symbolPen,
        )

    def mousePressEvent(self, event) -> None:
        """
        Detect if mouse press is on a point and start dragging.

        Args:
            event: Mouse event from PyQt.
        """
        pos = event.pos()
        for i, point in enumerate(self.scatter_points):
            # manhattanLength() is used as approximate distance
            if (point - pos).manhattanLength() < self.size:
                self.dragging = True
                self.dragged_index = i
                return
        event.ignore()

    def mouseMoveEvent(self, event) -> None:
        """
        Update point position while dragging.

        Args:
            event: Mouse move event from PyQt.
        """
        if self.dragging:
            pos = event.pos()
            self.scatter_points[self.dragged_index] = pos
            self.update_graph()
            if self.parent:
                self.parent.update_lines()

    def mouseReleaseEvent(self, event) -> None:
        """
        Stop dragging when mouse is released.

        Args:
            event: Mouse release event from PyQt.
        """
        self.dragging = False
        self.dragged_index = None

    def delete_point(self) -> None:
        """
        Delete the currently dragged point from the list.
        """
        if self.dragging and self.dragged_index is not None:
            del self.scatter_points[self.dragged_index]
            self.update_graph()
            self.dragging = False
            self.dragged_index = None

    def generateSvg(self, *args, **kwargs):
        """Required abstract method from GraphicsItem (not used)."""
        return None

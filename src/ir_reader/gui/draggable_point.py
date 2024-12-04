import pyqtgraph as pg
from PySide6.QtCore import QPointF

class DraggablePoint(pg.GraphItem):
    def __init__(self, x, y, size=10, color=(255, 0, 0), parent=None):
        pg.GraphItem.__init__(self)
        self.parent = parent
        self.size = size
        self.color = color
        self.scatter_points = [QPointF(x, y)]
        self.dragging = False
        self.updateGraph()

    def updateGraph(self):
        self.setData(pos=self.scatter_points, size=self.size, symbolBrush=self.color)

    def mousePressEvent(self, event):
        pos = event.pos()
        for i, point in enumerate(self.scatter_points):
            if (point - pos).manhattanLength() < self.size:
                self.dragging = True
                self.dragged_index = i
                return
        event.ignore()

    def mouseMoveEvent(self, event):
        if self.dragging:
            pos = event.pos()
            self.scatter_points[self.dragged_index] = pos
            self.updateGraph()
            if self.parent:
                self.parent.updateLines()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def deletePoint(self):
        if self.dragging:
            del self.scatter_points[self.dragged_index]
            self.updateGraph()
            self.dragging = False

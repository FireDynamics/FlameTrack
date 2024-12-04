import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QCursor
from .imshow_canvas import ImshowCanvas
from .draggable_point import DraggablePoint  # Ensure this file exists
from .plotting_utils import sort_corner_points

class SelectableImshowCanvas(ImshowCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Store all draggable points
        self.draggable_points = []
        self.lines = None

        # Connect the plot widget's mouse click event to a custom handler
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def on_click(self, event):
        # Map the clicked point to the plot coordinates
        if self.data is None:
            return
        if self.lines is None:
            self.lines = pg.PlotDataItem(pen='r')
            self.plot_widget.addItem(self.lines)
        if event.button() == Qt.MouseButton.LeftButton and len(self.draggable_points) < 6:
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()

            # Create a draggable point and add it to the plot
            point = DraggablePoint(x, y, parent=self)
            self.draggable_points.append(point)
            self.plot_widget.addItem(point)
        self.updateLines()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_D:
            self.deleteClosestPoint()

        if event.key() == Qt.Key.Key_C:
            self.clear()

    def deleteClosestPoint(self):
        if not self.draggable_points:
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(self.mapFromGlobal(QCursor.pos()))

        # Find the closest point
        closest_point = None
        min_dist = float('inf')

        for point in self.draggable_points:
            dist = (point.scatter_points[0] - mouse_point).manhattanLength()
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        # Delete the closest point
        if closest_point:
            self.plot_widget.removeItem(closest_point)
            self.draggable_points.remove(closest_point)
        self.updateLines()

    def clear(self):
        # Clear all draggable points from the plot
        for point in self.draggable_points:
            self.plot_widget.removeItem(point)
        self.draggable_points = []
        self.lines.setData([], [])

    def updateLines(self):
        # draw outline between points
        if len(self.draggable_points) < 2:
            self.lines.setData([], [])
            return

        points = [(point.scatter_points[0].x(), point.scatter_points[0].y()) for point in self.draggable_points]
        points = sort_corner_points(points)  # Ensure sort_corner_points is defined
        points.append(points[0])
        x, y = zip(*points)
        self.lines.setData(x=x, y=y)

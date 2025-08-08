import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ImshowCanvas(QWidget):
    """
    Widget to display 2D image data using pyqtgraph with a plasma colormap.
    Supports setting display intensity levels and coordinate system aligned
    with matplotlib (y=0 at top).
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.data: np.ndarray | None = None
        self.x_max: int = 0
        self.y_max: int = 0

        # Create pyqtgraph ImageItem for displaying image data
        self.image_item = pg.ImageItem(axisOrder="row-major")  # numpy default

        # Setup layout and add PlotWidget
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("IR - Data")

        # Use 'plasma' colormap for visualization
        self.colormap = pg.colormap.get("plasma")

        # Invert Y axis to match matplotlib coordinate system (y=0 top)
        self.plot_widget.getViewBox().invertY(True)

    def plot(self, data: np.ndarray, cmin: float, cmax: float) -> None:
        """
        Display the 2D data array in the widget using the plasma colormap.

        Args:
            data (np.ndarray): 2D image data to display.
            cmin (float): Minimum normalized intensity for display scaling (0-1).
            cmax (float): Maximum normalized intensity for display scaling (0-1).
        """
        self.data = data
        height, width = data.shape

        # Reset any previous transformations and set the new image data
        self.image_item.resetTransform()
        self.image_item.setImage(data)
        self.image_item.setLookupTable(self.colormap.getLookupTable(0.0, 1.0))
        self.image_item.setLevels([cmin * np.max(data), cmax * np.max(data)])
        self.image_item.setRect(QtCore.QRectF(0, 0, width, height))

        vb = self.plot_widget.plotItem.vb

        # Add image item to the viewbox if not already added
        if self.image_item not in vb.addedItems:
            vb.addItem(self.image_item)

        # Set bounds and visible range to fit the image exactly
        vb.setLimits(xMin=0, xMax=width, yMin=0, yMax=height)
        vb.setRange(xRange=(0, width), yRange=(0, height), padding=0)

    def update_colormap(self, cmin: float, cmax: float) -> None:
        """
        Update intensity levels of the current image display.

        Args:
            cmin (float): New minimum normalized intensity (0-1).
            cmax (float): New maximum normalized intensity (0-1).
        """
        if self.image_item is not None and self.data is not None:
            self.image_item.setLevels([cmin * self.data.max(), cmax * self.data.max()])

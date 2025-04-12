import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout

class ImshowCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.x_max, self.y_max = 0, 0
        self.image_item = None
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Set a custom colormap for the heatmap
        self.colormap = pg.colormap.get('plasma')

        # Create a PlotWidget instance
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("IR - Data")

    def plot(self, data, cmin, cmax):
        data = data.T
        self.data = data
        if self.image_item is None:
            self.image_item = pg.ImageItem(data)
            self.image_item.setLookupTable(self.colormap.getLookupTable(cmin, cmax))
            self.plot_widget.addItem(self.image_item)
            self.x_max, self.y_max = data.shape
            self.plot_widget.setLimits(xMin=0, xMax=self.x_max, yMin=0, yMax=self.y_max)
            self.update_colormap(cmin, cmax)
        else:
            self.image_item.setImage(data)
            self.update_colormap(cmin, cmax)

    def update_colormap(self, cmin, cmax):
        if self.image_item is not None:
            self.image_item.setLevels([cmin * self.data.max(), cmax * self.data.max()])

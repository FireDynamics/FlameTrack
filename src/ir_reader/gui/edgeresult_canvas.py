import pyqtgraph as pg
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout

class EdgeResultCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        layout.addWidget(self.plot_widget)


    def plot_edge_results(self, experiment, y_cutoff=0.5):
        if experiment is None:
            return
        edge_results1 = experiment.h5_file['edge_results_left']['data'][:]
        edge_results2 = experiment.h5_file['edge_results_right']['data'][:]
        edge_results1 = -edge_results1 + np.max(edge_results1)
        y1 = int(edge_results1.shape[1] * y_cutoff)
        y2 = int(edge_results2.shape[1] * y_cutoff)
        y1 = max(-y1 - 1, -edge_results1.shape[1])
        y2 = max(-y2 - 1, -edge_results2.shape[1])

        width1 = experiment.h5_file['dewarped_data_left'].attrs['target_pixels_width']
        width2 = experiment.h5_file['dewarped_data_right'].attrs['target_pixels_width']

        target_width = max(width1, width2)
        edge_results1 = edge_results1* target_width / width1
        edge_results2 = edge_results2* target_width / width2
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.plot_widget.plot(edge_results1[:, y1], pen='r',name='Left')
        self.plot_widget.plot(edge_results2[:, y2], pen='b',name='Right')
        self.plot_widget.showGrid(x=True, y=True)
        #set title0
        self.plot_widget.setTitle(f'Horizontal flamespread at y = {y_cutoff:.0%}')
import os.path
import sys
from datetime import datetime

import numpy as np
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QTabWidget, QFrame, QFileDialog, QGroupBox, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt

import dataset_handler
from IR_analysis import sort_corner_points, get_dewarp_parameters, dewarp_RCE_exp
from flamespread import calculate_edge_results_for_exp_name
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from DataTypes import RCE_Experiment
from dataset_handler import create_h5_file


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.experiment = None
        self.target_ratio = 3 / 2
        self.setWindowTitle('Flamespread Analysis Tool')

        # Create the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Sidebar layout with buttons
        sidebar = QVBoxLayout()
        main_layout.addLayout(sidebar)

        # Add buttons to the sidebar
        load_button = QPushButton('Load File')
        load_button.clicked.connect(self.load_file)
        sidebar.addWidget(load_button)

        sidebar.addStretch()

        # Right side layout (header + tabs)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        # Header label
        header = QLabel("Header Text", self)
        header.setAlignment(Qt.AlignCenter)
        header.setFrameStyle(QFrame.Box | QFrame.Plain)
        right_layout.addWidget(header)

        # Tab widget
        tabs = QTabWidget()
        right_layout.addWidget(tabs)

        # Tab 1 with plots
        dewarp_tab = QWidget()
        dewarp_tab_layout = QVBoxLayout()
        dewarp_tab.setLayout(dewarp_tab_layout)

        # Add plots to Tab 1
        top_part = QHBoxLayout()
        dewarp_tab_layout.addLayout(top_part)

        self.big_plot = SelectableImshowCanvas(self)
        top_part.addWidget(self.big_plot)

        # Add preview (maybe later)
        # self.small_plot = PlotCanvas(self)
        # top_part.addWidget(self.small_plot,5)

        self.dewarp_button_layout = QHBoxLayout()
        dewarp_button = QPushButton('Dewarp')
        dewarp_button.clicked.connect(lambda: self.dewarp())
        self.dewarp_button_layout.addWidget(dewarp_button)
        dewarp_tab_layout.addLayout(self.dewarp_button_layout)

        # Add sliders to the dewarping tab
        slider_group = QGroupBox("")
        slider_group_layout = QVBoxLayout()
        self.sliders = []
        for i in range(3):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setDisabled(True)
            self.sliders.append(slider)
            slider_group_layout.addWidget(slider)

        self.sliders[2].setValue(100)
        self.sliders[0].sliderReleased.connect(lambda: self.update_plot(framenr=self.sliders[0].value()))
        self.sliders[1].sliderReleased.connect(lambda: self.update_plot(cmin=self.sliders[1].value()))
        self.sliders[2].sliderReleased.connect(lambda: self.update_plot(cmax=self.sliders[2].value()))
        slider_group.setLayout(slider_group_layout)
        dewarp_tab_layout.addWidget(slider_group)

        edge_results_tab = QWidget()
        edge_results_tab_layout = QVBoxLayout()
        edge_results_tab.setLayout(edge_results_tab_layout)
        button = QPushButton('Calculate Edge Results')
        button.clicked.connect(lambda: self.calculate_edge_results())
        edge_results_tab_layout.addWidget(button)

        analysis_tab = QWidget()
        analysis_tab_layout = QHBoxLayout()
        analysis_tab.setLayout(analysis_tab_layout)

        y_slider = QSlider(Qt.Vertical)
        y_slider.setMinimum(0)
        y_slider.setMaximum(100)
        #add tick marks
        y_slider.setTickPosition(QSlider.TicksBothSides)
        y_slider.setTickInterval(10)




        y_slider.valueChanged.connect(lambda: self.edge_results_canvas.plot_edge_results(self.experiment, y_slider.value()/100))
        analysis_tab_layout.addWidget(y_slider)

        self.edge_results_canvas = EdgeResultCanvas()
        analysis_tab_layout.addWidget(self.edge_results_canvas)


        # Add tabs to the tab widget
        tabs.addTab(dewarp_tab, "Dewarping")
        tabs.addTab(edge_results_tab, "Edge recognition")
        tabs.addTab(analysis_tab, "Flamespread Analysis")

    def load_file(self):

        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.experiment = RCE_Experiment(folder)

        if self.experiment:
            self.update_plot(framenr=0)
            for slider in self.sliders:
                slider.setDisabled(False)
            self.sliders[0].setMinimum(0)
            self.sliders[0].setMaximum(self.experiment.get_IR_data().get_frame_count())

    def update_plot(self, framenr=None, cmin=None, cmax=None):
        if self.experiment is None:
            return
        # Assign slider value if not specified
        cmin = cmin or self.sliders[1].value()
        cmax = cmax or self.sliders[2].value()
        cmin = min(cmin, cmax) / 100
        cmax = max(cmin, cmax) / 100
        if framenr is None:
            self.big_plot.update_colormap(cmin, cmax)
        else:
            self.big_plot.plot(self.experiment.get_IR_data().get_frame(framenr), cmin, cmax)

    def dewarp(self):
        filename = os.path.join(self.experiment.folder_path, 'processed_data',
                                f'{self.experiment.exp_name}_results_RCE.h5')
        if os.path.exists(filename):
            # Ask for overwrite
            choice = QMessageBox.question(self, 'File exists', 'Overwrite existing file?',
                                          QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.No:
                return

        corner_points = sort_corner_points([(point.scatter_points[0].x(), point.scatter_points[0].y())
                                            for point in self.big_plot.draggable_points])
        corner_points = np.array(corner_points)
        if len(corner_points) < 6:
            QMessageBox.warning(self, 'Not enough points', 'Please select 6 points')
            return
        selected_points_left = corner_points[[0, 1, 4, 5]]
        selected_points_right = corner_points[[1, 2, 3, 4]]
        dewarp_params_left = get_dewarp_parameters(selected_points_left, target_ratio=self.target_ratio)
        dewarp_params_right = get_dewarp_parameters(selected_points_right, target_ratio=self.target_ratio)
        if self.experiment.h5_file is not None:
            self.experiment.h5_file.close()

        with create_h5_file(filename=filename) as h5_file:
            h5_file.create_group('dewarped_data_left')
            h5_file.create_group('dewarped_data_right')
            h5_file.create_group('edge_results_left')
            h5_file.create_group('edge_results_right')
            grp_left = h5_file['dewarped_data_left']
            grp_right = h5_file['dewarped_data_right']
            for grp, dewarp_params, selected_points in zip([grp_left, grp_right],
                                                           [dewarp_params_left, dewarp_params_right],
                                                           [selected_points_left, selected_points_right]):
                grp.attrs['transformation_matrix'] = dewarp_params['transformation_matrix']
                grp.attrs['target_pixels_width'] = dewarp_params['target_pixels_width']
                grp.attrs['target_pixels_height'] = dewarp_params['target_pixels_height']
                grp.attrs['target_ratio'] = dewarp_params['target_ratio']
                grp.attrs['selected_points'] = selected_points
                grp.attrs['frame_range'] = [0, self.experiment.get_IR_data().get_frame_count()]
                grp.attrs['points_selection_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dset_h, dset_w = dewarp_params['target_pixels_height'], dewarp_params['target_pixels_width']
                dset = grp.create_dataset('data',
                                          (dset_h, dset_w, 1),
                                          maxshape=(dset_h, dset_w, None),
                                          chunks=(dset_h, dset_w, 1),

                                          dtype=np.float32)

            self.experiment.h5_file = h5_file
            dewarp_button = self.dewarp_button_layout.itemAt(0).widget()
            dewarp_button.setDisabled(True)
            # refresh the window
            QApplication.processEvents()
            progressbar = QProgressBar()
            progressbar.setFormat('%p%')
            progressbar.setRange(0, self.experiment.get_IR_data().get_frame_count())
            self.dewarp_button_layout.addWidget(progressbar)

            for progress in dewarp_RCE_exp(self.experiment, testing=False,frequency=1):
                progressbar.setValue(progress)
                # refresh the window
                QApplication.processEvents()
            self.dewarp_button_layout.removeWidget(progressbar)
            dewarp_button.setDisabled(False)
            h5_file.close()

    def calculate_edge_results(self):
        if self.experiment is None or self.experiment.h5_file is None:
            QMessageBox.warning(self, 'No file loaded', 'Please load a file first and dewarp it')
            return
        dewarped_data_left = self.experiment.h5_file['dewarped_data_left']['data'][:]
        dewarped_data_right = self.experiment.h5_file['dewarped_data_right']['data'][:]

        results_left = calculate_edge_results_for_exp_name(self.experiment.exp_name, left=True,
                                                           dewarped_data=dewarped_data_left, save=False)
        results_right = calculate_edge_results_for_exp_name(self.experiment.exp_name, left=False,
                                                            dewarped_data=dewarped_data_right, save=False)
        self.experiment.h5_file['edge_results_left'].create_dataset('data', data=results_left)
        self.experiment.h5_file['edge_results_right'].create_dataset('data', data=results_right)
        self.experiment.h5_file.close()


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


class SelectableImshowCanvas(ImshowCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Store all draggable points
        self.draggable_points = []
        self.lines = None

        # Connect the plot widget's mouse click event to a custom handler
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)
        self.setFocusPolicy(Qt.StrongFocus)

    def on_click(self, event):
        # Map the clicked point to the plot coordinates
        if self.data is None:
            return
        if self.lines is None:
            self.lines = pg.PlotDataItem(pen='r')
            self.plot_widget.addItem(self.lines)
        if event.button() == Qt.LeftButton and len(self.draggable_points) < 6:
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()

            # Create a draggable point and add it to the plot
            point = DraggablePoint(x, y, parent=self)
            self.draggable_points.append(point)
            self.plot_widget.addItem(point)
        self.updateLines()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            self.deleteClosestPoint()

        if event.key() == Qt.Key_C:
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
        points = sort_corner_points(points)
        points.append(points[0])
        x, y = zip(*points)
        self.lines.setData(x=x, y=y, pen='r')


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

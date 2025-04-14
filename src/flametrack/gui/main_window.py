from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QApplication, QProgressBar, QSlider
from .ui_form import Ui_MainWindow
from gui.draggable_point import DraggablePoint
from gui.selectable_imshow_canvas import SelectableImshowCanvas
from analysis.DataTypes import RCE_Experiment
from analysis.dataset_handler import create_h5_file
from analysis.IR_analysis import sort_corner_points, get_dewarp_parameters, dewarp_RCE_exp
from analysis.flamespread import calculate_edge_results_for_exp_name
from datetime import datetime
import os
import numpy as np

DATATYPE = 'ir'


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Create an instance of the UI class
        self.ui = Ui_MainWindow()

        # Setup the UI
        self.ui.setupUi(self)

        self.experiment = None
        self.target_ratio = 3 / 2
        self.setWindowTitle('Flamespread Analysis Tool')

        ## Set up Dewarp page

        # Connect button 
        self.ui.button_open_folder.clicked.connect(self.load_file)

        # Initialize variable to store rotation factor
        self.rotation_factor = 0

        # Connect signal to slot (function) to handle selection
        self.ui.combo_rotation.currentIndexChanged.connect(
            lambda: self.update_plot(framenr=self.ui.slider_frame.value(), rotationfactor=self.rotation_factor))
        # Connect sliders
        self.ui.slider_frame.sliderReleased.connect(
            lambda: self.update_plot(framenr=self.ui.slider_frame.value(), rotationfactor=self.rotation_factor))
        self.ui.slider_scale_min.sliderReleased.connect(
            lambda: self.update_plot(cmin=self.ui.slider_scale_min.value(), rotationfactor=self.rotation_factor))
        self.ui.slider_scale_max.sliderReleased.connect(
            lambda: self.update_plot(cmax=self.ui.slider_scale_max.value(), rotationfactor=self.rotation_factor))

        # Setup dewarp button
        self.ui.button_dewarp.clicked.connect(self.dewarp)

        ## Set up edge recognition tab

        # Connect start  edge recognition button
        self.ui.button_find_edge.clicked.connect(self.calculate_edge_results)

        ## Set up analysis tab
        self.ui.slider_analysis_y.setMinimum(0)
        self.ui.slider_analysis_y.setMaximum(100)
        self.ui.slider_analysis_y.setTickPosition(QSlider.TicksBothSides)
        self.ui.slider_analysis_y.setTickInterval(10)
        self.ui.slider_analysis_y.valueChanged.connect(
            lambda: self.ui.plot_analysis.plot_edge_results(self.experiment, self.ui.slider_analysis_y.value() / 100))

    def load_file(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.experiment = RCE_Experiment(folder)  # Ensure RCE_Experiment is defined

        if self.experiment:
            self.update_plot(framenr=0)
            self.ui.slider_frame.setDisabled(False)
            self.ui.slider_scale_min.setDisabled(False)
            self.ui.slider_scale_max.setDisabled(False)
            self.ui.slider_frame.setMinimum(0)
            self.ui.slider_frame.setMaximum(self.experiment.get_data(DATATYPE).get_frame_count())

    def update_plot(self, framenr=None, rotationfactor=None, cmin=None, cmax=None):
        if self.experiment is None:
            return
        rotationfactor = self.ui.combo_rotation.currentIndex()
        # Assign slider value if not specified
        cmin = cmin or self.ui.slider_scale_min.value()
        cmax = cmax or self.ui.slider_scale_max.value()
        cmin = min(cmin, cmax) / 100
        cmax = max(cmin, cmax) / 100
        print("update plot:", rotationfactor)
        if framenr is None:
            self.ui.plot_dewarping.update_colormap(cmin, cmax)
        else:
            self.ui.plot_dewarping.plot(self.experiment.get_data(DATATYPE).get_frame(framenr, rotationfactor), cmin,
                                        cmax)

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
                                            for point in self.ui.plot_dewarping.draggable_points])
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
                grp.attrs['frame_range'] = [0, self.experiment.get_data(DATATYPE).get_frame_count()]
                grp.attrs['points_selection_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dset_h, dset_w = dewarp_params['target_pixels_height'], dewarp_params['target_pixels_width']
                dset = grp.create_dataset('data',
                                          (dset_h, dset_w, 1),
                                          maxshape=(dset_h, dset_w, None),
                                          chunks=(dset_h, dset_w, 1),

                                          dtype=np.float32)

            self.experiment.h5_file = h5_file
            self.ui.button_dewarp.setDisabled(True)
            # refresh the window
            self.ui.progress_dewarping.setFormat('%p%')
            self.ui.progress_dewarping.setRange(0, self.experiment.get_data(DATATYPE).get_frame_count() - 1)

            for progress in dewarp_RCE_exp(self.experiment, self.ui.combo_rotation.currentIndex(), testing=False,
                                           frequency=1,data_type=DATATYPE):
                self.ui.progress_dewarping.setValue(progress)
                # refresh the window
                QApplication.processEvents()
            self.ui.button_dewarp.setDisabled(False)
            h5_file.close()

    def calculate_edge_results(self):
        if self.experiment is None or self.experiment.h5_file is None:
            QMessageBox.warning(self, 'No file loaded', 'Please load a file first and dewarp it')
            return
        dewarped_data_left = self.experiment.h5_file['dewarped_data_left']['data'][:]
        dewarped_data_right = self.experiment.h5_file['dewarped_data_right']['data'][:]

        # # refresh the window
        # QApplication.processEvents()
        # progressbar = QProgressBar()
        # progressbar.setFormat('%p%')
        # progressbar.setRange(0, self.experiment.get_IR_data().get_frame_count()*2)

        # for progress in dewarp_RCE_exp(self.experiment, testing=False,frequency=1):
        #     progressbar.setValue(progress)
        #     # refresh the window
        #     QApplication.processEvents()
        # self.dewarp_button_layout.removeWidget(progressbar)
        # refresh the window
        # self.ui.progress_edge_finding.setFormat('%p%')
        # self.ui.progress_edge_finding.setRange(0, self.experiment.get_IR_data().get_frame_count()-1)

        # for progress in dewarp_RCE_exp(self.experiment, self.ui.combo_rotation.currentIndex(), testing=False,frequency=1):
        #    self.ui.progress_edge_finding.setValue(progress)
        #    # refresh the window
        #    QApplication.processEvents()

        results_left = calculate_edge_results_for_exp_name(self.experiment.exp_name, left=True,
                                                           dewarped_data=dewarped_data_left, save=False)
        results_right = calculate_edge_results_for_exp_name(self.experiment.exp_name, left=False,
                                                            dewarped_data=dewarped_data_right, save=False)
        self.experiment.h5_file['edge_results_left'].create_dataset('data', data=results_left)
        self.experiment.h5_file['edge_results_right'].create_dataset('data', data=results_right)
        self.experiment.h5_file.close()
        self.ui.button_dewarp.setDisabled(False)

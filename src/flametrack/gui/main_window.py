import logging
import os
from typing import Optional

import h5py
import numpy as np
import progressbar
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSlider,
)

from flametrack.analysis.data_types import RceExperiment
from flametrack.analysis.edge_worker import EdgeDetectionWorker
from flametrack.analysis.flamespread import (
    calculate_edge_data,
    calculate_edge_results_for_exp_name,
    left_most_point_over_threshold,
    right_most_point_over_threshold,
)
from flametrack.processing.dewarping import (
    DewarpConfig,
    dewarp_lateral_flame_spread,
    dewarp_room_corner_remap,
)
from flametrack.utils.math_utils import compute_target_ratio

from .ui_form import Ui_MainWindow

DATATYPE = "IR"

EXPERIMENT_CONFIG = {
    "Room Corner": {"required_points": 6},
    "Lateral Flame Spread": {"required_points": 4},
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.experiment: Optional[RceExperiment] = None
        self.experiment_type: str = "Lateral Flame Spread"
        self.required_points: int = 4
        self.rotation_index: int = 0
        self.target_ratio: Optional[float] = None
        self.target_pixels_width: Optional[int] = None
        self.target_pixels_height: Optional[int] = None
        self.edge_workers_done: int = 0

        # Initialize values
        self.plate_width_m: Optional[float] = None
        self.plate_height_m: Optional[float] = None
        self.console_bar: Optional[progressbar.ProgressBar] = None
        self.console_bar_started: bool = False
        self.console_bar_left: Optional[progressbar.ProgressBar] = None
        self.console_bar_right: Optional[progressbar.ProgressBar] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[EdgeDetectionWorker] = None
        self.thread_left: Optional[QThread] = None
        self.worker_left: Optional[EdgeDetectionWorker] = None
        self.thread_right: Optional[QThread] = None
        self.worker_right: Optional[EdgeDetectionWorker] = None

        self._setup_ui()
        self._setup_connections()
        self._initialize_defaults()

    def _setup_ui(self) -> None:
        """Initial UI setup: titles, visibility, and default states."""
        self.setWindowTitle("Flamespread Analysis Tool")
        self.ui.plot_dewarping.parent = self
        self.ui.progress_edge_finding_plate2.hide()
        self.ui.checkBox_mulithread.hide()
        self.ui.slider_analysis_y.setMinimum(0)
        self.ui.slider_analysis_y.setMaximum(100)
        self.ui.slider_analysis_y.setTickPosition(QSlider.TicksBothSides)
        self.ui.slider_analysis_y.setTickInterval(10)

    def _setup_connections(self) -> None:
        """Connect UI signals to their corresponding slots."""
        self.ui.button_open_folder.clicked.connect(self.load_file)
        self.ui.combo_rotation.currentIndexChanged.connect(self.set_rotation_index)
        self.ui.doubleSpinBox_plate_width.valueChanged.connect(self.update_target_ratio)
        self.ui.doubleSpinBox_plate_height.valueChanged.connect(
            self.update_target_ratio
        )
        self.ui.comboBox_experiment_type.currentTextChanged.connect(
            self.set_experiment_type
        )
        self.ui.comboBox_experiment_type.currentTextChanged.connect(
            self.update_flame_direction_visibility
        )
        self.ui.comboBox_flame_direction.currentTextChanged.connect(
            self.update_edge_preview
        )
        self.ui.slider_frame.sliderReleased.connect(
            lambda: self.update_plot(
                framenr=self.ui.slider_frame.value(),
                rotation_factor=self.rotation_index,
            )
        )
        self.ui.slider_scale_min.sliderReleased.connect(
            lambda: self.update_plot(
                cmin=self.ui.slider_scale_min.value(),
                rotation_factor=self.rotation_index,
            )
        )
        self.ui.slider_scale_max.sliderReleased.connect(
            lambda: self.update_plot(
                cmax=self.ui.slider_scale_max.value(),
                rotation_factor=self.rotation_index,
            )
        )
        self.ui.button_dewarp.clicked.connect(self.on_dewarp_clicked)
        self.ui.button_find_edge.clicked.connect(self.start_edge_detection)
        self.ui.slider_analysis_y.valueChanged.connect(self.update_analysis_plot)

    def _initialize_defaults(self) -> None:
        """Set initial values for UI elements and internal parameters."""
        self.ui.combo_rotation.setCurrentIndex(0)
        self.update_target_ratio()
        initial_type = self.ui.comboBox_experiment_type.currentText()
        self.set_experiment_type(initial_type)
        self.update_flame_direction_visibility()

    def set_rotation_index(self, index: int) -> None:
        self.rotation_index = index
        self.update_plot(framenr=self.ui.slider_frame.value(), rotation_factor=index)

    @staticmethod
    def _read_plate_mm_from_h5_root_first(
        h5,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Liefert (width_mm, height_mm).
        Reihenfolge:
          1) Room-Corner Root: plate_width_mm_left/right, plate_height_mm_left/right
          2) Gruppen: dewarped_data_left/right
          3) LFS Root: plate_width_mm, plate_height_mm
          4) Gruppe: dewarped_data
        Nimmt für die GUI (ein Paar Spinboxes) bevorzugt 'left', fällt sonst auf 'right' zurück.
        """

        def _f(x) -> Optional[float]:
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        # 1) Room-Corner Root (left preferred, fallback right)
        w_l = h5.attrs.get("plate_width_mm_left", None)
        h_l = h5.attrs.get("plate_height_mm_left", None)
        w_r = h5.attrs.get("plate_width_mm_right", None)
        h_r = h5.attrs.get("plate_height_mm_right", None)

        if w_l is not None or h_l is not None or w_r is not None or h_r is not None:
            w = w_l if w_l is not None else w_r
            h = h_l if h_l is not None else h_r
            return _f(w), _f(h)

        # 2) Gruppen (Room-Corner)
        if "dewarped_data_left" in h5 or "dewarped_data_right" in h5:
            w = h = None
            if "dewarped_data_left" in h5:
                g = h5["dewarped_data_left"].attrs
                w = g.get("plate_width_mm", None)
                h = g.get("plate_height_mm", None)
            if (w is None or h is None) and "dewarped_data_right" in h5:
                g = h5["dewarped_data_right"].attrs
                w = w if w is not None else g.get("plate_width_mm", None)
                h = h if h is not None else g.get("plate_height_mm", None)
            return _f(w), _f(h)

        # 3) LFS Root (einheitliche Platte)
        w = h5.attrs.get("plate_width_mm", None)
        h = h5.attrs.get("plate_height_mm", None)
        if w is not None or h is not None:
            return _f(w), _f(h)

        # 4) Gruppe dewarped_data (LFS)
        if "dewarped_data" in h5:
            g = h5["dewarped_data"].attrs
            w = g.get("plate_width_mm", None)
            h = g.get("plate_height_mm", None)
            return _f(w), _f(h)

        return None, None

    # ---- Helper inside MainWindow ------------------------------------------------
    def _detect_experiment_type_from_h5(self, h5) -> None:
        """Setzt experiment_type und ComboBox anhand der HDF5-Gruppen."""
        if "dewarped_data_left" in h5 and "dewarped_data_right" in h5:
            self.experiment_type = "Room Corner"
            self.ui.comboBox_experiment_type.setCurrentText("Room Corner")
        elif "dewarped_data" in h5:
            self.experiment_type = "Lateral Flame Spread"
            self.ui.comboBox_experiment_type.setCurrentText("Lateral Flame Spread")
        else:
            logging.debug("No dewarped data found in HDF5 - type unchanged.")

    def _read_plate_mm(self, h5) -> tuple[Optional[float], Optional[float]]:
        """Liest Plattenmaße (mm) – Root bevorzugt, dann Gruppen (Room Corner/LFS)."""

        def _f(x) -> Optional[float]:
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        # Room-Corner: Root (left/right)
        w_mm = h5.attrs.get("plate_width_mm_left")
        h_mm = h5.attrs.get("plate_height_mm_left")
        if w_mm is None:
            w_mm = h5.attrs.get("plate_width_mm_right")
        if h_mm is None:
            h_mm = h5.attrs.get("plate_height_mm_right")
        if w_mm is not None or h_mm is not None:
            return _f(w_mm), _f(h_mm)

        # Room-Corner: Gruppen
        if "dewarped_data_left" in h5:
            g = h5["dewarped_data_left"].attrs
            w_mm = g.get("plate_width_mm")
            h_mm = g.get("plate_height_mm")
        if (w_mm is None or h_mm is None) and "dewarped_data_right" in h5:
            g = h5["dewarped_data_right"].attrs
            w_mm = g.get("plate_width_mm") if w_mm is None else w_mm
            h_mm = g.get("plate_height_mm") if h_mm is None else h_mm
        if w_mm is not None or h_mm is not None:
            return _f(w_mm), _f(h_mm)

        # LFS: Root
        w_mm = h5.attrs.get("plate_width_mm")
        h_mm = h5.attrs.get("plate_height_mm")
        if w_mm is not None or h_mm is not None:
            return _f(w_mm), _f(h_mm)

        # LFS: Gruppe
        if "dewarped_data" in h5:
            g = h5["dewarped_data"].attrs
            return _f(g.get("plate_width_mm")), _f(g.get("plate_height_mm"))

        return None, None

    def _apply_plate_mm_to_spinboxes(
        self, w_mm: Optional[float], h_mm: Optional[float]
    ) -> None:
        """Schreibt (falls vorhanden) die Plattenmaße in die SpinBoxes und aktualisiert Ratio."""
        wrote_any = False
        if w_mm is not None:
            self.ui.doubleSpinBox_plate_width.setValue(w_mm)
            wrote_any = True
        if h_mm is not None:
            self.ui.doubleSpinBox_plate_height.setValue(h_mm)
            wrote_any = True
        if wrote_any:
            self.update_target_ratio()
            logging.debug("Loaded plate size (mm): %s x %s", w_mm, h_mm)
        else:
            logging.debug("Plate size (mm) not found in HDF5 (root or groups).")

    def _enable_controls_after_load(self) -> None:
        """Aktiviert Slider/Controls nach erfolgreichem Laden und setzt Bereiche."""
        self.ui.slider_frame.setDisabled(False)
        self.ui.slider_scale_min.setDisabled(False)
        self.ui.slider_scale_max.setDisabled(False)
        frame_count = self.experiment.get_data(DATATYPE).get_frame_count()
        self.ui.slider_frame.setMinimum(0)
        self.ui.slider_frame.setMaximum(frame_count)

    def load_file(self) -> None:
        """Open directory dialog, load experiment data, set UI elements."""
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not folder:
            return

        self.experiment = RceExperiment(folder)
        try:
            _ = self.experiment.get_data(DATATYPE)  # force load
            h5 = self.experiment.h5_file
            self.experiment.h5_path = h5.filename

            # Experiment-Typ & Plattenmaße ermitteln
            self._detect_experiment_type_from_h5(h5)
            w_mm, h_mm = self._read_plate_mm(h5)
            self._apply_plate_mm_to_spinboxes(w_mm, h_mm)

        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logging.debug("Error reading experiment type from HDF5: %s", exc)

        # UI & Plots aktualisieren
        self.update_plot(framenr=0)
        self._enable_controls_after_load()
        if self.experiment:
            self.experiment.experiment_type = self.experiment_type
        self.update_edge_preview()

        y_cutoff = self.ui.slider_analysis_y.value() / 100
        self.ui.plot_analysis.plot_edge_results(self.experiment, y_cutoff=y_cutoff)

    def update_target_ratio(self) -> None:
        """Compute and update target pixel size and ratio based on plate dimensions."""
        width_mm = self.ui.doubleSpinBox_plate_width.value()
        height_mm = self.ui.doubleSpinBox_plate_height.value()

        self.target_ratio = compute_target_ratio(width_mm, height_mm)

        mm_per_pixel = 5
        self.target_pixels_width = int(round(width_mm / mm_per_pixel))
        self.target_pixels_height = int(round(height_mm / mm_per_pixel))

        self.plate_width_m = width_mm / 1000.0
        self.plate_height_m = height_mm / 1000.0

        logging.debug("target_ratio: %s", self.target_ratio)
        logging.debug("target_pixels_width: %s", self.target_pixels_width)
        logging.debug("target_pixels_height: %s", self.target_pixels_height)

    def set_experiment_type(self, experiment_type: str) -> None:
        """Set experiment type and adjust UI visibility and required points."""
        self.experiment_type = experiment_type
        if self.experiment:
            self.experiment.experiment_type = experiment_type

        if experiment_type == "Room Corner":
            self.required_points = 6
            self.ui.progress_edge_finding_plate1.show()
            self.ui.progress_edge_finding_plate2.show()
            self.ui.checkBox_mulithread.show()
        elif experiment_type == "Lateral Flame Spread":
            self.required_points = 4
            self.ui.progress_edge_finding_plate1.show()
            self.ui.progress_edge_finding_plate2.hide()
            self.ui.checkBox_mulithread.hide()

        if hasattr(self.ui, "plot_dewarping"):
            self.ui.plot_dewarping.clear_points()

        logging.debug(
            "Set experiment type to %s - expecting %s points",
            experiment_type,
            self.required_points,
        )

    def update_plot(
        self,
        framenr: Optional[int] = None,
        rotation_factor: Optional[int] = None,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
    ) -> None:
        """Update image plot according to frame, rotation and color scaling."""
        if not self.experiment:
            return

        rotation_factor = (
            rotation_factor
            if rotation_factor is not None
            else self.ui.combo_rotation.currentIndex()
        )
        frame = framenr if framenr is not None else self.ui.slider_frame.value()
        cmin = cmin if cmin is not None else self.ui.slider_scale_min.value()
        cmax = cmax if cmax is not None else self.ui.slider_scale_max.value()

        # Ensure cmin <= cmax and normalize to 0..1 range
        cmin = min(cmin, cmax) / 100
        cmax = max(cmin, cmax) / 100

        logging.debug(
            "update_plot: frame=%s, rotation=%s, cmin=%s, cmax=%s",
            frame,
            rotation_factor,
            cmin,
            cmax,
        )

        img = self.experiment.get_data(DATATYPE).get_frame(frame, rotation_factor)
        self.ui.plot_dewarping.plot(img, cmin, cmax)

    def calculate_edge_results(self) -> None:
        """Calculate edge detection results and write to HDF5 datasets."""
        if not self.experiment or not self.experiment.h5_file:
            QMessageBox.warning(
                self, "No file loaded", "Please load a file first and dewarp it"
            )
            return

        # Left plate edge
        dewarped_data_left = self.experiment.h5_file["dewarped_data_left"]["data"]
        results_left = calculate_edge_results_for_exp_name(
            self.experiment.exp_name,
            left=True,
            dewarped_data=dewarped_data_left,
            save=False,
        )
        grp_left = self.experiment.h5_file["edge_results_left"]
        if "data" in grp_left:
            del grp_left["data"]
        grp_left.create_dataset("data", data=results_left)
        logging.debug("Finished LEFT edge calculation")

        # Right plate edge
        dewarped_data_right = self.experiment.h5_file["dewarped_data_right"]["data"]
        results_right = calculate_edge_results_for_exp_name(
            self.experiment.exp_name,
            left=False,
            dewarped_data=dewarped_data_right,
            save=False,
        )
        grp_right = self.experiment.h5_file["edge_results_right"]
        if "data" in grp_right:
            del grp_right["data"]
        grp_right.create_dataset("data", data=results_right)
        logging.debug("Finished RIGHT edge calculation")

        self.experiment.h5_file.close()
        self.ui.button_dewarp.setDisabled(False)

    def on_dewarp_clicked(self) -> None:
        """Trigger dewarping based on user-set points and experiment type."""
        points = [
            (p.scatter_points[0].x(), p.scatter_points[0].y())
            for p in self.ui.plot_dewarping.draggable_points
        ]

        if len(points) not in (4, 6):
            QMessageBox.warning(
                self, "Invalid point count", "Please set exactly 4 or 6 points."
            )
            return

        experiment_type = self.experiment_type
        rotation_index = self.ui.combo_rotation.currentIndex()
        plate_width_mm = self.ui.doubleSpinBox_plate_width.value()
        plate_height_mm = self.ui.doubleSpinBox_plate_height.value()

        cfg = DewarpConfig(
            target_ratio=self.target_ratio or 1.0,
            target_pixels_width=self.target_pixels_width or 100,
            target_pixels_height=self.target_pixels_height or 100,
            plate_width_mm=plate_width_mm,
            plate_height_mm=plate_height_mm,
            rotation_index=rotation_index,
            filename=None,  # optional; du kannst hier auch einen Pfad setzen
            frequency=1,
            testing=False,
        )

        try:
            if experiment_type == "Room Corner" and len(points) == 6:
                dewarp_generator = dewarp_room_corner_remap(
                    experiment=self.experiment,
                    points=points,
                    config=cfg,
                )
            elif experiment_type == "Lateral Flame Spread" and len(points) == 4:
                dewarp_generator = dewarp_lateral_flame_spread(
                    experiment=self.experiment,
                    points=points,
                    config=cfg,
                )
            else:
                QMessageBox.warning(
                    self,
                    "Invalid experiment type or point count",
                    f"{experiment_type} requires exactly 4 or 6 points.",
                )
                return

            self.ui.button_dewarp.setDisabled(True)
            frame_count = self.experiment.get_data(DATATYPE).get_frame_count()
            self.ui.progress_dewarping.setRange(0, frame_count - 1)

            console_bar = progressbar.ProgressBar(
                max_value=frame_count,
                widgets=[
                    "Dewarping: ",
                    progressbar.Percentage(),
                    " ",
                    progressbar.Bar(marker="█", left="[", right="]"),
                    " ",
                    progressbar.ETA(),
                ],
            )

            for progress in dewarp_generator:
                self.ui.progress_dewarping.setValue(progress)
                console_bar.update(progress)
                QApplication.processEvents()

            console_bar.finish()
            self.ui.button_dewarp.setDisabled(False)

            if hasattr(self.experiment, "h5_file"):
                self.experiment.h5_path = self.experiment.h5_file.filename

            self.update_edge_preview()

        except FileExistsError as err:
            choice = QMessageBox.question(
                self,
                "File exists",
                f"{err}\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if choice == QMessageBox.Yes:
                os.remove(str(err))
                self.on_dewarp_clicked()
            return
        except (OSError, ValueError, KeyError) as err:
            logging.error("Unexpected error during dewarping: %s", err)
            self.ui.button_dewarp.setDisabled(False)
            QMessageBox.critical(self, "Error", f"An unexpected error occurred:\n{err}")
            return

    def start_edge_detection(self) -> None:
        """Start edge detection with multi-threading support."""

        self._disable_ui_while_edge_detecting()

        frame_count = self.experiment.get_data(DATATYPE).get_frame_count()

        if self.experiment_type == "Lateral Flame Spread":
            if not hasattr(self, "console_bar") or self.console_bar is None:
                frame_count = self.experiment.get_data(DATATYPE).get_frame_count()
                print(frame_count)
                if frame_count <= 0:
                    logging.error("Frame count is 0 – cannot start edge detection.")
                    QMessageBox.critical(
                        self, "Error", "No frames found for edge detection."
                    )
                    return

                self.console_bar = self._create_progress_bar(
                    "Edge Finding: ", max_value=frame_count
                )
                self.console_bar_started = False
                print(
                    f"[DEBUG] Created progress bar with max_value={self.console_bar.max_value}"
                )

            self.ui.progress_edge_finding_plate1.setRange(0, 100)
            self.ui.progress_edge_finding_plate1.setValue(0)

            self.thread = QThread()
            flame_dir = self.ui.comboBox_flame_direction.currentText()
            method_fn = (
                left_most_point_over_threshold
                if flame_dir == "Right -> Left"
                else right_most_point_over_threshold
            )

            flame_dir_key = (
                "right_to_left" if flame_dir == "Right -> Left" else "left_to_right"
            )

            self.worker = EdgeDetectionWorker(
                h5_path=self.experiment.h5_path,
                dataset_key="dewarped_data/data",
                result_key="edge_results",
                threshold=280,
                method=lambda y, params=None: method_fn(y, threshold=280),
                flame_direction=flame_dir_key,
            )
            self._setup_edge_worker(self.thread, self.worker, "lfs")
            self.thread.start()
            return

        # Room Corner: separate threads for left and right
        if not hasattr(self, "console_bar_left") or self.console_bar_left is None:
            frame_count = self.experiment.get_data(DATATYPE).get_frame_count()

            self.console_bar_left = self._create_progress_bar(
                "Edge Left: ", max_value=frame_count
            )

        self.console_bar_right = self._create_progress_bar(
            "Edge Right: ", max_value=frame_count
        )
        if not hasattr(self, "console_bar_right") or self.console_bar_right is None:
            self.console_bar_right = self._create_progress_bar(
                "Edge Right: ", max_value=frame_count
            )

        self.console_bar_left.start()
        self.console_bar_right.start()

        self.ui.progress_edge_finding_plate1.setRange(0, 100)
        self.ui.progress_edge_finding_plate1.setValue(0)
        self.ui.progress_edge_finding_plate2.setRange(0, 100)
        self.ui.progress_edge_finding_plate2.setValue(0)

        # Left worker/thread
        self.thread_left = QThread()
        self.worker_left = EdgeDetectionWorker(
            h5_path=self.experiment.h5_path,
            dataset_key="dewarped_data_left/data",
            result_key="edge_results_left/data",
            threshold=280,
            method=lambda y, params=None: left_most_point_over_threshold(
                y, threshold=280
            ),
        )
        self._setup_edge_worker(self.thread_left, self.worker_left, "left")

        # Right worker/thread
        self.thread_right = QThread()
        self.worker_right = EdgeDetectionWorker(
            h5_path=self.experiment.h5_path,
            dataset_key="dewarped_data_right/data",
            result_key="edge_results_right/data",
            threshold=280,
            method=lambda y, params=None: right_most_point_over_threshold(
                y, threshold=280
            ),
        )
        self._setup_edge_worker(self.thread_right, self.worker_right, "right")

        if self.ui.checkBox_mulithread.isChecked():
            self.thread_left.start()
            self.thread_right.start()
        else:
            self.worker_left.finished.connect(self.thread_right.start)
            self.thread_left.start()

        y_cutoff = self.ui.slider_analysis_y.value() / 100
        self.ui.plot_analysis.plot_edge_results(self.experiment, y_cutoff=y_cutoff)

    def _disable_ui_while_edge_detecting(self) -> None:
        """Disable UI elements during edge detection to avoid user interference."""
        self.ui.button_open_folder.setEnabled(False)
        self.ui.button_find_edge.setEnabled(False)
        self.ui.comboBox_experiment_type.setEnabled(False)
        self.ui.checkBox_mulithread.setEnabled(False)

    def _create_progress_bar(
        self, label: str, max_value: int
    ) -> progressbar.ProgressBar:
        widgets = [
            label,
            progressbar.Percentage(),
            " ",
            progressbar.Bar(marker="█", left="[", right="]"),
            " ",
            progressbar.ETA(),
        ]
        return progressbar.ProgressBar(max_value=max_value, widgets=widgets)

    def _setup_edge_worker(
        self, thread: QThread, worker: EdgeDetectionWorker, side: str
    ) -> None:
        """Helper to set up worker-thread connections for edge detection."""
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(getattr(self, f"update_edge_progress_{side}"))
        worker.finished.connect(lambda result, _: self.handle_edge_result(result, side))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

    def handle_edge_result(self, result_array: np.ndarray, side: str) -> None:
        """Save detected edge data to HDF5 and manage progress UI."""
        with h5py.File(self.experiment.h5_path, "a") as f:
            if side == "lfs":
                group = f.require_group("edge_results")
                if self.console_bar:
                    self.console_bar.finish()
                    self.console_bar = None
                    self.console_bar_started = False
                self.ui.progress_edge_finding_plate1.setValue(100)

            else:
                group = f.require_group(f"edge_results_{side}")
                if side == "left" and self.console_bar_left:
                    self.console_bar_left.finish()
                    self.console_bar_left = None
                    self.ui.progress_edge_finding_plate1.setValue(100)
                elif side == "right" and self.console_bar_right:
                    self.console_bar_right.finish()
                    self.console_bar_right = None
                    self.ui.progress_edge_finding_plate2.setValue(100)

            if "data" in group:
                del group["data"]
            group.create_dataset("data", data=result_array)

        logging.info("Edge detection finished for %s side.", side.upper())

        self.edge_workers_done += 1
        if (
            self.experiment_type == "Lateral Flame Spread"
            and self.edge_workers_done == 1
        ) or (self.experiment_type == "Room Corner" and self.edge_workers_done == 2):
            self.enable_analysis_controls()

    def enable_analysis_controls(self) -> None:
        """Enable UI controls after edge detection is complete."""
        self.ui.button_open_folder.setEnabled(True)
        self.ui.button_find_edge.setEnabled(True)
        self.ui.comboBox_experiment_type.setEnabled(True)
        self.ui.checkBox_mulithread.setEnabled(True)
        self.edge_workers_done = 0

    def update_edge_progress_lfs(self, value: int) -> None:
        # import threading
        # print(f"[{threading.current_thread().name}] update_edge_progress_lfs CALLED WITH VALUE:", value)
        self.ui.progress_edge_finding_plate1.setValue(value)
        if hasattr(self, "console_bar") and self.console_bar:
            if not getattr(self, "console_bar_started", False):
                self.console_bar.start()
                self.console_bar_started = True
            # print(f"Console bar max: {self.console_bar.max_value}, value: {value}")
            self.console_bar.update(value)

    def update_edge_progress_left(self, value: int) -> None:
        self.ui.progress_edge_finding_plate1.setValue(value)
        if hasattr(self, "console_bar_left") and self.console_bar_left:
            self.console_bar_left.update(value)

    def update_edge_progress_right(self, value: int) -> None:
        self.ui.progress_edge_finding_plate2.setValue(value)
        if hasattr(self, "console_bar_right") and self.console_bar_right:
            self.console_bar_right.update(value)

    def update_edge_preview(self) -> None:
        """Update edge preview plot for current experiment and settings."""
        if not self.experiment:
            return

        try:
            frame_count = self.experiment.get_data(DATATYPE).get_frame_count()
            frame_index = frame_count // 2
            h5 = self.experiment.h5_file

            if self.experiment_type == "Lateral Flame Spread":
                if "dewarped_data" not in h5:
                    logging.debug("No dewarped_data for LFS - skipping preview.")
                    return
                dataset = h5["dewarped_data"]["data"]
                is_left = False
            elif self.experiment_type == "Room Corner":
                if "dewarped_data_left" not in h5:
                    logging.debug(
                        "No dewarped_data_left for Room Corner - skipping preview."
                    )
                    return
                dataset = h5["dewarped_data_left"]["data"]
                is_left = True
            else:
                logging.debug("Unknown experiment type: %s", self.experiment_type)
                return

            frame = dataset[:, :, frame_index]

            threshold = 280  # Default threshold

            def edge_left(y, params=None):
                return left_most_point_over_threshold(y, threshold=threshold)

            def edge_right(y, params=None):
                return right_most_point_over_threshold(y, threshold=threshold)

            def identity_filter(x):
                return x

            if self.experiment_type == "Lateral Flame Spread":
                flame_dir = self.ui.comboBox_flame_direction.currentText()
                if flame_dir == "Right -> Left":
                    edge_method = edge_left
                else:
                    edge_method = edge_right
            else:
                edge_method = edge_left

            edge = calculate_edge_data(
                data=dataset[:, :, frame_index : frame_index + 1],
                find_edge_point=edge_method,
                custom_filter=identity_filter,
            )[0]

            self.ui.plot_edge_preview.plot_with_edge(frame, edge, cmin=0.0, cmax=1.0)
            logging.debug(
                "Showing dataset shape: %s, from: %s",
                dataset.shape,
                "dewarped_data_left" if is_left else "dewarped_data",
            )

        except (
            KeyError,
            IndexError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
        ) as err:
            # HDF5-Schlüssel fehlt, Frameindex out-of-range, falsche Typen/Werte, IO
            logging.debug("Could not load edge preview: %s", err)

    def update_flame_direction_visibility(self) -> None:
        """Show or hide flame direction controls based on experiment type."""
        is_lfs = (
            self.ui.comboBox_experiment_type.currentText() == "Lateral Flame Spread"
        )
        self.ui.comboBox_flame_direction.setVisible(is_lfs)

    def update_analysis_plot(self):
        if self.experiment is None:
            return

        y_cutoff = self.ui.slider_analysis_y.value() / 100
        self.ui.plot_analysis.plot_edge_results(self.experiment, y_cutoff)

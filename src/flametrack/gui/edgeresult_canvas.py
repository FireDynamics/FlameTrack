import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget


def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply a simple moving average smoothing to a 1D numpy array.

    Args:
        y: Input 1D numpy array.
        window: Window size for the moving average.

    Returns:
        Smoothed numpy array of same length as input.
    """
    if window < 2:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


class EdgeResultCanvas(QWidget):
    """
    Widget to display edge tracking results using PyQtGraph.

    Supports plotting for both Lateral Flame Spread and Room Corner experiments.
    Includes optional smoothing and visual error bands.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Configure pyqtgraph for scientific plot style
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.plot_widget = pg.PlotWidget()

        # Configure grid and axis labels with font styling
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.getAxis("left").setLabel(
            "Edge Position [mm]", **{"font-size": "10pt"}
        )
        self.plot_widget.getAxis("bottom").setLabel("Frame", **{"font-size": "10pt"})
        self.plot_widget.getAxis("left").setStyle(tickFont=QFont("Helvetica", 9))
        self.plot_widget.getAxis("bottom").setStyle(tickFont=QFont("Helvetica", 9))

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def plot_edge_results(
        self,
        experiment,
        y_cutoff: float = 0.5,
        smooth: bool = True,
        smooth_window: int = 7,
    ) -> None:
        """
        Plot edge tracking results from an experiment's HDF5 data.

        Args:
            experiment: Experiment object with attribute `h5_file`.
            y_cutoff: Vertical position ratio (0-1) to select the y-line for plotting.
            smooth: Whether to apply moving average smoothing.
            smooth_window: Window size for smoothing.
        """
        if experiment is None:
            return

        self.plot_widget.clear()
        legend = self.plot_widget.addLegend()
        legend.setBrush(pg.mkBrush(255, 255, 255, 180))  # semi-transparent white

        h5 = experiment.h5_file
        exp_type = getattr(experiment, "experiment_type", "Room Corner")

        try:
            if exp_type == "Lateral Flame Spread":
                data = h5["edge_results"]["data"][:]
                y_index = int(data.shape[1] * y_cutoff)
                y_index = max(-y_index - 1, -data.shape[1])

                edge = -data[:, y_index] + np.max(data[:, y_index])

                flame_direction = h5["edge_results"].attrs.get(
                    "flame_direction", "right_to_left"
                )

                if flame_direction == "left_to_right":
                    edge = edge.max() - edge

                if smooth:
                    edge = moving_average(edge, window=smooth_window)

                x_vals = np.arange(len(edge))

                width = h5["dewarped_data"].attrs["target_pixels_width"]
                err = h5["dewarped_data"].attrs.get("error_mm_width", None)

                self.plot_widget.plot(
                    x_vals, edge, pen=pg.mkPen((0, 100, 180), width=2), name="Edge"
                )

                if err is not None:
                    band_upper = edge + err
                    band_lower = edge - err

                    upper_curve = pg.PlotCurveItem(x_vals, band_upper, pen=None)
                    lower_curve = pg.PlotCurveItem(x_vals, band_lower, pen=None)
                    self.plot_widget.addItem(upper_curve)
                    self.plot_widget.addItem(lower_curve)

                    fill = pg.FillBetweenItem(
                        curve1=upper_curve,
                        curve2=lower_curve,
                        brush=QColor(0, 100, 180, 50),
                    )
                    self.plot_widget.addItem(fill)
                else:
                    print("[WARN] No error_mm_width in dewarped_data")

            else:  # Room Corner experiment
                d1 = h5["edge_results_left"]["data"][:]
                d2 = h5["edge_results_right"]["data"][:]
                y1 = int(d1.shape[1] * y_cutoff)
                y2 = int(d2.shape[1] * y_cutoff)
                y1 = max(-y1 - 1, -d1.shape[1])
                y2 = max(-y2 - 1, -d2.shape[1])

                edge1 = -d1[:, y1] + np.max(d1[:, y1])
                edge2 = -d2[:, y2] + np.max(d2[:, y2])
                edge2 = edge2.max() - edge2  # mirror right side

                if smooth:
                    edge1 = moving_average(edge1, window=smooth_window)
                    edge2 = moving_average(edge2, window=smooth_window)

                width1 = h5["dewarped_data_left"].attrs["target_pixels_width"]
                width2 = h5["dewarped_data_right"].attrs["target_pixels_width"]
                target_width = max(width1, width2)

                edge1 = edge1.astype(float) * target_width / width1
                edge2 = edge2.astype(float) * target_width / width2

                x_vals = np.arange(len(edge1))

                self.plot_widget.plot(
                    x_vals, edge1, pen=pg.mkPen((220, 20, 60), width=2), name="Left"
                )
                self.plot_widget.plot(
                    x_vals,
                    edge2,
                    pen=pg.mkPen((0, 0, 128), width=2),
                    name="Right (mirrored)",
                )

                err1 = h5["dewarped_data_left"].attrs.get("error_mm_width", None)
                err2 = h5["dewarped_data_right"].attrs.get("error_mm_width", None)

                if err1 is not None:
                    band1_upper = edge1 + err1
                    band1_lower = edge1 - err1

                    upper_curve1 = pg.PlotCurveItem(x_vals, band1_upper, pen=None)
                    lower_curve1 = pg.PlotCurveItem(x_vals, band1_lower, pen=None)
                    self.plot_widget.addItem(upper_curve1)
                    self.plot_widget.addItem(lower_curve1)

                    fill_left = pg.FillBetweenItem(
                        curve1=upper_curve1,
                        curve2=lower_curve1,
                        brush=QColor(220, 20, 60, 50),
                    )
                    self.plot_widget.addItem(fill_left)
                else:
                    print("[WARN] No error_mm_width in dewarped_data_left")

                if err2 is not None:
                    band2_upper = edge2 + err2
                    band2_lower = edge2 - err2

                    upper_curve2 = pg.PlotCurveItem(x_vals, band2_upper, pen=None)
                    lower_curve2 = pg.PlotCurveItem(x_vals, band2_lower, pen=None)
                    self.plot_widget.addItem(upper_curve2)
                    self.plot_widget.addItem(lower_curve2)

                    fill_right = pg.FillBetweenItem(
                        curve1=upper_curve2,
                        curve2=lower_curve2,
                        brush=QColor(0, 0, 128, 50),
                    )
                    self.plot_widget.addItem(fill_right)
                else:
                    print("[WARN] No error_mm_width in dewarped_data_right")

            self.plot_widget.setTitle(
                f"Edge progression at y = {y_cutoff:.0%} (0 = plate start)",
                size="10pt",
            )

        except KeyError as e:
            self.plot_widget.setTitle("⚠️ Edge result data not found")
            print(f"[DEBUG] plot_edge_results KeyError: {e}")

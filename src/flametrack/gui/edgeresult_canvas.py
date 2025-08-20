import logging

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _y_index(self, height: int, y_cutoff: float) -> int:
        """Mappe y_cutoff (0..1) auf einen negativen Index von unten."""
        idx = int(height * y_cutoff)
        return max(-idx - 1, -height)

    def _add_error_band(
        self,
        x_vals: np.ndarray,
        center: np.ndarray,
        err_mm: float | None,
        color_rgba: tuple[int, int, int, int],
    ) -> None:
        """Zeichnet optional einen ±err Band um die Kurve."""
        if err_mm is None:
            return
        upper = center + err_mm
        lower = center - err_mm
        upper_curve = pg.PlotCurveItem(x_vals, upper, pen=None)
        lower_curve = pg.PlotCurveItem(x_vals, lower, pen=None)
        self.plot_widget.addItem(upper_curve)
        self.plot_widget.addItem(lower_curve)
        self.plot_widget.addItem(
            pg.FillBetweenItem(
                curve1=upper_curve, curve2=lower_curve, brush=QColor(*color_rgba)
            )
        )

    def _plot_lfs(self, h5, y_cutoff: float, smooth: bool, smooth_window: int) -> None:
        data = h5["edge_results"]["data"][:]
        y_idx = self._y_index(data.shape[1], y_cutoff)

        # Edge-Kurve (von unten gemessen, aufsteigend)
        edge = -data[:, y_idx] + np.max(data[:, y_idx])

        # Flammenrichtung berücksichtigen
        flame_dir = h5["edge_results"].attrs.get("flame_direction", "right_to_left")
        if flame_dir == "left_to_right":
            edge = edge.max() - edge

        if smooth:
            edge = moving_average(edge, window=smooth_window)

        x_vals = np.arange(len(edge))
        self.plot_widget.plot(
            x_vals, edge, pen=pg.mkPen((0, 100, 180), width=2), name="Edge"
        )

        # Fehlerband (falls vorhanden)
        err = h5["dewarped_data"].attrs.get("error_mm_width", None)
        self._add_error_band(x_vals, edge, err, (0, 100, 180, 50))

    def _plot_room_corner(
        self, h5, y_cutoff: float, smooth: bool, smooth_window: int
    ) -> None:
        d_left = h5["edge_results_left"]["data"][:]
        d_right = h5["edge_results_right"]["data"][:]

        y1 = self._y_index(d_left.shape[1], y_cutoff)
        y2 = self._y_index(d_right.shape[1], y_cutoff)

        edge1 = -d_left[:, y1] + np.max(d_left[:, y1])
        edge2 = -d_right[:, y2] + np.max(d_right[:, y2])
        edge2 = edge2.max() - edge2  # rechts spiegeln

        if smooth:
            edge1 = moving_average(edge1, window=smooth_window)
            edge2 = moving_average(edge2, window=smooth_window)

        # auf gemeinsame Zielbreite normieren
        w1 = int(h5["dewarped_data_left"].attrs["target_pixels_width"])
        w2 = int(h5["dewarped_data_right"].attrs["target_pixels_width"])
        target_w = max(w1, w2)
        edge1 = edge1.astype(float) * target_w / w1
        edge2 = edge2.astype(float) * target_w / w2

        x_vals = np.arange(len(edge1))
        self.plot_widget.plot(
            x_vals, edge1, pen=pg.mkPen((220, 20, 60), width=2), name="Left"
        )
        self.plot_widget.plot(
            x_vals, edge2, pen=pg.mkPen((0, 0, 128), width=2), name="Right (mirrored)"
        )

        # Fehlerbänder (falls vorhanden)
        err1 = h5["dewarped_data_left"].attrs.get("error_mm_width", None)
        try:
            # Tippfehler-Variante robust abfangen
            err2 = h5["dewarwed_data_right"].attrs.get("error_mm_width", None)
        except KeyError:
            err2 = h5["dewarped_data_right"].attrs.get("error_mm_width", None)

        self._add_error_band(x_vals, edge1, err1, (220, 20, 60, 50))
        self._add_error_band(x_vals, edge2, err2, (0, 0, 128, 50))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_edge_results(
        self,
        experiment,
        y_cutoff: float = 0.5,
        smooth: bool = True,
        smooth_window: int = 7,
    ) -> None:
        """Plot edge tracking results from an experiment's HDF5 data."""
        if experiment is None:
            return

        self.plot_widget.clear()
        legend = self.plot_widget.addLegend()
        legend.setBrush(pg.mkBrush(255, 255, 255, 180))

        h5 = experiment.h5_file
        exp_type = getattr(experiment, "experiment_type", "Room Corner")

        try:
            if exp_type == "Lateral Flame Spread":
                self._plot_lfs(h5, y_cutoff, smooth, smooth_window)
            else:
                self._plot_room_corner(h5, y_cutoff, smooth, smooth_window)

            self.plot_widget.setTitle(
                f"Edge progression at y = {y_cutoff:.0%} (0 = plate start)",
                size="10pt",
            )
        except KeyError as err:
            self.plot_widget.setTitle("⚠️ Edge result data not found")
            # bewusst kein f-string in logging (Pylint W1203)
            # (hier print ist ok, weil GUI-Widget; du kannst auch logging benutzen)
            logging.debug("plot_edge_results KeyError: %s", err)

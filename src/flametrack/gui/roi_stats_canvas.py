from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class RoiStatsCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("ROI Mean Temperature Over Time")
        self.plot_widget.setLabel("bottom", "Frame")
        self.plot_widget.setLabel("left", "Mean Temperature")
        self.plot_widget.addLegend()
        layout.addWidget(self.plot_widget)

    def plot_series(self, series: dict[str, dict[str, list[float]]]) -> None:
        self.plot_widget.clear()
        legend = self.plot_widget.addLegend()
        if legend is not None:
            legend.clear()
        for i, (region_id, values) in enumerate(series.items()):
            color = pg.intColor(i, hues=max(len(series), 6))
            self.plot_widget.plot(
                values["mean"], pen=pg.mkPen(color, width=2), name=region_id
            )

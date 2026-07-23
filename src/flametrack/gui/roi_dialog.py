from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)

from ..analysis.region_manager import RegionManager
from .regions_canvas import RegionCanvas


class ROIEditorDialog(QDialog):
    """
    ROI editor dialog. Displays a RegionCanvas over the given image,
    seeded from a private copy of the passed-in RegionManager so edits
    made don't affect the manager unless Done is pressed.
    """

    def __init__(self, image, manager: RegionManager, parent=None):
        """
        Building of the dialog, with all necesary controls.
        """
        super().__init__(parent)
        self.setWindowTitle("Region Of Interest Editor")
        self.resize(1000, 700)

        self._manager = manager.copy()

        self.canvas = RegionCanvas()
        self.canvas.plot(image, cmin=0.0, cmax=1.0)
        self.canvas.set_manager(self._manager)

        self.button_done = QPushButton("Done")
        self.button_cancel = QPushButton("Cancel")

        btn_rect = QPushButton("Add rectangle")
        btn_poly = QPushButton("Add polygon")
        btn_clear = QPushButton("Clear all")

        btn_rect.clicked.connect(self.canvas.add_rectangle)
        btn_poly.clicked.connect(self.canvas.add_polygon)
        btn_clear.clicked.connect(self.canvas.clear_regions)

        shape_row = QHBoxLayout()
        for btn in (btn_rect, btn_poly, btn_clear):
            shape_row.addWidget(btn)

        confirm_row = QHBoxLayout()
        confirm_row.addStretch()
        confirm_row.addWidget(self.button_cancel)
        confirm_row.addWidget(self.button_done)

        layout = QVBoxLayout()
        layout.addLayout(shape_row)
        layout.addWidget(self.canvas)
        layout.addLayout(confirm_row)

        self.button_cancel.clicked.connect(self.reject)
        self.button_done.clicked.connect(self.accept)
        self.setLayout(layout)

    def result_manager(self) -> RegionManager:
        return self._manager

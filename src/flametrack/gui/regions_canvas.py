from __future__ import annotations

import itertools
import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QInputDialog, QMenu

from flametrack.analysis.emissivity_correction import correct_temperature, region_mask
from flametrack.analysis.region import Region, RegionShape
from flametrack.analysis.region_manager import RegionManager

from .imshow_canvas import ImshowCanvas

_DEFAULT_COLORS: list[tuple[int, int, int]] = [(0, 0, 0)]


class RegionCanvas(ImshowCanvas):
    """
    Interactive canvas for drawing, editing, and inspecting regions of interest (ROIs) on top of a displayed IR frame.
    Wraps ImShowCanvas with pyqtgraph ROI widgets (rectangle / polygon [could add free hand drawing later]), backed y a RegionManager that owns the actual geo/emissivity data independent of the Qt widgets.
    """

    region_added = Signal(str)
    region_removed = Signal(str)
    region_changed = Signal(str)
    region_selected = Signal(str)

    def __init__(self, parent=None):
        """
        Set up ROI/manager state, focus handling, and mouse-hover tracking.
        """

        super().__init__(parent)
        self.manager = RegionManager()
        self._roi_items: dict[str, pg.ROI] = {}
        self._color_cycle = itertools.cycle(_DEFAULT_COLORS)
        self._selected_region_id: str | None = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.plot_widget.setFocusPolicy(Qt.NoFocus)

        self._cmin = 0.0
        self._cmax = 1.0
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._raw_frame: np.ndarray | None = None

    def plot(self, data, cmin: float, cmax: float) -> None:
        """
        Display a new frame, caching the raw data and intensity range so region-mean
        temperature lookups always use the original (uncorrected) values regardless
        of what's currently drawn.
        """
        self._cmin = cmin
        self._cmax = cmax
        self._raw_frame = data.copy()
        super().plot(data, cmin, cmax)

    def _on_mouse_moved(self, scene_pos) -> None:
        """
        Show the raw temperature at the pixel under the cursor as a tooltip.
        """
        vb = self.plot_widget.plotItem.vb
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        point = vb.mapSceneToView(scene_pos)
        x, y = int(point.x()), int(point.y())
        if self.data is None:
            return
        h, w = self.data.shape[:2]
        if 0 <= y < h and 0 <= x < w:
            self.setToolTip(f"x={x}, y={y}\nTemperature: {self.data[y, x]:.2f}")

    def add_rectangle(self, name: str | None = None) -> str:
        """
        Create a new draggable/resizable rectangle ROI and register it.
        """
        color = next(self._color_cycle)
        x0, y0, w, h = self._default_geometry()

        roi = pg.RectROI([x0, y0], [w, h], pen=pg.mkPen(color, width=3))
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])

        region = self.manager.add_rectangle(
            x0, y0, x0 + w, y0 + h, name=name, color=color
        )
        self._register_roi(region.region_id, roi)

        roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        return region.region_id

    def add_polygon(self, name: str | None = None) -> str:
        """
        Create a new draggable polygon ROI (default 4-point square) and register it, capping its point count to 8.
        """
        color = next(self._color_cycle)
        x0, y0, w, h = self._default_geometry()
        pts = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]

        #### PRINT FOR DEBUGGING
        print("DRAW: ", len(pts))

        roi = pg.PolyLineROI(pts, closed=True, pen=pg.mkPen(color, width=3))

        region = self.manager.add_polygon(
            [tuple(p) for p in pts], name=name, color=color
        )

        roi.sigRegionChanged.connect(
            lambda _roi, rid=region.region_id: self._limit_polygon_points(
                rid, max_points=8
            )
        )

        self._register_roi(region.region_id, roi)

        roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        return region.region_id

    def _limit_polygon_points(self, region_id: str, max_points: int) -> None:
        """
        Deferred handle-count cap for a polygon ROI, preventing runaway vertex
        counts from repeated clicks; deferred via QTimer to avoid mutating the ROI's
        handle list mid-signal.
        """

        def _do_remove():
            try:
                roi = self._roi_items.get(region_id)
            except RuntimeError:
                return
            if roi is None:
                return
            handles = roi.getHandles()
            pts = []

            for h in handles:
                p = roi.mapToParent(h.pos())
                pts.append((p.x(), p.y()))

            if len(handles) > max_points:
                roi.removeHandle(handles[-1])

        QTimer.singleShot(0, _do_remove)

    def _default_geometry(self) -> tuple[float, float, float, float]:
        """
        Compute a default rectangle centered on the current frame, sized proportionally to it.
        """
        if self.data is not None:
            h, w = self.data.shape[:2]
        else:
            h, w = 100, 100
        rw, rh = max(w * 0.2, 10), max(h * 0.2, 10)
        x0, y0 = (w - rw) / 2, (h - rh) / 2
        return x0, y0, rw, rh

    def _register_roi(
        self, region_id: str, roi: pg.ROI, emit_added: bool = True
    ) -> None:
        """
        Wire signal connections for a newly created/loaded ROI widget, store it,
        and optionally announce/select it as newly added.
        """
        self._roi_items[region_id] = roi
        self.plot_widget.addItem(roi)

        roi.sigRegionChangeFinished.connect(
            lambda _roi, rid=region_id: self._on_roi_changed(rid)
        )
        roi.sigClicked.connect(
            lambda _roi, ev, rid=region_id: (
                self._on_roi_clicked(rid, ev),
                print(ev.button()),
            )
        )

        self._update_tooltip(region_id)
        logging.debug("[REGION] Added %s", region_id)

        if emit_added:
            self.region_added.emit(region_id)
            self.select_region(region_id)

    def _on_roi_changed(self, region_id: str) -> None:
        """
        Sync a region's stored points after the user finishes dragging or
        reshaping its ROI, rejectng degenerate (too-few-point) polygons.
        """
        roi = self._roi_items.get(region_id)
        region = self.manager.get(region_id)

        if roi is None or region is None:
            return

        points = self._roi_to_points(roi, region.shape)
        if region.shape == RegionShape.POLYGON and len(points) < 3:
            return
        self.manager.update_points(region_id, points)
        self.region_changed.emit(region_id)
        self.select_region(region_id)

    @staticmethod
    def _roi_to_points(roi: pg.ROI, shape: RegionShape) -> list[tuple[float, float]]:
        """
        Convert a pyqtgraph ROI's current geometry into a list of image-coordinate points,
        deduplicating the closing vertex for polygons.
        """
        if shape == RegionShape.RECTANGLE:
            pos = roi.pos()
            size = roi.size()
            x0, y0 = pos.x(), pos.y()
            x1, y1 = x0 + size.x(), y0 + size.y()

            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        # Something about PolyLineROI and how no rotations are needed, and that pos() + local offset is enough
        pos = roi.pos()
        state = roi.getState()

        pts = [
            (roi.mapToParent(p).x(), roi.mapToParent(p).y()) for p in state["points"]
        ]
        ### PRINTS FOR DEBUGGING:
        print("state points:", len(state["points"]))
        print(state["points"])

        if shape == RegionShape.POLYGON and len(pts) > 1 and pts[0] == pts[-1]:
            return pts[:-1]
        return pts

    def remove_region(self, region_id: str) -> None:
        """
        Delete a region: disconnect its ROI's signals, remove it from the scene, ad drop it from the manager and
        selection state.
        """
        roi = self._roi_items.pop(region_id, None)
        if roi is not None:
            ### New addition
            try:
                roi.sigRegionChangeFinished.disconnect()
                roi.sigClicked.disconnect()
                roi.sigRegionChangeStarted.disconnect()
            except (TypeError, RuntimeError):
                pass
            ### End New Addition
            self.plot_widget.removeItem(roi)
        self.manager.remove(region_id)
        if self._selected_region_id == region_id:
            self._selected_region_id = None
        logging.debug("[REGION] removed %s", region_id)
        self.region_removed.emit(region_id)

    def clear_regions(self) -> None:
        """
        Remove every region currently on the canvas.
        """
        for region_id in list(self._roi_items):
            self.remove_region(region_id)

    def select_region(self, region_id: str | None) -> None:
        """
        Mark a region as selected, updating highlight state and emitting region_selected.
        Passing None clears the current selection.
        """
        if self._selected_region_id == region_id:
            return
        self._set_highlight(self._selected_region_id, selected=False)
        self._selected_region_id = region_id
        self.setFocus()
        self._set_highlight(region_id, selected=True)
        self.region_selected.emit(region_id or "")
        ### PRINT FOR DEBUGGING
        print("Selected", region_id)

    def _set_highlight(self, region_id: str | None, selected: bool) -> None:
        """
        Adjust a region's ROI border width to reflect selected/unselected state.
        """
        if not region_id:
            return
        roi = self._roi_items.get(region_id)
        region = self.manager.get(region_id)
        if roi is None or region is None:
            return
        roi.setPen(pg.mkPen(region.color, width=4 if selected else 2))

    def delete_selected_region(self) -> None:
        """
        Delete the currently selected region, if any.
        """
        if self._selected_region_id is not None:
            self.remove_region(self._selected_region_id)

    def keyPressEvent(self, event) -> None:
        """
        Delete the selected region on Delete/Backspace; otherwise default handling.
        """
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_region()
        else:
            super().keyPressEvent(event)

    def set_manager(self, manager: RegionManager) -> None:
        """
        Replace the backing RegionManager and rebuild ROI widgets to match its regions
        (used when reopening the ROI editor with existing data).
        """
        self.manager = manager
        self.load_from_manager()

    def load_from_manager(self) -> None:
        """
        Rebuild all ROI widgets from the current manager's regions, fully tearing down
        (and disconnecting) any existing ones first.
        """
        for region_id in list(self._roi_items):
            roi = self._roi_items.pop(region_id)
            try:
                roi.sigRegionChangeFinished.disconnect()
                roi.sigClicked.disconnect()
                roi.sigRegionChangeStarted.disconnect()
            except (TypeError, RuntimeError):
                pass

            self.plot_widget.removeItem(roi)
        self._selected_region_id = None

        for region in self.manager.all():
            roi = self._build_roi_for_region(region)
            self._register_roi(region.region_id, roi, emit_added=False)

    def _on_roi_clicked(self, region_id: str, ev):
        """
        Select the clicked region and, on right-click, open its context menu.
        """
        self.select_region(region_id)
        if ev is not None and ev.button() == Qt.RightButton:
            self._show_context_menu(region_id)

    def _show_context_menu(self, region_id: str):
        """
        Show the Delete/Duplicate/Set-Emissivity context menu for a region.
        """
        menu = QMenu(self)
        act_delete = menu.addAction("Delete Shape")
        act_duplicate = menu.addAction("Duplicate Shape")
        act_emissivity = menu.addAction("Set Emissivity")

        action = menu.exec(QCursor.pos())

        if action == act_delete:
            self.remove_region(region_id)
        elif action == act_duplicate:
            self.duplicate_region(region_id)
        elif action == act_emissivity:
            self._prompt_emissivity(region_id)

    def _prompt_emissivity(self, region_id):
        """
        Prompt for a new emissivity value [0 … 1] and apply it to the region.
        """
        region = self.manager.get(region_id)
        if region is None:
            return
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Emissivity")
        dialog.setLabelText("Emissivity")
        dialog.setDoubleDecimals(2)
        dialog.setDoubleRange(0.0, 1.0)
        dialog.setDoubleStep(0.01)
        dialog.setDoubleValue(region.emissivity)
        ok = dialog.exec()
        value = dialog.doubleValue()

        if ok:
            self.manager.set_emissivity(region_id, value)
            self._update_tooltip(region_id)

    def duplicate_region(self, region_id: str):
        """
        Create a copy of a region offset by a fixed amount, preserving its shape, color and emissivity.
        """
        region = self.manager.get(region_id)
        if region is None:
            return None
        offset_pts = [(x + 10, y + 10) for x, y in region.points]

        if region.shape == RegionShape.RECTANGLE:
            xs = [p[0] for p in offset_pts]
            ys = [p[1] for p in offset_pts]
            new_region = self.manager.add_rectangle(
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                name=f"{region.name} copy",
                color=region.color,
            )
        else:
            new_region = self.manager.add_polygon(
                offset_pts, name=f"{region.name} copy", color=region.color
            )
        new_region.emissivity = region.emissivity
        roi = self._build_roi_for_region(new_region)
        self._register_roi(new_region.region_id, roi)

        return new_region.region_id

    def _build_roi_for_region(self, region: Region):
        """
        Construct a pyqtgraph ROI widget matching an existing Region's shape and
        geometry without touching the manager.
        """
        if region.shape == RegionShape.RECTANGLE:
            xs = [p[0] for p in region.points]
            ys = [p[1] for p in region.points]
            x0, y0 = min(xs), min(ys)
            w, h = max(xs) - x0, max(ys) - y0
            roi = pg.RectROI([x0, y0], [w, h], pen=pg.mkPen(region.color, width=2))
            roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
            return roi

        roi = pg.PolyLineROI(
            [list(p) for p in region.points],
            closed=True,
            pen=pg.mkPen(region.color, width=2),
        )
        roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        roi.sigRegionChanged.connect(
            lambda _roi, rid=region.region_id: self._limit_polygon_points(
                rid, max_points=8
            )
        )
        return roi

    def _update_tooltip(self, region_id: str) -> None:
        """
        Refresh a ROI's tooltip with ID, emissivity and region's temperatures.
        """
        region = self.manager.get(region_id)
        roi = self._roi_items.get(region_id)
        if region is None or roi is None:
            return
        temp_str = ""
        if self.data is not None:
            mask = region_mask(region.points, self.data.shape[:2])
            if mask.any():
                raw_mean = float(self.data[mask].mean())
                corrected_mean = float(
                    correct_temperature(np.array([raw_mean]), region.emissivity)[0]
                )
                temp_str = (
                    f"\nRaw Temp: {raw_mean:.2f}\nCorrected Temp: {corrected_mean:.2f}"
                )
        roi.setToolTip(
            f"ID: {region.region_id}\nEmissivity: {region.emissivity}{temp_str}"
        )

    def get_regions(self) -> list[Region]:
        """
        Return all regions currently in manager.
        """
        return self.manager.all()

    def rename_region(self, region_id: str, name: str) -> None:
        """
        Rename a region by id.
        """
        self.manager.rename(region_id, name)

    def set_region_emissivity(self, region_id: str, value: float) -> None:
        """
        Set a region's emissivity value by id.
        """
        self.manager.set_emissivity(region_id, value)

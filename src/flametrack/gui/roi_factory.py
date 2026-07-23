"""
Document started to Modularize Code later …
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer

from flametrack.analysis.region import Region, RegionShape

MAX_POLYGON_POINTS = 8


def build_rect_roi(x0: float, y0: float, w: float, h: float, color) -> pg.RectROI:
    roi = pg.RectROI([x0, y0], [w, h], pen=pg.mkPen(color, width=3))
    roi.addScaleHandle([1, 1], [0, 0])
    roi.addScaleHandle([0, 0], [1, 1])
    roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
    return roi


def build_poly_roi(
    points: list[tuple[float, float]],
    color,
    on_change: Callable[[pg.ROI], None] | None = None,
) -> pg.PolyLineROI:
    roi = pg.PolyLineROI(
        [list(p) for p in points], closed=True, pen=pg.mkPen(color, width=3)
    )
    roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
    if on_change is not None:
        roi.sigRegionChanged.connect(lambda _roi: on_change(roi))
    return roi


def limit_polygon_points(
    roi_items: dict[str, pg.ROI], region_id: str, max_points: int = MAX_POLYGON_POINTS
) -> None:
    def _do_remove():
        roi = roi_items.get(region_id)
        if roi is None:
            return
        handles = roi.getHandles()
        if len(handles) > max_points:
            roi.removeHandle(handles[-1])

    QTimer.singleShot(0, _do_remove)


def default_geometry(
    data: np.ndarray | None, fallback_size: tuple[int, int] = (100, 100)
) -> tuple[float, float, float, float]:
    if data is not None:
        h, w = data.shape[:2]
    else:
        h, w = fallback_size
    rw, rh = max(w * 0.2, 10), max(h * 0.2, 10)
    x0, y0 = (w - rw) / 2, (h - rh) / 2

    return x0, y0, rw, rh


def build_roi_for_region(
    region: Region, on_polygon_change: Callable[[pg.ROI], None] | None = None
) -> pg.ROI:
    if region.shape == RegionShape.RECTANGLE:
        xs = [p[0] for p in region.points]
        ys = [p[1] for p in region.points]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0, max(ys) - y0
        return build_rect_roi(x0, y0, w, h, region.color)
    return build_poly_roi(region.points, region.color, on_change=on_polygon_change)

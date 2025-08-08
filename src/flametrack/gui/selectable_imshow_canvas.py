import logging
from typing import List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor

from .draggable_point import DraggablePoint
from .imshow_canvas import ImshowCanvas
from .plotting_utils import sort_corner_points


class SelectableImshowCanvas(ImshowCanvas):
    """
    Erweiterung von ImshowCanvas mit interaktiver Punkt-Auswahl (DraggablePoints)
    und Anzeige von Linien zwischen ausgewählten Punkten.
    """

    def __init__(self, parent: Optional[pg.GraphicsObject] = None):
        super().__init__(parent)

        self.draggable_points: List[DraggablePoint] = (
            []
        )  # Liste der interaktiven Punkte
        self.lines: Optional[pg.PlotDataItem] = (
            None  # Linienverbindung zwischen Punkten
        )

        # MouseClick-Signal für die Interaktion abonnieren
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)

        # Fokus erlauben für Tastatureingaben (z.B. Löschen)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def on_click(self, event) -> None:
        """Behandelt Mausklicks im Plotbereich zum Setzen neuer Punkte."""
        logging.debug("[DEBUG] Mouse click received")

        if self.data is None:
            logging.debug("[DEBUG] No data available – click ignored")
            return

        if self.lines is None:
            logging.debug("[DEBUG] Creating lines object")
            self.lines = pg.PlotDataItem(pen="r")
            self.plot_widget.addItem(self.lines)

        if event.button() == Qt.MouseButton.LeftButton:
            logging.debug(
                f"[DEBUG] Left click – current point count: {len(self.draggable_points)}"
            )

            # Anzahl erforderlicher Punkte abhängig vom Parent (Experiment)
            required_points = (
                self.parent.required_points
                if self.parent and hasattr(self.parent, "required_points")
                else 6
            )
            if len(self.draggable_points) >= required_points:
                print(
                    f"[INFO] Already {required_points} points set – ignoring further clicks."
                )
                return

            # Mausposition in Datenkoordinaten konvertieren
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()
            logging.debug(f"[DEBUG] Adding point at ({x:.2f}, {y:.2f})")

            # Neuen draggable Punkt erzeugen und hinzufügen
            point = DraggablePoint(x, y, parent=self)
            self.draggable_points.append(point)
            self.plot_widget.addItem(point)

            self.updateLines()

    def keyPressEvent(self, event) -> None:
        """
        Behandelt Tastendruck-Events.
        'D' löscht den nächstgelegenen Punkt,
        'C' löscht alle Punkte.
        """
        if event.key() == Qt.Key.Key_D:
            self.deleteClosestPoint()
        elif event.key() == Qt.Key.Key_C:
            self.clear_points()

    def deleteClosestPoint(self) -> None:
        """Löscht den dem Mauszeiger nächstgelegenen Punkt."""
        if not self.draggable_points:
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(
            self.mapFromGlobal(QCursor.pos())
        )

        closest_point = None
        min_dist = float("inf")

        # Nächsten Punkt anhand der Manhattan-Distanz finden
        for point in self.draggable_points:
            dist = (point.scatter_points[0] - mouse_point).manhattanLength()
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        if closest_point:
            self.plot_widget.removeItem(closest_point)
            self.draggable_points.remove(closest_point)

        self.updateLines()

    def updateLines(self) -> None:
        """
        Zeichnet Linien zwischen den gesetzten Punkten.
        Sortiert Punkte je nach Experimenttyp.
        """
        if len(self.draggable_points) < 2:
            if self.lines:
                self.lines.setData([], [])
            return

        points: List[Tuple[float, float]] = [
            (point.scatter_points[0].x(), point.scatter_points[0].y())
            for point in self.draggable_points
        ]

        experiment_type = (
            self.parent.experiment_type
            if self.parent and hasattr(self.parent, "experiment_type")
            else "Room Corner"
        )

        try:
            # Punkte sortieren (Standard: gegen den Uhrzeigersinn)
            sorted_points = sort_corner_points(
                points, experiment_type=experiment_type, direction="anticlockwise"
            )
            sorted_points.append(sorted_points[0])  # Polygon schließen
        except ValueError as e:
            logging.debug(f"[WARNING] Cannot update lines: {e}")
            return

        x, y = zip(*sorted_points)
        self.lines.setData(x=x, y=y)

    def plot(self, data, cmin=0.0, cmax=1.0) -> None:
        """Zeichnet das Bild und setzt ggf. die interaktiven Punkte und Linien."""
        super().plot(data, cmin, cmax)

        vb = self.plot_widget.plotItem.vb

        # Punkte erneut hinzufügen, falls nicht vorhanden
        for point in self.draggable_points:
            if point not in vb.addedItems:
                vb.addItem(point)

        # Linien erneut hinzufügen und aktualisieren
        if self.lines:
            if self.lines in vb.addedItems:
                vb.removeItem(self.lines)
            vb.addItem(self.lines)
            self.updateLines()
        else:
            logging.debug("[DEBUG] No lines instance present")

    def clear_points(self) -> None:
        """Löscht alle gesetzten Punkte und die Linien."""
        for p in self.draggable_points:
            self.plot_widget.removeItem(p)
        self.draggable_points.clear()

        if self.lines:
            self.plot_widget.removeItem(self.lines)
            self.lines = None

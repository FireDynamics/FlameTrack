from typing import Optional

import numpy as np
import pyqtgraph as pg

from .imshow_canvas import ImshowCanvas


class EdgePreviewCanvas(ImshowCanvas):
    """
    Erweiterung von ImshowCanvas, um Flammenkanten in einem Plot darzustellen.
    """

    def __init__(self, parent: Optional[object] = None):
        """
        Initialisiert den EdgePreviewCanvas.

        Args:
            parent: Optionaler Parent-Widget.
        """
        super().__init__(parent)
        self._edge_plot = None

    def plot_with_edge(
        self, image: np.ndarray, edge: np.ndarray, cmin: float = 0.0, cmax: float = 1.0
    ) -> None:
        """
        Zeigt das Bild an und zeichnet die Flammenkante als Linie.

        Args:
            image: 2D Array mit dem Bild.
            edge: 1D Array mit Kantenpositionen (x-Koordinate für jeden y-Wert).
            cmin: Minimaler Farbwert für Bildskalierung.
            cmax: Maximaler Farbwert für Bildskalierung.
        """
        # Bild plotten (Basisfunktion)
        super().plot(image, cmin, cmax)

        # Falls vorherige Kante existiert, diese entfernen
        if self._edge_plot and self._edge_plot in self.plot_widget.listDataItems():
            self.plot_widget.removeItem(self._edge_plot)

        # Neue Kante plotten (cyanfarbene Linie, 2 Pixel breit)
        y_vals = np.arange(len(edge))
        x_vals = np.array(edge)

        self._edge_plot = self.plot_widget.plot(
            x=x_vals, y=y_vals, pen=pg.mkPen(color="c", width=2), name="Edge"
        )

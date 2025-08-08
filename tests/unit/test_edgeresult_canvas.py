import numpy as np
import pytest

from flametrack.gui.edgeresult_canvas import EdgeResultCanvas


@pytest.fixture
def canvas(qtbot):
    widget = EdgeResultCanvas()
    qtbot.addWidget(widget)
    return widget


def test_plot_edge_results_no_experiment(canvas):
    # Aufruf mit None (soll nichts machen, kein Fehler)
    canvas.plot_edge_results(None)
    assert canvas.plot_widget.listDataItems() == []


def test_plot_edge_results_lfs_experiment(canvas):
    """LFS mit flame_direction=right_to_left."""

    class DummyAttrs:
        def __getitem__(self, key):
            if key == "target_pixels_width":
                return 100
            raise KeyError()

        def get(self, key, default=None):
            if key == "error_mm_width":
                return 1.5
            return default

    class DummyGroup:
        attrs = DummyAttrs()

    class EdgeResultsAttrs:
        def get(self, key, default=None):
            if key == "flame_direction":
                return "right_to_left"
            return default

    class DummyEdgeResults:
        attrs = EdgeResultsAttrs()

        def __getitem__(self, key):
            if key == "data":
                return np.random.rand(10, 5)
            raise KeyError()

    class DummyH5:
        def __getitem__(self, key):
            if key == "edge_results":
                return DummyEdgeResults()
            if key == "dewarped_data":
                return DummyGroup()
            raise KeyError()

    class DummyExp:
        h5_file = DummyH5()
        experiment_type = "Lateral Flame Spread"

    canvas.plot_edge_results(DummyExp())
    assert len(canvas.plot_widget.listDataItems()) > 0


def test_plot_edge_results_lfs_left_to_right(canvas):
    """LFS mit flame_direction=left_to_right."""

    class DummyAttrs:
        def __getitem__(self, key):
            return 100

        def get(self, key, default=None):
            return None

    class DummyGroup:
        attrs = DummyAttrs()

    class EdgeResultsAttrs:
        def get(self, key, default=None):
            if key == "flame_direction":
                return "left_to_right"
            return default

    class DummyEdgeResults:
        attrs = EdgeResultsAttrs()

        def __getitem__(self, key):
            if key == "data":
                return np.random.rand(8, 4)
            raise KeyError()

    class DummyH5:
        def __getitem__(self, key):
            if key == "edge_results":
                return DummyEdgeResults()
            if key == "dewarped_data":
                return DummyGroup()
            raise KeyError()

    class DummyExp:
        h5_file = DummyH5()
        experiment_type = "Lateral Flame Spread"

    canvas.plot_edge_results(DummyExp())
    assert len(canvas.plot_widget.listDataItems()) > 0


def test_plot_edge_results_room_corner(canvas):
    """Room-Corner Plot mit beiden Platten."""

    class DummyAttrs:
        def __getitem__(self, key):
            return 100

        def get(self, key, default=None):
            return 2.0

    class DummyGroup:
        attrs = DummyAttrs()

    class DummyH5:
        def __getitem__(self, key):
            if key.startswith("edge_results_"):

                class DummyEdgeResults:
                    attrs = {}

                    def __getitem__(self, subkey):
                        if subkey == "data":
                            return np.random.rand(6, 3)
                        raise KeyError()

                return DummyEdgeResults()
            if key.startswith("dewarped_data_"):
                return DummyGroup()
            raise KeyError()

    class DummyExp:
        h5_file = DummyH5()
        experiment_type = "Room Corner"

    canvas.plot_edge_results(DummyExp())
    assert len(canvas.plot_widget.listDataItems()) > 0


def test_plot_edge_results_missing_flame_direction(canvas):
    """LFS ohne flame_direction -> Default verwenden."""

    class DummyAttrs:
        def __getitem__(self, key):
            return 100

        def get(self, key, default=None):
            return None

    class DummyGroup:
        attrs = DummyAttrs()

    class EdgeResultsAttrs:
        def get(self, key, default=None):
            return None  # kein flame_direction

    class DummyEdgeResults:
        attrs = EdgeResultsAttrs()

        def __getitem__(self, key):
            if key == "data":
                return np.random.rand(5, 2)
            raise KeyError()

    class DummyH5:
        def __getitem__(self, key):
            if key == "edge_results":
                return DummyEdgeResults()
            if key == "dewarped_data":
                return DummyGroup()
            raise KeyError()

    class DummyExp:
        h5_file = DummyH5()
        experiment_type = "Lateral Flame Spread"

    canvas.plot_edge_results(DummyExp())
    assert len(canvas.plot_widget.listDataItems()) > 0


def test_plot_edge_results_keyerror(canvas):
    """KeyError im Zugriff -> Warnsymbol im Titel."""

    class DummyExp:
        h5_file = {}
        experiment_type = "Lateral Flame Spread"

    canvas.plot_edge_results(DummyExp())  # Soll KeyError fangen und Titel setzen
    title_text = canvas.plot_widget.plotItem.titleLabel.text
    assert "⚠️" in title_text

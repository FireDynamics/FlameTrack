import numpy as np
import pytest

from flametrack.gui.edge_preview_canvas import EdgePreviewCanvas


@pytest.fixture
def canvas(qtbot):
    widget = EdgePreviewCanvas()
    qtbot.addWidget(widget)
    return widget


def test_plot_with_edge_adds_edge_line(canvas):
    # Erzeuge ein Dummy-Bild und eine Dummy-Kante
    image = np.random.rand(100, 100)
    edge = np.linspace(10, 90, 100)  # Kante von x=10 bis x=90 über y=0..99

    # Plot ohne Fehler aufrufen
    canvas.plot_with_edge(image, edge)

    # Prüfen, ob der Edge-Plot in der Liste der DataItems ist
    data_items = canvas.plot_widget.listDataItems()
    assert any(item.name() == "Edge" for item in data_items)

    # Prüfen, ob das zuletzt hinzugefügte Item der Kante entspricht
    last_item = data_items[-1]
    # Überprüfen, ob x-Werte des Plots mit edge übereinstimmen
    np.testing.assert_allclose(last_item.xData, edge)
    # y-Werte sollten 0..len(edge)-1 sein
    np.testing.assert_allclose(last_item.yData, np.arange(len(edge)))


def test_plot_with_edge_removes_previous_edge(canvas):
    image = np.random.rand(50, 50)
    edge1 = np.full(50, 20)
    edge2 = np.full(50, 30)

    # Erstes Plot mit Kante 1
    canvas.plot_with_edge(image, edge1)
    data_items_before = canvas.plot_widget.listDataItems()

    # Zweites Plot mit Kante 2 (soll alte Kante entfernen)
    canvas.plot_with_edge(image, edge2)
    data_items_after = canvas.plot_widget.listDataItems()

    # Es sollte nicht mehr Items als vorher geben (alte Kante wurde entfernt)
    assert len(data_items_after) <= len(data_items_before)

    # Die letzte Kante hat x-Werte wie edge2
    last_item = data_items_after[-1]
    np.testing.assert_allclose(last_item.xData, edge2)

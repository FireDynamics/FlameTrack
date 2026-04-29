# tests/conftest.py
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt
import pytest
from PySide6.QtWidgets import QApplication

from flametrack.gui.main_window import MainWindow
from tests.utils.plot_helpers import save_dewarp_comparison_figure


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def mainwindow(app):
    return MainWindow()


@pytest.fixture
def save_comparison_image(request):
    test_name = request.node.name
    output_root = os.getenv("FLAMETRACK_TEST_OUTPUTS", "tests/_outputs")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save(grad_left, grad_right, left_data, right_data):
        file_path = output_dir / f"{test_name}_comparison.png"
        save_dewarp_comparison_figure(
            grad_left, grad_right, left_data, right_data, file_path
        )

    return _save


@pytest.fixture
def save_test_plot(request):
    test_name = request.node.name
    output_root = os.getenv("FLAMETRACK_TEST_OUTPUTS", "tests/_outputs")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, suffix="plot"):
        file_path = output_dir / f"{test_name}_{suffix}.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)

    return _save

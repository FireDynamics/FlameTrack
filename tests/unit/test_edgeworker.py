import os
import tempfile

import h5py
import numpy as np
import pytest

from flametrack.analysis.edge_worker import EdgeDetectionWorker


def dummy_edge_method(y, params=None):
    """A dummy edge detection method returning the max index of the input array."""
    return int(np.argmax(y))


@pytest.fixture
def temp_h5_file():
    """Creates a temporary HDF5 file with a test dataset."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        h5_path = tmp.name

    with h5py.File(h5_path, "w") as f:
        data = np.random.rand(10, 10, 5).astype(np.float32)  # shape: (homography, W, T)
        f.create_dataset("dewarped_data/data", data=data)

    yield h5_path
    os.remove(h5_path)


def test_edge_detection_worker_emits_signals(app, temp_h5_file):
    """Tests that EdgeDetectionWorker runs and emits the 'finished' signal with results."""
    worker = EdgeDetectionWorker(
        h5_path=temp_h5_file,
        dataset_key="dewarped_data/data",
        result_key="test_result_key",
        threshold=100,
        method=dummy_edge_method,
    )

    result_container = {}

    def on_finished(data, key):
        result_container["data"] = data
        result_container["key"] = key

    # Verbinde Signal
    worker.finished.connect(on_finished)

    # Direkt synchron ausführen
    worker.run()

    # Überprüfe Ergebnis
    assert "data" in result_container
    assert isinstance(result_container["data"], np.ndarray)
    assert result_container["data"].shape == (5, 10)  # 5 frames, 1D edge per frame
    assert result_container["key"] == "test_result_key"

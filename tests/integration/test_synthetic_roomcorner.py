from unittest.mock import MagicMock

import cv2
import h5py
import numpy as np
import pytest

from flametrack.analysis.data_types import RceExperiment
from flametrack.analysis.ir_analysis import read_ir_data
from tests.utils.test_helpers import assert_image_similarity


def test_synthetic_roomcorner(mainwindow, tmp_path, save_comparison_image):
    # === 1. Grauverlauf mit Beschriftung erzeugen ===
    height, width = 200, 300
    flat_image = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = 30, 40, 120, 160
    x1_r, x2_r = 180, 270
    w, h = x2 - x1, y2 - y1
    w_r = x2_r - x1_r

    grad_left = np.fromfunction(
        lambda y, x: np.clip(100 + 155 * (x + y) / (w + h), 0, 255), (h, w), dtype=int
    ).astype(np.uint8)
    grad_right = np.fromfunction(
        lambda y, x: np.clip(100 + 155 * ((w_r - x) + y) / (w_r + h), 0, 255),
        (h, w_r),
        dtype=int,
    ).astype(np.uint8)

    cv2.putText(
        grad_left,
        "L",
        (w // 2 - 10, h // 2 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        255,
        2,
    )
    cv2.putText(
        grad_right,
        "R",
        (w_r // 2 - 10, h // 2 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        255,
        2,
    )

    flat_image[y1:y2, x1:x2] = grad_left
    flat_image[y1:y2, x1_r:x2_r] = grad_right

    # === 2. Verzerren zur Raumecke ===
    canvas_size = (400, 400)
    src_left = np.array([[30, 40], [120, 40], [120, 160], [30, 160]], dtype=np.float32)
    src_right = np.array(
        [[180, 40], [270, 40], [270, 160], [180, 160]], dtype=np.float32
    )
    dst_left = np.array(
        [[100, 110], [150, 100], [150, 300], [100, 310]], dtype=np.float32
    )
    dst_right = np.array(
        [[150, 100], [200, 110], [200, 310], [150, 300]], dtype=np.float32
    )

    left_mask = np.zeros_like(flat_image)
    left_mask[y1:y2, x1:x2] = grad_left
    right_mask = np.zeros_like(flat_image)
    right_mask[y1:y2, x1_r:x2_r] = grad_right

    warped_left = cv2.warpPerspective(
        left_mask, cv2.getPerspectiveTransform(src_left, dst_left), canvas_size
    )
    warped_right = cv2.warpPerspective(
        right_mask, cv2.getPerspectiveTransform(src_right, dst_right), canvas_size
    )
    warped = np.clip(warped_left + warped_right, 0, 255).astype(np.uint8)

    # === 3. Warped CSV speichern ===
    csv_path = tmp_path / "exported_data" / "frame_0000.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="latin-1") as f:
        f.write("[Data]\n")
        np.savetxt(f, warped.astype(np.float32), fmt="%.2f", delimiter=";")
    assert csv_path.exists()

    # === 4. Experiment initialisieren & Dummy-Datenquelle einfügen ===
    experiment = RceExperiment(str(tmp_path))
    mainwindow.experiment = experiment
    mainwindow.set_experiment_type("Room Corner")

    class DummyIRData:
        def __init__(self, frame):
            self.frames = [frame]
            self.data_numbers = [0]

        def get_frame(self, idx, rotation):
            return self.frames[idx]

        def get_raw_frame(self, idx):
            return self.frames[idx]

        def get_frame_count(self):
            return len(self.frames)

    dummy_data = DummyIRData(read_ir_data(csv_path))
    experiment.get_data = lambda _: dummy_data

    mainwindow.target_ratio = grad_left.shape[0] / grad_left.shape[1]
    mainwindow.target_pixels_width = grad_left.shape[1]
    mainwindow.target_pixels_height = grad_left.shape[0]

    # === 5. Punkte mocken (P0–P5)
    points = [[100, 110], [150, 100], [200, 110], [200, 310], [150, 300], [100, 310]]
    mock_points = []
    for x, y in points:
        p = MagicMock()
        sp = MagicMock()
        sp.x.return_value = x
        sp.y.return_value = y
        p.scatter_points = [sp]
        mock_points.append(p)
    mainwindow.ui.plot_dewarping = MagicMock()
    mainwindow.ui.plot_dewarping.draggable_points = mock_points

    # === 6. Dewarp ausführen ===
    mainwindow.on_dewarp_clicked()

    # === 7. Ergebnisse prüfen ===
    h5_path = tmp_path / "processed_data" / f"{experiment.exp_name}_results_RCE.h5"
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as f:
        left_data = f["dewarped_data_left"]["data"][:, :, 0]
        right_data = f["dewarped_data_right"]["data"][:, :, 0]
        actual_ratio = f["dewarped_data_left"].attrs["target_ratio"]

    expected_ratio = mainwindow.target_pixels_height / mainwindow.target_pixels_width
    assert actual_ratio == pytest.approx(expected_ratio, abs=0.05)

    assert_image_similarity(
        grad_left, left_data, ssim_thresh=0.94, mae_thresh=5.0, name="Left"
    )
    assert_image_similarity(
        grad_right, right_data, ssim_thresh=0.94, mae_thresh=5.0, name="Right"
    )

    # === Vergleichsbild speichern ===
    save_comparison_image(grad_left, grad_right, left_data, right_data)

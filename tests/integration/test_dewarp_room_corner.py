from pathlib import Path
from unittest.mock import MagicMock

import h5py
import pytest

from flametrack.analysis.DataTypes import RCE_Experiment
from flametrack.analysis.IR_analysis import read_IR_data
from tests.utils.data_helpers import create_room_corner_test_dataset


def test_dewarp_room_corner_with_real_image(mainwindow, tmp_path):
    # 1. Testdaten vorbereiten
    fixture_csv = Path("tests/fixtures/PMMA_DE_6mm_RCE_1m_R1_0001.csv")
    # fixture_csv = Path("tests/fixtures/room_corner_frame_0001.csv")
    base = create_room_corner_test_dataset(tmp_path, source_csv=fixture_csv)

    # 2. Experiment initialisieren
    experiment = RCE_Experiment(str(base))
    mainwindow.experiment = experiment
    mainwindow.set_experiment_type("Room Corner")
    plate_width_mm = 1000  # Beispiel
    plate_height_mm = 1500
    mainwindow.target_pixels_width = int(plate_width_mm / 10)
    mainwindow.target_pixels_height = int(plate_height_mm / 10)

    # 3. Testpunkte für Dewarp (Mock)
    points = [(10, 20), (60, 20), (110, 20), (110, 100), (60, 100), (10, 100)]
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

    # 4. IR-Daten laden & überprüfen (korrekt über read_IR_data)
    csv_path = base / "exported_data" / "frame_0000.csv"

    assert csv_path.exists(), "CSV file was not created"

    print("CSV preview:")
    print(read_IR_data(csv_path)[:3, :3])

    # 5. Dewarp durchführen
    mainwindow.on_dewarp_clicked()

    # 6. Ergebnisse prüfen
    h5_path = base / "processed_data" / f"{experiment.exp_name}_results_RCE.h5"
    assert h5_path.exists(), "HDF5 file was not created"

    with h5py.File(h5_path, "r") as f:
        print(list(f.keys()))
        left_data = f["dewarped_data_left"]["data"][:, :, 0]
        print(left_data.shape)
        right_data = f["dewarped_data_right"]["data"][:]

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # axs[0].imshow(left_data, cmap='gray')
    # axs[0].set_title("Dewarped Left")
    # axs[1].imshow(right_data, cmap='gray')
    # axs[1].set_title("Dewarped Right")
    # for ax in axs: ax.axis("off")
    # plt.tight_layout()
    # plt.show()

    with h5py.File(h5_path, "r") as h5f:
        assert "dewarped_data_left" in h5f
        assert "dewarped_data_right" in h5f
        assert h5f["dewarped_data_left"]["data"].shape[2] >= 1
        assert "transformation_matrix" in h5f["dewarped_data_left"].attrs
        expected_ratio = (
            mainwindow.target_pixels_height / mainwindow.target_pixels_width
        )
        assert h5f["dewarped_data_left"].attrs["target_ratio"] == pytest.approx(
            expected_ratio, abs=0.05
        )

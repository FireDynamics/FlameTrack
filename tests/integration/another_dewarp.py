import numpy as np
import pytest

from flametrack.analysis.data_types import RceExperiment
from flametrack.analysis.ir_analysis import read_ir_data
from flametrack.processing.dewarping import dewarp_room_corner_remap


@pytest.mark.remap
def test_synthetic_roomcorner_remap_2(tmp_path):
    # === 1. Erzeuge synthetisches Bild mit zwei konstanten Platten ===
    height, width = 100, 50
    left = np.full((height, width), 100, dtype=np.float32)
    right = np.full((height, width), 200, dtype=np.float32)
    canvas = np.zeros((200, 200), dtype=np.float32)
    canvas[50:150, 25:75] = left
    canvas[50:150, 125:175] = right

    # === 2. Speichere als CSV im IR-Format ===
    csv_path = tmp_path / "exported_data" / "frame_0000.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="latin-1") as f:
        f.write("[Data]\n")
        np.savetxt(f, canvas, fmt="%.2f", delimiter=";")

    # === 3. Experiment und Dummy-Datenstruktur ===
    class DummyIRData:
        def __init__(self, frame):
            self.frames = [frame]
            self.data_numbers = [0]

        def get_frame(self, idx, rotation_index):
            return self.frames[idx]

        def get_raw_frame(self, idx):
            return self.frames[idx]

        def get_frame_count(self):
            return len(self.frames)

    dummy_data = DummyIRData(read_ir_data(csv_path))
    experiment = RceExperiment(str(tmp_path))
    experiment.get_data = lambda _: dummy_data

    # === 4. Definiere Punkte auf linker & rechter Platte ===
    points = [(25, 50), (75, 50), (125, 50), (175, 150), (125, 150), (25, 150)]

    # === 5. Führe Dewarping durch ===
    h5_path = tmp_path / "processed_data" / f"{experiment.exp_name}_results_RCE.h5"
    generator = dewarp_room_corner_remap(
        experiment=experiment,
        points=points,
        target_ratio=1.0,
        target_pixels_width=width,
        target_pixels_height=height,
        rotation_index=0,
        filename=str(h5_path),
    )
    for _ in generator:
        pass

    # === 6. Vergleiche Inhalte ===
    with experiment.h5_file as h5:
        left_result = h5["dewarped_data_left"]["data"][:, :, 0]
        right_result = h5["dewarped_data_right"]["data"][:, :, 0]

        # Inhalt prüfen
        assert np.allclose(left_result, 100, atol=2)
        assert np.allclose(right_result, 200, atol=2)

    # === 7. Gib Pfad zur Datei aus ===
    print(f"[TEST] HDF5-Datei gespeichert unter: {h5_path.resolve()}")

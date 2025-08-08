import shutil


def create_room_corner_test_dataset(tmp_path, source_csv):
    base = tmp_path / "room_corner_test_exp"
    (base / "exported_data").mkdir(parents=True)

    # Kopiere Testframe in das echte Zielverzeichnis
    shutil.copy(source_csv, base / "exported_data" / "frame_0000.csv")
    return base

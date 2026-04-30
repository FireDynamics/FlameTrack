from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from flametrack.analysis.data_types import ImageData, IrData, RceExperiment, VideoData


# Dummy VideoCapture Mock, der read() korrekt als (ret, frame) liefert
class DummyCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def isOpened(self):
        return self.index < len(self.frames)

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def release(self):
        pass

    def set(self, prop_id, val):
        pass

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0


def test_imagedata_get_frame_count(tmp_path):
    fake_file = tmp_path / "img1.JPG"
    fake_file.write_bytes(b"fake data")

    with patch("cv2.imread", return_value=np.ones((100, 100, 3), dtype=np.uint8)):
        data = ImageData(str(tmp_path))
        assert data.get_frame_count() == 1
        data.get_frame(0, 0)


def test_imagedata_get_frame_raises_index_error(tmp_path):
    data = ImageData(str(tmp_path))
    with pytest.raises(IndexError):
        data.get_frame(0, 0)


def test_videodata_frame_count_mocked():
    with patch("cv2.VideoCapture") as mock_capture:
        mock_cap = DummyCapture([np.zeros((10, 10, 3), dtype=np.uint8)] * 10)
        mock_capture.return_value = mock_cap

        data = VideoData("fakefile.mp4", load_to_memory=False)
        assert data.get_frame_count() == 10


def test_irdata_get_frame_and_raw_frame(tmp_path):
    files = [tmp_path / f"file{i}.csv" for i in range(2)]
    for f in files:
        f.write_text("dummy")

    with (
        patch("glob.glob", return_value=[str(f) for f in files]),
        patch(
            "flametrack.analysis.data_types.read_ir_data",
            return_value=np.ones((10, 10)),
        ),
        patch("numpy.rot90", side_effect=lambda x, k=1: x),
    ):
        ir = IrData(str(tmp_path))
        frame = ir.get_frame(0, 1)
        raw_frame = ir.get_raw_frame(1)
        assert frame.shape == (10, 10)
        assert raw_frame.shape == (10, 10)


def test_irdata_get_frame_count(tmp_path):
    files = [tmp_path / f"file{i}.csv" for i in range(3)]
    for f in files:
        f.write_text("dummy")

    with patch("glob.glob", return_value=[str(f) for f in files]):
        ir = IrData(str(tmp_path))
        count = ir.get_frame_count()
        assert count == 3


def test_rce_experiment_get_data_variants(tmp_path):
    for folder in ["exported_data", "video", "images", "processed_data"]:
        (tmp_path / folder).mkdir()

    (tmp_path / "exported_data" / "frame_0000.csv").write_text("dummy")
    (tmp_path / "images" / "frame_0000.JPG").write_bytes(b"dummy")
    video_file = tmp_path / "video" / "vid1.mp4"
    video_file.write_bytes(b"dummy")

    rce = RceExperiment(str(tmp_path))

    with (
        patch("flametrack.analysis.data_types.IrData") as mock_ir,
        patch("flametrack.analysis.data_types.VideoData") as mock_vid,
        patch("flametrack.analysis.data_types.ImageData") as mock_img,
    ):
        mock_ir.return_value = IrData(str(tmp_path / "exported_data"))
        mock_vid.return_value = VideoData(str(video_file))
        mock_img.return_value = ImageData(str(tmp_path / "images"))

        ir_data = rce.get_data("ir")
        vid_data = rce.get_data("video")
        img_data = rce.get_data("picture")
        proc_data = rce.get_data("processed")

        assert isinstance(ir_data, IrData)
        assert isinstance(vid_data, VideoData)
        assert isinstance(img_data, ImageData)
        assert isinstance(proc_data, IrData)

    with pytest.raises(ValueError):
        rce.get_data("unknown")


def test_rce_experiment_h5_file_property(tmp_path):
    proc_dir = tmp_path / "processed_data"
    proc_dir.mkdir()
    test_file = proc_dir / f"{tmp_path.name}_results_RCE.h5"
    test_file.write_bytes(b"dummy content")

    rce = RceExperiment(str(tmp_path))

    with patch("h5py.File") as mock_file:
        mock_handle = MagicMock()
        mock_file.return_value = mock_handle

        file_handle = rce.h5_file
        assert file_handle == mock_handle

        rce.h5_file = mock_handle
        assert rce._h5_file == mock_handle


def test_rce_experiment_get_data_file_not_found(tmp_path):
    rce = RceExperiment(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        rce.get_data("ir")
    with pytest.raises(FileNotFoundError):
        rce.get_data("video")
    with pytest.raises(FileNotFoundError):
        rce.get_data("picture")
    with pytest.raises(FileNotFoundError):
        rce.get_data("processed")


# ---------------------------------------------------------------------------
# VideoData – load_to_memory=True (lines 74–82)
# ---------------------------------------------------------------------------


def test_videodata_load_to_memory():
    """VideoData loads all frames into self.data when load_to_memory=True."""
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    mock_cap = DummyCapture(frames)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        data = VideoData("fake.mp4", load_to_memory=True)

    assert len(data.data) == 5
    assert data.get_frame_count() == 5


def test_videodata_get_frame_from_memory():
    """get_frame reads from self.data when loaded to memory."""
    frame_arr = np.ones((10, 10, 3), dtype=np.uint8) * 128
    mock_cap = DummyCapture([frame_arr])

    with patch("cv2.VideoCapture", return_value=mock_cap):
        data = VideoData("fake.mp4", load_to_memory=True)

    with patch("cv2.cvtColor", return_value=np.ones((10, 10), dtype=np.uint8)):
        f = data.get_frame(0, 0)
    assert f is not None


def test_videodata_get_frame_on_demand_read_failure():
    """get_frame raises IndexError when cap.read() fails."""

    # cap that returns False on read
    class FailCap:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

        def set(self, *a):
            pass

        def get(self, prop):
            return 3

    with patch("cv2.VideoCapture") as mock_cv:
        mock_cv.return_value = FailCap()
        data = VideoData("fake.mp4", load_to_memory=False)

    with patch("cv2.VideoCapture", return_value=FailCap()):
        with pytest.raises(IndexError):
            data.get_frame(0, 0)


# ---------------------------------------------------------------------------
# ImageData – custom extension (lines 125–126)
# ---------------------------------------------------------------------------


def test_imagedata_custom_extension(tmp_path):
    """ImageData with explicit image_extension builds correct glob pattern."""
    tif_file = tmp_path / "frame_001.tif"
    tif_file.write_bytes(b"fake tif content")

    with patch("cv2.imread", return_value=np.ones((100, 100, 3), dtype=np.uint8)):
        data = ImageData(str(tmp_path), image_extension="tif")

    assert data.get_frame_count() == 1


def test_imagedata_get_org_frame(tmp_path):
    """get_org_frame returns BGR array from disk."""
    fake_file = tmp_path / "img1.JPG"
    fake_file.write_bytes(b"fake")

    bgr = np.ones((50, 50, 3), dtype=np.uint8)
    with patch("cv2.imread", return_value=bgr):
        data = ImageData(str(tmp_path))
        result = data.get_org_frame(0)

    assert result.shape == (50, 50, 3)


def test_imagedata_get_org_frame_missing_raises(tmp_path):
    """get_org_frame raises FileNotFoundError when cv2.imread returns None."""
    fake_file = tmp_path / "img1.JPG"
    fake_file.write_bytes(b"fake")

    with patch("cv2.imread", return_value=np.ones((10, 10, 3), dtype=np.uint8)):
        data = ImageData(str(tmp_path))

    with patch("cv2.imread", return_value=None):
        with pytest.raises(FileNotFoundError):
            data.get_org_frame(0)


def test_imagedata_get_frame_missing_raises(tmp_path):
    """get_frame raises FileNotFoundError when cv2.imread returns None."""
    fake_file = tmp_path / "img1.JPG"
    fake_file.write_bytes(b"fake")

    with patch("cv2.imread", return_value=np.ones((10, 10, 3), dtype=np.uint8)):
        data = ImageData(str(tmp_path))

    with patch("cv2.imread", return_value=None):
        with pytest.raises(FileNotFoundError):
            data.get_frame(0, 0)


# ---------------------------------------------------------------------------
# DataClass.get_frame_size / get_raw_frame (lines 48–57)
# ---------------------------------------------------------------------------


def test_dataclass_get_frame_size(tmp_path):
    """get_frame_size returns (height, width) tuple."""
    fake_file = tmp_path / "img1.JPG"
    fake_file.write_bytes(b"fake")

    with patch("cv2.imread", return_value=np.ones((30, 40, 3), dtype=np.uint8)):
        data = ImageData(str(tmp_path))

    with patch("cv2.imread", return_value=np.ones((30, 40, 3), dtype=np.uint8)):
        with patch("cv2.cvtColor", return_value=np.ones((30, 40), dtype=np.uint8)):
            h, w = data.get_frame_size()
    assert h == 30
    assert w == 40


def test_dataclass_get_raw_frame_delegates_to_get_frame(tmp_path):
    """Default get_raw_frame calls get_frame(..., rotation_index=0)."""
    fake_file = tmp_path / "img1.JPG"
    fake_file.write_bytes(b"fake")

    gray = np.ones((20, 20), dtype=np.uint8)
    with patch("cv2.imread", return_value=np.ones((20, 20, 3), dtype=np.uint8)):
        data = ImageData(str(tmp_path))

    with patch("cv2.imread", return_value=np.ones((20, 20, 3), dtype=np.uint8)):
        with patch("cv2.cvtColor", return_value=gray):
            raw = data.get_raw_frame(0)
    assert raw.shape == (20, 20)

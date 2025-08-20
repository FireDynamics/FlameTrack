# pylint: disable=unused-private-member

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, List, Optional, Tuple, TypeAlias, cast

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

from .ir_analysis import read_ir_data

# Typalias: Frames können ganzzahlig (u. a. uint8) oder float sein
FrameInt: TypeAlias = NDArray[np.integer]
FrameFloat: TypeAlias = NDArray[np.floating]
Frame: TypeAlias = FrameInt | FrameFloat


class DataClass(ABC):
    """
    Abstract base class for different types of experiment data sources.
    Defines interface for frame access and metadata.
    """

    def __init__(self) -> None:
        # List of valid frame indices or identifiers
        self.data_numbers: list[int] = []

    @abstractmethod
    def get_frame(self, framenr: int, rotation_index: int) -> NDArray[Any]:
        """
        Return a single frame by index, rotated as specified.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_frame_count(self) -> int:
        """Return the total number of frames available."""

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Returns shape (height, width) of a single frame.
        """
        h, w = self.get_frame(0, 0).shape[:2]
        return int(h), int(w)


class VideoData(DataClass):
    """
    Handles video files, supports lazy loading or loading full video into memory.
    """

    def __init__(self, videofile: str, load_to_memory: bool = False) -> None:
        super().__init__()
        self.data: list[NDArray[np.uint8]] = []  # In-memory frames if loaded (BGR)
        self.videofile = videofile

        cap = cv2.VideoCapture(videofile)

        if load_to_memory:
            # Load all frames into memory for fast repeated access
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # frame: HxWx3 uint8 (BGR)
                frame_u8 = np.asarray(frame, dtype=np.uint8)
                self.data.append(frame_u8)
            cap.release()
            self.data_numbers = list(range(len(self.data)))
        else:
            # Just store indices, load frames on demand
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.data_numbers = list(range(count))

    def get_frame(self, framenr: int, rotation_index: int) -> NDArray[np.uint8]:
        if self.data:
            frame_bgr: NDArray[np.uint8] = self.data[framenr]
        else:
            cap = cv2.VideoCapture(self.videofile)
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenr)
            ret, frame_bgr_any = cap.read()
            cap.release()
            if not ret or frame_bgr_any is None:
                raise IndexError(f"Frame {framenr} could not be read.")
            frame_bgr = cast(NDArray[np.uint8], frame_bgr_any)

        # OpenCV gibt laut Stubs Mat|ndarray zurück → zu uint8 casten
        frame_gray = cast(
            NDArray[np.uint8], cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        )

        # np.rot90 behält dtype, aber mypy kennt das nicht → cast
        rotated = cast(NDArray[np.uint8], np.rot90(frame_gray, rotation_index))
        return rotated

    def get_frame_count(self) -> int:
        return len(self.data_numbers)


class ImageData(DataClass):
    """
    Lädt sequenzielle Bilddateien (z. B. JPG/PNG) aus einem Ordner.
    Liefert uint8-Graustufenframes.
    """

    def __init__(self, image_folder: str, image_extension: str | None = None) -> None:
        super().__init__()
        patterns: list[str]
        if image_extension:
            # Case-insensitive Pattern aus dem Extension-String bauen
            ext_pat = "".join(f"[{c.lower()}{c.upper()}]" for c in image_extension)
            patterns = [os.path.join(image_folder, f"*.{ext_pat}")]
        else:
            # Gängige Formate, case-insensitiv
            patterns = [
                os.path.join(image_folder, "*.[Jj][Pp][Gg]"),
                os.path.join(image_folder, "*.[Jj][Pp][Ee][Gg]"),
                os.path.join(image_folder, "*.[Pp][Nn][Gg]"),
                os.path.join(image_folder, "*.[Tt][Ii][Ff]"),
                os.path.join(image_folder, "*.[Tt][Ii][Ff][Ff]"),
            ]

        files: list[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat))

        self.files = sorted(files)
        self.data_numbers = list(range(len(self.files)))

    def get_frame(self, framenr: int, rotation_index: int) -> NDArray[np.uint8]:
        path = self.files[framenr]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image file not found: {path}")
        gray = cast(NDArray[np.uint8], cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        rotated = np.rot90(gray, rotation_index)
        return rotated

    def get_org_frame(self, framenr: int) -> NDArray[np.uint8]:
        path = self.files[framenr]
        bgr = cast(NDArray[np.uint8], cv2.imread(path, cv2.IMREAD_COLOR))
        if bgr is None:
            raise FileNotFoundError(f"Image file not found: {path}")
        return bgr

    def get_frame_count(self) -> int:
        return len(self.files)


class IrData(DataClass):
    """
    Handles infrared CSV data files.
    """

    def __init__(self, data_folder: str) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.files: list[str] = glob.glob(f"{self.data_folder}/*.csv")
        self.files.sort()
        self.data_numbers = list(range(len(self.files)))

    def __sort_files(self) -> None:
        """Placeholder for file sorting logic if needed."""
        # keep for future logic
        return None

    def get_frame(self, framenr: int, rotation_index: int) -> NDArray[np.float64]:
        frame = read_ir_data(self.files[framenr])  # float
        return np.rot90(frame, k=rotation_index)

    def get_raw_frame(self, framenr: int) -> FrameFloat:
        """Return unrotated raw IR frame."""
        path = self.files[framenr]
        frame = read_ir_data(path)
        return frame

    def get_frame_count(self) -> int:
        return len(self.data_numbers)


class RceExperiment:
    """
    Manages experiment data and access to various data sources.
    """

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self.exp_name = os.path.basename(folder_path)
        self.ir_data: Optional[IrData] = None
        self.video_data: Optional[VideoData] = None
        self.picture_data: Optional[ImageData] = None
        self._h5_file: Optional[h5py.File] = None

    @property
    def h5_file(self) -> h5py.File:
        """Lazy-load or reload HDF5 file for experiment results."""
        with suppress(OSError, AttributeError, ValueError):
            if self._h5_file is not None:
                self._h5_file.close()

        self._h5_file = h5py.File(
            os.path.join(
                self.folder_path, "processed_data", self.exp_name + "_results_RCE.h5"
            ),
            "a",
        )
        return self._h5_file

    @h5_file.setter
    def h5_file(self, value: h5py.File) -> None:
        """Setter for HDF5 file handle."""
        self._h5_file = value

    def get_data(self, data_type: str) -> DataClass:
        """
        Retrieve data handler for specified data type.
        """
        dt = data_type.lower()
        if dt == "ir":
            return self.get_ir_data()
        if dt == "video":
            return self._get_video_data()
        if dt == "picture":
            return self._get_picture_data()
        if dt == "processed":
            return self._get_processed_data()
        raise ValueError(f"Unknown data type: {data_type}")

    def get_ir_data(self) -> IrData:
        """Lazily load IR data from exported_data folder."""
        exported_dir = os.path.join(self.folder_path, "exported_data")
        if not os.path.exists(exported_dir):
            raise FileNotFoundError("No exported data found")
        if self.ir_data is None:
            self.ir_data = IrData(exported_dir)
        return self.ir_data

    def _get_video_data(self) -> VideoData:
        """Lazily load video data from video folder."""
        video_dir = os.path.join(self.folder_path, "video")
        if not os.path.exists(video_dir):
            raise FileNotFoundError("No video data found")
        file_list = glob.glob(os.path.join(video_dir, "*.mp4"))
        if not file_list:
            raise FileNotFoundError("No mp4 video files found in video directory")
        return VideoData(file_list[0])

    def _get_picture_data(self) -> ImageData:
        """Lazily load image data from images folder."""
        image_dir = os.path.join(self.folder_path, "images")
        if not os.path.exists(image_dir):
            raise FileNotFoundError("No image data found")
        return ImageData(image_dir)

    def _get_processed_data(self) -> IrData:
        """Lazily load processed IR data from processed_data folder."""
        processed_dir = os.path.join(self.folder_path, "processed_data")
        if not os.path.exists(processed_dir):
            raise FileNotFoundError("No processed data found")
        return IrData(processed_dir)

import glob
import os
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np

from .IR_analysis import read_IR_data


class DataClass:
    """
    Abstract base class for different types of experiment data sources.
    Defines interface for frame access and metadata.
    """

    def __init__(self):
        # List of valid frame indices or identifiers
        self.data_numbers: list[int] = []

    def get_frame(self, framenr: int, rotation_index: int) -> np.ndarray:
        """
        Return a single frame by index, rotated as specified.
        Must be implemented by subclasses.

        Args:
            framenr: Index of the frame to retrieve.
            rotation_index: Number of 90 degree rotations (counterclockwise).

        Returns:
            np.ndarray: Frame image array.
        """
        pass

    def get_frame_count(self) -> int:
        """
        Return the total number of frames available.
        Must be implemented by subclasses.

        Returns:
            int: Number of frames.
        """
        pass

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Returns shape (height, width) of a single frame.

        Returns:
            Tuple[int, int]: Frame size as (height, width).
        """
        return self.get_frame(0, 0).shape


class VideoData(DataClass):
    """
    Handles video files, supports lazy loading or loading full video into memory.
    """

    def __init__(self, videofile: str, load_to_memory: bool = False):
        super().__init__()
        self.data: list[np.ndarray] = []  # In-memory frames if loaded
        self.videofile = videofile

        # Open video capture
        cap = cv2.VideoCapture(videofile)

        if load_to_memory:
            # Load all frames into memory for fast repeated access
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.data.append(frame)
            self.data_numbers = list(range(self.get_frame_count()))
        else:
            # Just store indices, load frames on demand
            self.data_numbers = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    def get_frame(self, framenr: int, rotation_index: int) -> np.ndarray:
        """
        Retrieve a video frame, converting to grayscale and applying rotation.

        Args:
            framenr: Frame index.
            rotation_index: Number of 90 degree CCW rotations.

        Returns:
            np.ndarray: Rotated grayscale frame.
        """
        if len(self.data) > 0:
            # In-memory access
            frame = self.data[framenr]
        else:
            # Load frame from file on demand
            cap = cv2.VideoCapture(self.videofile)
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenr)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise IndexError(f"Frame {framenr} could not be read.")
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Rotate counterclockwise
        return np.rot90(frame, rotation_index)

    def get_frame_count(self) -> int:
        """
        Get number of frames in video.

        Returns:
            int: Frame count.
        """
        return len(self.data_numbers)


class ImageData(DataClass):
    """
    Handles sequential image files (e.g. JPG) from a folder.
    """

    def __init__(self, image_folder: str, image_extension: str = "JPG"):
        super().__init__()
        # Get all image files matching extension, sorted by modification time
        self.files: list[str] = glob.glob(f"{image_folder}/*.{image_extension}")
        self.files.sort(key=lambda x: os.path.getmtime(x))
        self.data_numbers = list(range(len(self.files)))

    def get_frame(self, framenr: int, rotation_index: int) -> np.ndarray:
        """
        Read and return the resized frame for the given index and rotation.
        Only the green channel is used.

        Args:
            framenr: Frame index.
            rotation_index: Rotation in 90 deg steps CCW (ignored currently).

        Returns:
            np.ndarray: Resized frame.
        """
        frame = cv2.imread(self.files[framenr])
        if frame is None:
            raise FileNotFoundError(f"Image file not found: {self.files[framenr]}")
        # Use green channel (index 1)
        frame = frame[:, :, 1]
        # Resize to fixed 1500x1000 (hardcoded)
        return cv2.resize(frame, (1500, 1000))

    def get_org_frame(self, framenr: int) -> np.ndarray:
        """
        Read original frame without rotation or modification.

        Args:
            framenr: Frame index.

        Returns:
            np.ndarray: Resized original frame.
        """
        frame = cv2.imread(self.files[framenr])
        if frame is None:
            raise FileNotFoundError(f"Image file not found: {self.files[framenr]}")
        return cv2.resize(frame, (1500, 1000))

    def get_frame_count(self) -> int:
        """
        Number of image frames.

        Returns:
            int: Number of images.
        """
        return len(self.files)


class IrData(DataClass):
    """
    Handles infrared CSV data files.
    """

    def __init__(self, data_folder: str):
        super().__init__()
        self.data_folder = data_folder
        # Collect all CSV files in folder
        self.files: list[str] = glob.glob(f"{self.data_folder}/*.csv")
        self.files.sort()
        self.data_numbers = list(range(len(self.files)))
        # TODO: Add check for missing files in sequence

    def __sort_files(self):
        """
        Placeholder for file sorting logic if needed.
        """
        pass

    def get_frame(self, framenr: int, rotation_index: int) -> np.ndarray:
        """
        Load and return rotated IR data frame.

        Args:
            framenr: Frame index.
            rotation_index: Number of 90 degree rotations CCW.

        Returns:
            np.ndarray: Rotated IR data frame.
        """
        file = self.files[framenr]
        frame = read_IR_data(file)
        return np.rot90(frame, k=rotation_index)

    def get_raw_frame(self, framenr: int) -> np.ndarray:
        """
        Return unrotated raw IR frame.

        Args:
            framenr: Frame index.

        Returns:
            np.ndarray: Raw IR data frame.
        """
        file = self.files[framenr]
        frame = read_IR_data(file)
        return frame

    def get_frame_count(self) -> int:
        """
        Number of IR frames available.

        Returns:
            int: Frame count.
        """
        return len(self.data_numbers)


class RCE_Experiment:
    """
    Manages experiment data and access to various data sources.

    Attributes:
        folder_path: Path to experiment folder.
        exp_name: Experiment name derived from folder.
        IR_data: Cached IR data handler.
        Video_data: Cached video data handler.
        Picture_data: Cached image data handler.
        _h5_file: Cached h5py file handle.
    """

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.exp_name = os.path.basename(folder_path)
        self.IR_data: Optional[IrData] = None
        self.Video_data: Optional[VideoData] = None
        self.Picture_data: Optional[ImageData] = None
        self._h5_file: Optional[h5py.File] = None

    @property
    def h5_file(self) -> h5py.File:
        """
        Lazy-load or reload HDF5 file for experiment results.

        Returns:
            h5py.File: Open file handle.
        """
        try:
            if self._h5_file is not None:
                self._h5_file.close()
        except Exception:
            pass
        self._h5_file = h5py.File(
            os.path.join(
                self.folder_path, "processed_data", self.exp_name + "_results_RCE.h5"
            ),
            "a",
        )
        return self._h5_file

    @h5_file.setter
    def h5_file(self, value: h5py.File) -> None:
        """
        Setter for HDF5 file handle.

        Args:
            value: New file handle.
        """
        self._h5_file = value

    def get_data(self, data_type: str) -> DataClass:
        """
        Retrieve data handler for specified data type.

        Args:
            data_type: One of 'ir', 'video', 'picture', 'processed'.

        Returns:
            DataClass: Corresponding data handler instance.

        Raises:
            ValueError: If unknown data type is requested.
        """
        data_type = data_type.lower()
        if data_type == "ir":
            return self._get_IR_data()
        if data_type == "video":
            return self._get_video_data()
        if data_type == "picture":
            return self._get_picture_data()
        if data_type == "processed":
            return self._get_processed_data()
        raise ValueError(f"Unknown data type: {data_type}")

    def _get_IR_data(self) -> IrData:
        """
        Lazily load IR data from exported_data folder.

        Returns:
            IrData: IR data handler.

        Raises:
            FileNotFoundError: If expected folder is missing.
        """
        exported_dir = os.path.join(self.folder_path, "exported_data")
        if not os.path.exists(exported_dir):
            raise FileNotFoundError("No exported data found")
        if self.IR_data is None:
            self.IR_data = IrData(exported_dir)
        return self.IR_data

    def _get_video_data(self) -> VideoData:
        """
        Lazily load video data from video folder.

        Returns:
            VideoData: Video data handler.

        Raises:
            FileNotFoundError: If expected folder or files missing.
        """
        video_dir = os.path.join(self.folder_path, "video")
        if not os.path.exists(video_dir):
            raise FileNotFoundError("No video data found")
        file_list = glob.glob(os.path.join(video_dir, "*.mp4"))
        if not file_list:
            raise FileNotFoundError("No mp4 video files found in video directory")
        return VideoData(file_list[0])

    def _get_picture_data(self) -> ImageData:
        """
        Lazily load image data from images folder.

        Returns:
            ImageData: Image data handler.

        Raises:
            FileNotFoundError: If folder missing.
        """
        image_dir = os.path.join(self.folder_path, "images")
        if not os.path.exists(image_dir):
            raise FileNotFoundError("No image data found")
        return ImageData(image_dir)

    def _get_processed_data(self) -> IrData:
        """
        Lazily load processed IR data from processed_data folder.

        Returns:
            IrData: Processed IR data handler.

        Raises:
            FileNotFoundError: If folder missing.
        """
        processed_dir = os.path.join(self.folder_path, "processed_data")
        if not os.path.exists(processed_dir):
            raise FileNotFoundError("No processed data found")
        return IrData(processed_dir)

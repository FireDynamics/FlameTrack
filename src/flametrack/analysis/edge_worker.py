import logging

import h5py
import numpy as np
from PySide6.QtCore import QObject, Signal

from flametrack.analysis.flamespread import EdgeFn, calculate_edge_data


class EdgeDetectionWorker(QObject):
    """
    Worker for performing edge detection on HDF5 datasets in a separate thread.

    Signals:
        progress (int): Emits progress percentage during processing.
        finished (np.ndarray, str): Emits the result array and result key after completion.
    """

    progress = Signal(int)
    finished = Signal(np.ndarray, str)

    def __init__(
        self,
        h5_path: str,
        dataset_key: str,
        result_key: str,
        threshold: int,
        method: EdgeFn,
        flame_direction: str | None = None,
        use_otsu_masking: bool = True,
    ):
        """
        Initialize the edge detection worker.

        Args:
            h5_path: Path to the HDF5 file.
            dataset_key: HDF5 key of the dataset to process (e.g. "dewarped_data/data").
            result_key: Key under which results should be stored or identified.
            threshold: Threshold for edge detection.
            method: Callable method for edge calculation. Must accept (line, params).
            flame_direction: Optional flame direction metadata ("left_to_right" or "right_to_left").
            use_otsu_masking: If True (default), Otsu thresholding is used to narrow
                the per-row search window before applying the edge method.  Set False
                to scan the full row with the raw intensity threshold.
        """
        super().__init__()
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.result_key = result_key
        self.threshold = threshold
        self.method: EdgeFn = method
        self.flame_direction = flame_direction
        self.use_otsu_masking = use_otsu_masking

    def run(self) -> None:
        """
        Execute the edge detection process frame by frame.

        Emits:
            - progress: Current progress percentage.
            - finished: Final result as NumPy array and associated result key.
        """
        logging.info("[EDGE WORKER] Starting edge detection on %s", self.dataset_key)

        result = []

        with h5py.File(self.h5_path, "r") as f:
            data = f[self.dataset_key]
            total_frames = data.shape[-1]
            method = self.method

            for i in range(total_frames):
                frame = data[:, :, i]

                # Perform edge detection using provided method
                edge = calculate_edge_data(
                    np.expand_dims(frame, axis=-1),
                    method,
                    use_otsu_masking=self.use_otsu_masking,
                )[0]

                result.append(edge)
                self.progress.emit(i + 1)

        result_array = np.stack(result, axis=0)

        logging.info("[EDGE WORKER] Finished processing %d frames.", total_frames)
        self.finished.emit(result_array, self.result_key)

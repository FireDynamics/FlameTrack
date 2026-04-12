# pylint: disable=too-many-arguments,too-many-positional-arguments

import logging
from typing import Optional

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
        flame_direction: Optional[str] = None,
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
        """
        super().__init__()
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.result_key = result_key
        self.threshold = threshold
        self.method: EdgeFn = method
        self.flame_direction = flame_direction

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
            print(
                f"[WORKER {self.dataset_key}] total_frames={total_frames}", flush=True
            )
            method = self.method

            for i in range(total_frames):
                frame = data[:, :, i]

                # Perform edge detection using provided method
                edge = calculate_edge_data(
                    np.expand_dims(frame, axis=-1),
                    method,
                )

                result.append(edge[0])  # only one frame processed
                self.progress.emit(i + 1)

        print(
            f"[WORKER {self.dataset_key}] loop done, stacking {len(result)} results",
            flush=True,
        )
        result_array = np.stack(result, axis=0)

        print(f"[WORKER {self.dataset_key}] emitting finished", flush=True)
        logging.info("[EDGE WORKER] Finished processing %d frames.", total_frames)
        self.finished.emit(result_array, self.result_key)
        print(f"[WORKER {self.dataset_key}] finished emitted", flush=True)

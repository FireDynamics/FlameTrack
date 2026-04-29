from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def compute_remap_from_homography(
    homography: NDArray[np.float32] | NDArray[np.float64],
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute pixelwise remap grids from a homography.

    Args:
        homography (numpy.ndarray): 3×3 homography mapping output → input coordinates.
        width (int): Target image width in pixels.
        height (int): Target image height in pixels.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Two arrays ``(src_x, src_y)`` with shape
        ``(height, width)``, dtype ``float32`` suitable for ``cv2.remap``.
    """
    # Pixelzentren (x+0.5, y+0.5)
    map_x_f = np.arange(width, dtype=np.float32) + 0.5
    map_y_f = np.arange(height, dtype=np.float32) + 0.5
    map_x, map_y = np.meshgrid(map_x_f, map_y_f, indexing="xy")

    ones = np.ones_like(map_x, dtype=np.float32)
    target_coords = np.stack([map_x, map_y, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    homography_f: NDArray[np.float32] = np.asarray(homography, dtype=np.float32)
    source_coords = homography_f @ target_coords
    source_coords /= source_coords[2, :]  # normalize

    src_x = source_coords[0, :].reshape((height, width)).astype(np.float32, copy=False)
    src_y = source_coords[1, :].reshape((height, width)).astype(np.float32, copy=False)

    return src_x, src_y


def read_ir_data(filename: str) -> NDArray[np.float64]:
    """
    Read raw IR data from a CSV-like ASCII export.

    The file is scanned until a line ``[Data]`` is found; subsequent lines are
    parsed using ``;`` as delimiter and a comma-to-dot decimal replacement.

    Args:
        filename (str): Path to the IR data file.

    Returns:
        numpy.ndarray: 2D array of IR values (dtype ``float64``).

    Raises:
        ValueError: If no ``[Data]`` section is found in the file.
    """
    with open(filename, encoding="latin-1") as f:
        line = f.readline()
        while line:
            if line.startswith("[Data]"):
                arr = np.genfromtxt(
                    (line.replace(",", ".")[:-2] for line in f.readlines()),
                    delimiter=";",
                )
                return np.asarray(arr, dtype=np.float64)
            line = f.readline()

    raise ValueError("No data found in file, check file format!")


def get_dewarp_parameters(
    corners: NDArray[np.float32] | Sequence[tuple[float, float]],
    target_pixels_width: int | None = None,
    target_pixels_height: int | None = None,
    target_ratio: float | None = None,
    *,
    plate_width_m: float | None = None,
    plate_height_m: float | None = None,
    pixels_per_millimeter: int = 1,
) -> dict[str, Any]:
    """
    Calculate homography and target geometry for dewarping.

    You can either pass physical plate dimensions (``plate_width_m``,
    ``plate_height_m``) plus a pixel density, or infer target geometry
    from the selected corners and a desired aspect ratio.

    Args:
        corners (numpy.ndarray | Sequence[tuple[float, float]]): Four corner points
            in pixel coordinates, ordered clockwise starting at top-left.
        target_pixels_width (int, optional): Target width in pixels. If omitted,
            it will be derived from ``target_ratio`` and the measured corner distances.
        target_pixels_height (int, optional): Target height in pixels. If omitted,
            it will be derived from ``target_ratio`` and the measured corner distances.
        target_ratio (float, optional): Desired aspect ratio ``height / width``.
            Required if target size is not specified and no physical plate size is provided.
        plate_width_m (float, optional): Physical plate width in meters. Used with
            ``pixels_per_millimeter`` to derive target size if provided with ``plate_height_m``.
        plate_height_m (float, optional): Physical plate height in meters. Used with
            ``pixels_per_millimeter`` to derive target size if provided with ``plate_width_m``.
        pixels_per_millimeter (int, optional): Pixel density (px/mm) used when physical
            dimensions are given. Default is 1.

    Returns:
        dict[str, Any]: Dictionary with:
            - ``transformation_matrix`` (numpy.ndarray): 3×3 homography (float32).
            - ``target_pixels_width`` (int): Target width in pixels.
            - ``target_pixels_height`` (int): Target height in pixels.
            - ``target_ratio`` (float): ``height / width`` of the target.

    Raises:
        ValueError: If neither physical dimensions nor a target ratio are provided.

    Notes:
        Current conversion multiplies meter values by ``pixels_per_millimeter``.
        For strict unit consistency, consider using millimeters or ``pixels_per_meter``.
    """
    buffer = 1.1
    source_corners: NDArray[np.float32] = np.asarray(corners, dtype=np.float32)

    # Falls echte Plattenmaße gegeben sind, direkt daraus Pixel ableiten
    if plate_width_m is not None and plate_height_m is not None:
        target_pixels_width = int(plate_width_m * pixels_per_millimeter)
        target_pixels_height = int(plate_height_m * pixels_per_millimeter)

    # Sonst versucht: aus Ecken + Ratio ab zuleiten
    if target_pixels_width is None or target_pixels_height is None:
        if target_ratio is None:
            raise ValueError("Either plate dimensions or target ratio must be provided")

        # grobe Abschätzung Breite/Höhe in Pixeln aus den Ecken
        max_width = max(
            float(source_corners[1][0] - source_corners[0][0]),
            float(source_corners[2][0] - source_corners[3][0]),
        )
        max_height = max(
            float(source_corners[2][1] - source_corners[1][1]),
            float(source_corners[3][1] - source_corners[0][1]),
        )
        target_pixels_height = int(
            max(max_height, max_width / float(target_ratio)) * buffer
        )
        target_pixels_width = int(target_pixels_height * float(target_ratio))

    tpw = int(target_pixels_width)  # mypy-safe ints
    tph = int(target_pixels_height)

    target_corners = np.array(
        [
            [0.0, 0.0],
            [float(tpw), 0.0],
            [float(tpw), float(tph)],
            [0.0, float(tph)],
        ],
        dtype=np.float32,
    )

    transformation_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)

    return {
        "transformation_matrix": np.asarray(transformation_matrix, dtype=np.float32),
        "target_pixels_width": tpw,
        "target_pixels_height": tph,
        # Beibehaltung deines bisherigen Verhältnisses (height/width):
        "target_ratio": float(tph) / float(tpw),
    }

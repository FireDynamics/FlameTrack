from __future__ import annotations

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

from flametrack.analysis.region import Region


def correct_temperature(
    apparent_temp: NDArray[np.floating],
    emissivity: float,
    ambient_temp_k: float = 293.15,
) -> NDArray[np.floating]:
    """
    Correct an apparent temperature for a surface's true emissivity, using Stefan-Boltzmann approximation.
    """
    t_app_k = apparent_temp + 273.15
    t_true_k4 = (t_app_k**4 - (1 - emissivity) * ambient_temp_k**4) / emissivity
    return np.power(np.clip(t_true_k4, 0, None), 0.25) - 273.15


def region_mask(
    points: list[tuple[float, float]], shape: tuple[int, int]
) -> NDArray[np.bool_]:
    """
    Bitmapsize a region's polygon/rectangle points into a boolean pixel mask of a given (h, w) shape.
    Used to selet exactly which pixels of a frame a region's emissivity correction applies to.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def apply_region_corrections(
    frame: NDArray[np.floating], regions: list[Region]
) -> NDArray[np.floating]:
    """
    Apply each region's own emissivity correction to a single frame, leaving pixels outside every region unchanged.
    Regions are assumed non-overlapping; if they overlap, the last region in the list wins for shared pixels.
    """
    corrected = frame.copy()
    for region in regions:
        mask = region_mask(region.points, frame.shape)
        if not mask.any():
            continue
        corrected[mask] = correct_temperature(frame[mask], region.emissivity)
    return corrected


def apply_corrections_to_experiment(
    h5_group: h5py.Group, regions: list[Region]
) -> None:
    """
    Apply region-based emissivity correction across every frame of a dewarped dataset and write the result
    to a sibling 'corrected_data' dataset in the same HDF5 group, replacing any previous version.
    """
    data = h5_group["data"][:]
    corrected = np.empty_like(data)
    for t in range(data.shape[2]):
        corrected[:, :, t] = apply_region_corrections(data[:, :, t], regions)

    if "corrected_data" in h5_group:
        del h5_group[
            "corrected_data"
        ]  ## Maybe can be optimized here instead of just del everything and re-writing everything ?

    h5_group.create_dataset("corrected_data", data=corrected)

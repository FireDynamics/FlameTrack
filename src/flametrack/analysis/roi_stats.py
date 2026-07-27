from __future__ import annotations

import csv

import h5py
import numpy as np

from flametrack.analysis.emissivity_correction import region_mask
from flametrack.analysis.region import Region


def region_stats(frame: np.ndarray, region: Region) -> dict[str, float]:
    """
    …
    """
    mask = region_mask(region.points, frame.shape)
    if not mask.any():
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    values = frame[mask]
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def region_stats_all(
    frame: np.ndarray, regions: list[Region]
) -> dict[str, dict[str, float]]:
    """
    …
    """
    return {region.region_id: region_stats(frame, region) for region in regions}


def extract_time_series(
    h5_group: h5py.Group, regions: list[Region]
) -> dict[str, dict[str, list[float]]]:
    """
    …
    """
    dataset_key = "corrected_data" if "corrected_data" in h5_group else "data"
    data = h5_group[dataset_key]
    series: dict[str, dict[str, list[float]]] = {
        region.region_id: {"min": [], "max": [], "mean": []} for region in regions
    }
    for t in range(data.shape[2]):
        frame = data[:, :, t]
        stats = region_stats_all(frame, regions)
        for region_id, values in stats.items():
            series[region_id]["min"].append(values["min"])
            series[region_id]["max"].append(values["max"])
            series[region_id]["mean"].append(values["mean"])

    return series


def save_time_series(
    h5_group: h5py.Group, series: dict[str, dict[str, list[float]]]
) -> None:
    """
    …
    """
    if "roi_stats" in h5_group:
        del h5_group["roi_stats"]
    stats_grp = h5_group.create_group("roi_stats")
    for region_id, values in series.items():
        region_grp = stats_grp.create_group(region_id)
        region_grp.create_dataset("min", data=np.array(values["min"], dtype=np.float32))
        region_grp.create_dataset("max", data=np.array(values["max"], dtype=np.float32))
        region_grp.create_dataset(
            "mean", data=np.array(values["mean"], dtype=np.float32)
        )


def export_time_series_csv(
    series: dict[str, dict[str, list[float]]], path: str
) -> None:
    """
    …
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "region_id", "min", "max", "mean"])
        for region_id, values in series.items():
            n_frames = len(values["mean"])
            for t in range(n_frames):
                writer.writerow(
                    [
                        t,
                        region_id,
                        values["min"][t],
                        values["max"][t],
                        values["mean"][t],
                    ]
                )

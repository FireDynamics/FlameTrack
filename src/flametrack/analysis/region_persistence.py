from __future__ import annotations

import h5py

from flametrack.analysis.region_manager import RegionManager


def save_regions(h5_group: h5py.Group, region_manager: RegionManager) -> None:
    if "regions" in h5_group:
        del h5_group["regions"]
    h5_group.create_dataset("regions", data=region_manager.to_json())


def load_regions(
    h5_group: h5py.Group, region_manager_cls: type[RegionManager]
) -> RegionManager:
    if "regions" not in h5_group:
        return region_manager_cls()
    raw = h5_group["regions"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return region_manager_cls.from_json(raw)

from __future__ import annotations

import itertools

from .region import Region, RegionShape


class RegionManager:
    def __init__(self) -> None:
        self._regions: dict[str, Region] = {}
        self._id_counter = itertools.count(1)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._id_counter)}"

    def add_rectangle(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        name: str | None = None,
        color: tuple[int, int, int] = (255, 200, 0),
    ) -> Region:

        region_id = self._new_id("rect")
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        region = Region(
            region_id, name or region_id, RegionShape.RECTANGLE, points, color
        )
        self._regions[region_id] = region

        return region

    def add_polygon(
        self,
        points: list[tuple[float, float]],
        name: str | None = None,
        color: tuple[int, int, int] = (255, 200, 0),
    ) -> Region:

        region_id = self._new_id("poly")
        region = Region(
            region_id, name or region_id, RegionShape.POLYGON, list(points), color
        )
        self._regions[region_id] = region

        return region

    def remove(self, region_id: str) -> None:
        self._regions.pop(region_id, None)

    def get(self, region_id: str) -> Region | None:
        return self._regions.get(region_id)

    def all(self) -> list[Region]:
        return list(self._regions.values())

    def clear(self) -> None:
        self._regions.clear()

    def rename(self, region_id: str, name: str) -> None:
        if region_id in self._regions:
            self._regions[region_id].name = name
        else:
            "Region not found !!!!! "

    def set_emissivity(self, region_id: str, value: float):
        if region_id in self._regions:
            self._regions[region_id].emissivity = value
        else:
            "Region not found (emissivity)!!!!"

    def update_points(self, region_id: str, points: list[tuple[float, float]]) -> None:
        if region_id in self._regions:
            self._regions[region_id].points = points

    @classmethod
    def from_list(cls, data: list[dict]) -> RegionManager:
        mgr = cls()
        max_n = 0
        for entry in data:
            region = Region.from_dict(entry)
            mgr._regions[region.region_id] = region
            n = (
                int(region.region_id.split("_")[-1])
                if region.region_id.split("_")[-1].isdigit()
                else 0
            )
            max_n = max(max_n, n)
        mgr._id_counter = itertools.count(max_n + 1)
        return mgr

    def copy(self) -> RegionManager:
        """Independent deep copy — lets ROIEditorDialog discard edits on Cancel."""
        return RegionManager.from_list(self.to_list())

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self._regions.values()]

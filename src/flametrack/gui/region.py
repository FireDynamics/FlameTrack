from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RegionShape(str, Enum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


@dataclass
class Region:
    region_id: str
    name: str
    shape: RegionShape
    points: list[tuple[float, float]]
    color: tuple[int, int, int] = (255, 200, 0)
    emissivity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "name": self.name,
            "shape": self.shape.value,
            "points": [list(p) for p in self.points],
            "color": list(self.color),
            "emissivity": self.emissivity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Region:
        return cls(
            region_id=data["region_id"],
            name=data["name"],
            shape=RegionShape(data["shape"]),
            points=[tuple(p) for p in data["points"]],
            color=tuple(data.get("color", (255, 200, 0))),
            emissivity=float(data.get("emissivity", 1.0)),
        )

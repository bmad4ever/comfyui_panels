from dataclasses import dataclass
from shapely.geometry import Polygon
from typing import Tuple


@dataclass
class BBoxSnap:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    snap_on_bbox: bool = True

    @classmethod
    def from_bounds(cls, bounds: Tuple[float, float, float, float], snap_on_bbox: bool = True) -> "BBoxSnap":
        """Create from (xmin, ymin, xmax, ymax)."""
        return cls(*bounds, snap_on_bbox)

    @classmethod
    def from_polygon(cls, poly: Polygon, snap_on_bbox: bool = True) -> "BBoxSnap":
        """Create from a Shapely polygon's bounds."""
        return cls(*poly.bounds, snap_on_bbox)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return just the numeric bounds as a tuple."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class CurvatureParams:
    curvature: float = 0.0        # bevel radius
    iterations: int = 1           # how many times to apply beveling
    resolution: int = 16          # buffer resolution (segments per quarter circle)

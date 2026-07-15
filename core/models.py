from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class Geotransform(NamedTuple):
    """GDAL affine geotransform (6 floats).

    Maps pixel (col, row) to world (x, y) via:
        x = origin_x + col * pixel_width + row * rot_x
        y = origin_y + col * rot_y   + row * pixel_height

    For north-up rasters (this project): rot_x == rot_y == 0,
    pixel_width > 0, pixel_height < 0.

    Indices:
        0: origin_x    — x coordinate of the top-left pixel center
        1: pixel_width — x resolution (meters/pixel, positive)
        2: rot_x       — x rotation term (0 for north-up)
        3: origin_y    — y coordinate of the top-left pixel center
        4: rot_y       — y rotation term (0 for north-up)
        5: pixel_height — y resolution (meters/pixel, negative for north-up)
    """

    origin_x: float
    pixel_width: float
    rot_x: float
    origin_y: float
    rot_y: float
    pixel_height: float


class Offset(NamedTuple):
    """Pixel offset of an ROI within the full image.

    Added to ROI-local Coords to lift them back to full-image space
    (the ROI's (row0, col0)).

    Fields:
        row: Row offset (first numpy axis).
        col: Column offset (second numpy axis).
    """

    row: int
    col: int


class RasterShape(NamedTuple):
    """Shape of a 2-D raster array (height, width) in pixels.

    Fields:
        height: Number of rows (first numpy axis).
        width: Number of columns (second numpy axis).
    """

    height: int
    width: int


@dataclass(frozen=True)
class Coord:
    """A location in raster/array space (numpy row, col convention).

    A pure coordinate: an address into the skeleton array, not a stored
    value. Frozen so it is hashable and usable as a dict key or set member.

    Attributes:
        row: Zero-based row index (first numpy axis).
        col: Zero-based column index (second numpy axis).
    """

    row: int
    col: int


@dataclass(frozen=True)
class Point:
    """A location in geographic/Cartesian space.

    Produced from a Coord by applying the raster's GDAL geotransform. Kept
    distinct from Coord so the coordinate system is visible in the type.

    Attributes:
        x: Horizontal coordinate (derived from a Coord's column).
        y: Vertical coordinate (derived from a Coord's row).
    """

    x: float
    y: float


@dataclass
class Roi:
    """A rectangular region of interest in raster space.

    Attributes:
        row0: Top row of the region (inclusive).
        row1: Bottom row of the region (exclusive).
        col0: Left column of the region (inclusive).
        col1: Right column of the region (exclusive).
    """

    row0: int
    row1: int
    col0: int
    col1: int


@dataclass
class Polyline:
    """An open chain of pixels in raster space (the no-polygon fallback and
    debug-line output of cv2_pipeline).

    Its points stay in ROI pixel coordinates (Coord); world.py converts them
    to geographic coordinates (Point) at the end of the pipeline.

    Attributes:
        id: Unique integer identifier within one extraction result.
        points: Ordered pixels along the line, first and last inclusive.
    """

    id: int
    points: list[Coord] = field(default_factory=list)

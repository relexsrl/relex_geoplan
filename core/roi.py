import numpy as np

from .models import Coord, Geotransform, Point, RasterShape, Roi
from .world import world_to_coord


def roi_from_world_corners(
    corner_a: Point,
    corner_b: Point,
    geotransform: Geotransform,
    shape: RasterShape,
) -> Roi:
    """Build a pixel Roi covering a world-space rectangle.

    Converts two opposite corners through the inverse geotransform, takes the
    bounding rows/cols, and clamps to the array. Because the planos have no
    rotation, a world-aligned rectangle maps to a pixel-aligned one, so two
    corners fully determine the box.

    Args:
        corner_a: One corner of the selection, in world coordinates.
        corner_b: The opposite corner, in world coordinates.
        geotransform: GDAL GetGeoTransform() 6-tuple.
        shape: The full raster (height, width), used to clamp in-bounds.

    Returns:
        A Roi with row0/col0 inclusive and row1/col1 exclusive, clamped to
        [0, height] x [0, width].
    """
    a = world_to_coord(corner_a, geotransform)
    b = world_to_coord(corner_b, geotransform)
    height, width = shape

    row0 = max(0, min(a.row, b.row))
    row1 = min(height, max(a.row, b.row) + 1)
    col0 = max(0, min(a.col, b.col))
    col1 = min(width, max(a.col, b.col) + 1)
    return Roi(row0=row0, row1=row1, col0=col0, col1=col1)


def roi_from_world_polygon(
    world_points: list[Point],
    geotransform: Geotransform,
    shape: RasterShape,
) -> tuple[Roi, list[Coord]]:
    """Build a pixel Roi and pixel polygon from world-space polygon vertices.

    Converts each world point to a pixel coord, computes the bounding box
    as the Roi, and returns the full-image pixel coords of the polygon.

    Args:
        world_points: Polygon vertices in world coordinates.
        geotransform: GDAL GetGeoTransform() 6-tuple.
        shape: The full raster (height, width), used to clamp in-bounds.

    Returns:
        A tuple (roi, pixel_coords) where pixel_coords are the polygon
        vertices in full-image pixel space.
    """
    pixels = [world_to_coord(p, geotransform) for p in world_points]
    height, width = shape
    rows = [p.row for p in pixels]
    cols = [p.col for p in pixels]
    roi = Roi(
        row0=max(0, min(rows)),
        row1=min(height, max(rows) + 1),
        col0=max(0, min(cols)),
        col1=min(width, max(cols) + 1),
    )
    return roi, pixels


def expand_roi(roi: Roi, shape: RasterShape, margin: int) -> Roi:
    height, width = shape
    return Roi(
        row0=max(0, roi.row0 - margin),
        row1=min(height, roi.row1 + margin),
        col0=max(0, roi.col0 - margin),
        col1=min(width, roi.col1 + margin),
    )



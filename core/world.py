import math

import numpy as np

from .models import Coord, Geotransform, Offset, Point, Polyline


def world_to_coord(point: Point, geotransform: Geotransform) -> Coord:
    """Map a world-space Point to a raster Coord (inverse geotransform).

    Assumes an axis-aligned geotransform with no rotation or skew
    (rot_x == rot_y == 0), which holds for the georeferenced planos in this
    project. The result is floored to the pixel that contains the point.

    Args:
        point: Location in world/CRS coordinates.
        geotransform: GDAL GetGeoTransform() 6-tuple.

    Returns:
        The Coord (row, col) whose pixel contains the point.
    """
    gt = geotransform
    col = (point.x - gt.origin_x) / gt.pixel_width
    row = (point.y - gt.origin_y) / gt.pixel_height
    return Coord(row=int(math.floor(row)), col=int(math.floor(col)))


def coord_to_point(coord: Coord, geotransform: Geotransform) -> Point:
    """Map a raster Coord to a world-space Point (forward geotransform).

    The exact inverse of world_to_coord. Uses the full GDAL affine including
    the rotation terms; those are zero for this project's planos but are
    carried for generality. Note the axis swap: the column drives x,
    the row drives y.

    Args:
        coord: A pixel in full-image raster coordinates. ROI-local pixels must
            be lifted by the ROI offset first; polyline_to_world folds the
            offset in.
        geotransform: GDAL GetGeoTransform() 6-tuple.

    Returns:
        The pixel's location in world/CRS coordinates.
    """
    gt = geotransform
    x = gt.origin_x + coord.col * gt.pixel_width + coord.row * gt.rot_x
    y = gt.origin_y + coord.col * gt.rot_y + coord.row * gt.pixel_height
    return Point(x=x, y=y)


def polyline_to_world(
    polyline: Polyline, geotransform: Geotransform, offset: Offset = Offset(0, 0)
) -> list[Point]:
    """Convert a pixel-space Polyline to world-space Points in one pass.

    Folds in the ROI offset (lifting ROI-local pixels back to full-image space)
    and applies the geotransform as a single vectorized affine over all
    vertices, rather than per-Coord Python arithmetic.

    Args:
        polyline: A Polyline in ROI pixel coordinates.
        geotransform: GDAL GetGeoTransform() 6-tuple.
        offset: The (row, col) offset of the ROI within the full image.
            Defaults to (0, 0) for geometry already in full-image coordinates.

    Returns:
        The polyline's vertices as world-space Points, in order. The polyline's
        id is not carried by this function; preserve it at the call site if the
        downstream layer needs it.
    """
    pts = polyline.points
    n = len(pts)
    if n == 0:
        return []

    rows = np.fromiter((c.row for c in pts), dtype=float, count=n) + offset.row
    cols = np.fromiter((c.col for c in pts), dtype=float, count=n) + offset.col

    gt = geotransform
    xs = gt.origin_x + cols * gt.pixel_width + rows * gt.rot_x
    ys = gt.origin_y + cols * gt.rot_y + rows * gt.pixel_height
    return [Point(x=float(x), y=float(y)) for x, y in zip(xs, ys)]

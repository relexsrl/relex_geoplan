"""OpenCV-based parcel extraction workflow.

Pipeline (PLAN.md §10, validated 2026-06-26):
  1. Binarize (Otsu) inside the ROI.
  2. Stage C — separate parcel-boundary strokes from dimension/text ink via
     opening-by-reconstruction + a connected-component text filter.
  3. Stage A — clip to the user polygon, dilated so boundary strokes survive.
  4. Bridge dash gaps so dashed boundaries form one line.
  5. Detect lines, de-duplicate collinear ones, and extract parcel faces from
     their noded planar arrangement (PLAN.md §10).

The user draws a polygon ROI around the parcel(s) — it defines the analysis
area, not the parcel vertices, which are found automatically. The domain is
single parcels and N **adjacent** parcels that tile a region with coincident
shared edges (there is no nested/sub-parcel case). Validated on single
(486A IoU 0.97), adjacent pairs (04 0.99/0.98, coincident edges), arbitrary-N
adjacent (320 three parcels 0.92/0.97/0.98; 007 two adjacent 0.98/0.98), and
faint-boundary sheets via ``recover_weak_shared_boundaries`` (064177 group).
"""

from __future__ import annotations

import math
import sys
from typing import Any, cast

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon as ShapelyPolygon
from shapely.ops import polygonize, polygonize_full, unary_union

cv2: Any
try:
    import cv2
except ImportError:
    cv2 = None

from .models import Coord, Geotransform, Offset, Point, Polyline, Roi
from .preprocess import mask_padding
from .world import coord_to_point, polyline_to_world, world_to_coord

Shape2D = tuple[int, int]
LineSegment = tuple[Coord, Coord]
PointRing = list[Point]
BoundaryExtraction = tuple[
    np.ndarray,
    list[LineSegment],
    list[LineSegment],
    list[PointRing],
    list[PointRing],
]
ExtractionAttempt = tuple[
    np.ndarray,
    np.ndarray,
    list[LineSegment],
    list[LineSegment],
    list[PointRing],
    list[PointRing],
]


def _ensure_cv2() -> None:
    if cv2 is None:
        command = (
            "import subprocess, sys; "
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', "
            "'opencv-contrib-python-headless'])"
        )
        msg = (
            "OpenCV (cv2) is not installed.\n\n"
            "Install it into QGIS's Python environment, then restart QGIS.\n\n"
            "QGIS Python Console command:\n"
            f"{command}\n\n"
            f"Current QGIS Python path:\n{sys.executable}"
        )
        raise ImportError(msg)


def extract_with_cv2(
    array: np.ndarray,
    geotransform: Geotransform,
    roi: Roi,
    line_width_px: float | None = None,
    return_intermediates: bool = False,
    polygon_mask: np.ndarray | None = None,
    clip_polygon: list[Point] | None = None,
    recover_weak_shared_boundaries: bool = False,
    trace_boundary: bool = False,
    try_alternate_detector: bool = False,
) -> dict[str, Any]:
    """Extract parcel linework/polygons from a user-selected ROI.

    The user-drawn polygon defines the analysis area (ROI), not the parcel
    vertices: vertices come from the noded planar arrangement of the detected
    lines (see the module docstring for the stage breakdown).

    Args:
        array: Full image array.
        geotransform: GDAL geotransform.
        roi: Region of interest.
        line_width_px: Expected line width in pixels (optional).
        return_intermediates: If True, include debug images in result.
        polygon_mask: Deprecated for detection. Kept for compatibility, but not
            used as an early hard pixel clip because that distorts boundary lines.
        clip_polygon: Optional user-drawn polygon in world coordinates. Final
            polygon outputs are filtered against this limit after detection.
        recover_weak_shared_boundaries: If True, add long raw-threshold line
            candidates removed by stroke separation. Useful for faint adjacent
            shared boundaries, but may over-split noisy drawings.
        trace_boundary: If True, extract a SINGLE parcel by tracing its closed
            boundary loop and filling it, instead of the line arrangement. Robust
            for large/complex single parcels the arrangement fragments (e.g. PM6
            09_256, 47-vertex, 0.38 → 0.98). Must NOT be used for multi-parcel
            selections — one enclosing loop would merge them.
        try_alternate_detector: If True, try OpenCV FastLineDetector (EDLines-family)
            rescue candidates and keep them only when their ROI coverage is clearly
            better than the Hough candidate.

    Returns:
        Dict with keys ``polylines``, ``polygons``, and optionally
        ``debug`` (dict of debug arrays).
    """
    _ensure_cv2()

    # Crop first, then detect padding on the ROI: an interior parcel selection
    # has no warp padding, so this avoids flood-filling the full TIFF every run.
    array_roi = array[roi.row0:roi.row1, roi.col0:roi.col1]
    valid_roi = mask_padding(array_roi)
    offset = Offset(roi.row0, roi.col0)

    # 1. Binary mask
    threshold_mask = _binarize(array_roi, valid_roi)

    # Auto-estimate line width if not provided (granulometry; D19/§10-C).
    auto_width = line_width_px is None
    if auto_width:
        estimate_mask = threshold_mask
        if clip_polygon is not None and len(clip_polygon) >= 3:
            estimate_mask = threshold_mask & _plain_polygon_mask(
                clip_polygon, geotransform, offset, cast(Shape2D, threshold_mask.shape)
            )
        line_width_px = _estimate_line_width(estimate_mask)

    # Coarse-to-fine width fallback: granulometry over-estimates the stroke width on
    # faint-boundary sheets, so Stage C erases the boundary and nothing is detected.
    # In auto mode, retry at progressively smaller widths and keep the first that yields
    # faces (the auto width is kept as the baseline if none do). Working sheets detect on
    # the first try, so this adds no cost or risk there.
    candidates = _width_candidates(line_width_px) if auto_width else [line_width_px]
    # Escalate to weak-boundary recovery only as a last resort: if no width detects any
    # face, the boundary is too faint for Stage C even at small widths, so re-add the
    # raw-threshold lines it erased. Triggered only on total failure, so working sheets
    # are untouched and the over-split risk of recovery is bounded.
    recover_modes = [recover_weak_shared_boundaries]
    if (
        auto_width
        and not recover_weak_shared_boundaries
        and clip_polygon is not None
        and len(clip_polygon) >= 3
    ):
        recover_modes.append(True)

    # Width selection. A too-small width over-segments — many tiny noise faces (interior
    # text/dimension ink) fill more ROI than a few correct ones, so raw coverage *rewards*
    # over-segmentation (PM6 09_255: lw=4 → 184 faces, coverage 0.77 but IoU 0.13; lw=14 →
    # 2 faces, 0.58 / IoU 0.98). So among widths whose coverage is comparable (within
    # _WIDTH_SELECT_MARGIN of the best), prefer the LARGEST — least fragmented, most reliable.
    # A genuinely faint boundary still wins by a wide margin at a smaller width (285B:
    # 0.00 → 0.86), so faint recovery is preserved.
    def _select_width(recover: bool, prefer_largest: bool) -> tuple[float, float, ExtractionAttempt]:
        scored: list[tuple[float, float, ExtractionAttempt]] = []
        for candidate in candidates:
            attempt = _extract_at_width(
                threshold_mask, clip_polygon, geotransform, offset, candidate, recover,
            )
            score = _roi_coverage_score(attempt[4], clip_polygon)
            if score < 0.50:
                score = 0.001 if attempt[4] else 0.0
            scored.append((score, candidate, attempt))
        with_faces = [item for item in scored if item[2][4]]
        if not with_faces:
            return scored[0]
        best = max(item[0] for item in with_faces)
        if not prefer_largest:
            # Faint-recovery path: the over-segmentation bias does not apply (a small width is
            # legitimately needed), so take plain max coverage. Prefer-largest here wrongly
            # picks an over-covering large width (280: lw=10 0.50 over the correct lw=4 0.94).
            return max(with_faces, key=lambda item: item[0])
        eligible = [item for item in with_faces if item[0] >= best - _WIDTH_SELECT_MARGIN]
        return max(eligible, key=lambda item: item[1])

    for recover in recover_modes:
        _, line_width_px, attempt = _select_width(recover, prefer_largest=not recover)
        boundary, mask, lines, support_lines, polygons, world_lines = attempt
        if polygons:
            break

    min_polygon_area = _min_polygon_area(line_width_px, geotransform)
    clip_margin = 4.0 * line_width_px * abs(geotransform.pixel_width)

    # Single-parcel boundary tracing: for a large/complex parcel the line arrangement
    # fragments, so trace the closed boundary loop and fill it instead (single only — a
    # group would merge into one blob). Reuses the auto-selected line width.
    if trace_boundary:
        ring = _trace_boundary_face(
            threshold_mask, clip_polygon, geotransform, offset, line_width_px
        )
        traced = [ring] if ring is not None else []
        if traced and clip_polygon is not None:
            traced = _filter_polygons_contained_in_roi(traced, clip_polygon, clip_margin)
        if traced:
            polygons = traced
            world_lines = traced

    if try_alternate_detector and not trace_boundary and clip_polygon is not None:
        alternate = _alternate_detector_rescue(
            threshold_mask, clip_polygon, geotransform, offset, line_width_px,
            recover_weak_shared_boundaries, (boundary, mask, lines, support_lines, polygons, world_lines),
        )
        if alternate is not None:
            line_width_px, boundary, mask, lines, support_lines, polygons, world_lines = alternate

    if not trace_boundary and not _has_reasonable_polygons(polygons, clip_polygon):
        fallback_polylines = _fallback_polylines(support_lines[:8], cast(Shape2D, mask.shape))
        fallback_world_lines = [
            polyline_to_world(pl, geotransform, offset=offset) for pl in fallback_polylines
        ]
        fallback_polygons = _polygonize_world_lines(fallback_world_lines, min_polygon_area)
        if clip_polygon is not None:
            fallback_polygons = _filter_fallback_by_roi(fallback_polygons, clip_polygon)
            fallback_polygons = _filter_polygons_contained_in_roi(
                fallback_polygons, clip_polygon, clip_margin
            )
            fallback_polygons = _filter_polygons_by_roi_area(
                fallback_polygons, clip_polygon, clip_margin
            )
        if fallback_polygons:
            world_lines = fallback_world_lines
            polygons = fallback_polygons

    result = {"polylines": world_lines, "polygons": polygons}

    if return_intermediates:
        result["debug"] = {
            "threshold_mask": threshold_mask,
            "boundary": boundary,
            "mask": mask,
            "lines": lines,
            "dominant_lines": support_lines,
            "offset": offset,
            "line_width": line_width_px,
            # Topology diagnostics describe the line arrangement; skip them when the
            # boundary-trace mode bypassed it (its linework is one open ring, not faces).
            "diagnostics": {} if trace_boundary else _arrangement_diagnostics(
                world_lines, clip_polygon, min_polygon_area
            ),
        }

    return result


# Coverage gap a smaller line width must beat to be chosen over a larger one. Below this,
# the smaller width's higher coverage is over-segmentation noise, not a real boundary; above
# it, a faint boundary the larger width missed. Sized between 09_255 (+0.20 spurious) and
# 285B (+0.86 genuine faint recovery).
_WIDTH_SELECT_MARGIN = 0.30


def _width_candidates(auto_width: float) -> list[float]:
    """Coarse-to-fine line-width candidates: the auto estimate first, then smaller widths.

    The granulometry estimate is clamped to [6, 16] and over-estimates on faint sheets;
    a too-large width makes Stage C's opening-by-reconstruction erase the thin boundary.
    The smaller fallbacks (≈0.6×, 0.4×, and a 4 px floor) recover those sheets.
    """
    candidates = [auto_width]
    for smaller in (round(auto_width * 0.6), round(auto_width * 0.4), 4):
        value = float(smaller)
        if 3.0 <= value < auto_width and value not in candidates:
            candidates.append(value)
    return candidates


def _roi_coverage_score(
    polygons: list[PointRing], clip_polygon: list[Point] | None
) -> float:
    if not polygons:
        return 0.0
    if clip_polygon is None or len(clip_polygon) < 3:
        return 1.0

    roi = ShapelyPolygon([(point.x, point.y) for point in clip_polygon])
    if not roi.is_valid or roi.area <= 0:
        return 1.0

    faces = []
    for ring in polygons:
        candidate = ShapelyPolygon([(point.x, point.y) for point in ring])
        if candidate.is_valid and candidate.area > 0:
            faces.append(candidate)
    if not faces:
        return 0.0
    return unary_union(faces).intersection(roi).area / roi.area


def _extract_at_width(
    threshold_mask: np.ndarray,
    clip_polygon: list[Point] | None,
    geotransform: Geotransform,
    offset: Offset,
    line_width_px: float,
    recover_weak_shared_boundaries: bool,
    detector: str = "hough",
    cleanup_fracs: tuple[float, ...] = (0.25, 0.20),
) -> ExtractionAttempt:
    """Run Stage C + arrangement at one line width.

    Returns ``(boundary, mask, lines, support_lines, polygons, world_lines)``. Tries the
    raw boundary and the thin-stroke-suppressed boundary, keeping the suppressed result
    when it yields reasonable polygons (the corner/hatching cleanup).
    """
    boundary = _separate_strokes(threshold_mask, line_width_px)
    if clip_polygon is not None and len(clip_polygon) >= 3:
        boundary = boundary & _dilated_polygon_mask(
            clip_polygon, geotransform, offset, cast(Shape2D, boundary.shape), line_width_px
        )
    min_polygon_area = _min_polygon_area(line_width_px, geotransform)
    clip_margin = 4.0 * line_width_px * abs(geotransform.pixel_width)
    mask, lines, support_lines, polygons, world_lines = _extract_from_boundary(
        boundary, threshold_mask, geotransform, offset, line_width_px,
        min_polygon_area, clip_polygon, clip_margin, recover_weak_shared_boundaries, detector,
    )
    # Thin-stroke suppression sharpens corners and strips interior clutter. Try the
    # standard strength first (it strips clutter well on normal sheets, e.g. 007); if it
    # erases too much to yield faces — a very thin boundary, e.g. 320 — retry gentler
    # before giving up, so thin sheets still get a cleaned mask instead of falling back
    # to the cluttered raw boundary. Each sheet thus gets the strength it can survive.
    for frac in cleanup_fracs:
        cleaned_boundary = _suppress_thin_strokes(boundary, line_width_px, frac)
        if not cleaned_boundary.any():
            continue
        cleaned = _extract_from_boundary(
            cleaned_boundary, threshold_mask, geotransform, offset, line_width_px,
            min_polygon_area, clip_polygon, clip_margin, recover_weak_shared_boundaries, detector,
        )
        if _has_reasonable_polygons(cleaned[3], clip_polygon):
            mask, lines, support_lines, polygons, world_lines = cleaned
            break
    return boundary, mask, lines, support_lines, polygons, world_lines


def _alternate_detector_rescue(
    threshold_mask: np.ndarray,
    clip_polygon: list[Point],
    geotransform: Geotransform,
    offset: Offset,
    selected_width: float,
    recover_weak_shared_boundaries: bool,
    current: ExtractionAttempt,
) -> tuple[float, np.ndarray, np.ndarray, list[LineSegment], list[LineSegment], list[PointRing], list[PointRing]] | None:
    if not _has_fast_line_detector():
        return None

    current_score = _roi_coverage_score(current[4], clip_polygon)
    best = (current_score, selected_width, current)
    # A few widths FLD actually wins at (235 @10, 290 @7); each _extract_at_width already tries
    # raw + both cleanup strengths internally, so one call per width suffices — no separate
    # per-cleanup loop (that tripled the cost for no gain).
    widths = []
    for width in (selected_width, 4.0, 7.0, 10.0):
        if 3.0 <= width <= 16.0 and width not in widths:
            widths.append(float(width))
    for width in widths:
        attempt = _extract_at_width(
            threshold_mask, clip_polygon, geotransform, offset, width,
            recover_weak_shared_boundaries, detector="fld",
        )
        score = _roi_coverage_score(attempt[4], clip_polygon)
        if score > best[0]:
            best = (score, width, attempt)

    # Switch only when an FLD candidate covers the ROI MORE than Hough (best[2] is not current
    # ⇒ some FLD attempt beat Hough's coverage) by a small margin. An absolute floor (the old
    # max(0.50, cov+0.20)) blocked genuine single-parcel wins: a correct face covers only ~0.3
    # of the buffered-hull ROI (235 0.55→0.98, 03_18a 0.65→0.98 were blocked). Opt-in only.
    if best[2] is current or best[0] < current_score + 0.03:
        return None
    return best[1], best[2][0], best[2][1], best[2][2], best[2][3], best[2][4], best[2][5]


def _binarize(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Otsu threshold + morphology."""
    img = array.astype(np.uint8)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary.astype(bool)
    binary &= valid
    binary_u8 = binary.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def _dilated_polygon_mask(
    clip_polygon: list[Point],
    geotransform: Geotransform,
    offset: Offset,
    shape: Shape2D,
    line_width_px: float,
) -> np.ndarray:
    """ROI-local boolean mask of the user polygon, dilated to protect strokes.

    The dominant contamination (avenue, manzana grid, compass, title block) lies
    outside the parcels and outranks short parcel edges. ANDing the line mask
    with this dilated polygon removes it, while the margin keeps boundary strokes
    intact rather than clipping them (D22).
    """
    roi_pixels = []
    for point in clip_polygon:
        coord = world_to_coord(point, geotransform)
        roi_pixels.append([coord.col - offset.col, coord.row - offset.row])

    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.array(roi_pixels, dtype=np.int32)], 1)
    margin = max(1, int(round(line_width_px * 4)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
    )
    return cv2.dilate(canvas, kernel) > 0


def _plain_polygon_mask(
    clip_polygon: list[Point],
    geotransform: Geotransform,
    offset: Offset,
    shape: Shape2D,
) -> np.ndarray:
    """ROI-local boolean fill of the user polygon (no dilation)."""
    roi_pixels = []
    for point in clip_polygon:
        coord = world_to_coord(point, geotransform)
        roi_pixels.append([coord.col - offset.col, coord.row - offset.row])
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.array(roi_pixels, dtype=np.int32)], 1)
    return canvas > 0


def _estimate_line_width(mask: np.ndarray, r_max: int = 10, frac: float = 0.25) -> float:
    """Estimate boundary stroke width via granulometry (G&W Ch. 9).

    Opens with growing disks; the pattern spectrum's heavy-stroke population is
    the largest radius still losing significant area. Stroke width ~ 2r; a ~20%
    margin covers the gap/length thresholds that also scale with line_width_px.
    """
    if not mask.any():
        return 12.0
    # Granulometry on a downsampled mask: stroke-width *ratios* are scale-invariant,
    # so estimate at 1/scale resolution (≈ scale^2 less work) and rescale back.
    scale = 2
    small = mask[::scale, ::scale].astype(np.uint8)
    r_small = max(2, r_max // scale)
    areas = [int(cv2.countNonZero(small))]
    for r in range(1, r_small + 1):
        opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, _ellipse(r))
        areas.append(int(cv2.countNonZero(opened)))
    spectrum = [areas[i - 1] - areas[i] for i in range(1, len(areas))]
    peak = max(spectrum)
    if peak <= 0:
        return 12.0
    threshold = frac * peak
    boundary_r = max(r for r in range(1, len(spectrum) + 1) if spectrum[r - 1] >= threshold)
    # width ~ 2*boundary_r (in small px) * scale, + ~20% margin -> 2.4 * scale.
    return float(max(6, min(16, round(2.4 * boundary_r * scale))))


def _ellipse(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _open_by_reconstruction(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode to a thin-stroke-free marker, then reconstruct heavy strokes whole.

    Unlike a plain opening this keeps the surviving boundary strokes whole (no
    corner rounding). Morphological reconstruction, G&W Ch. 9. Erosion uses
    OpenCV (much faster than scipy for large structures); the geodesic
    reconstruction stays in scipy and needs full resolution for correctness.
    """
    marker = cv2.erode(mask.astype(np.uint8), _ellipse(radius)).astype(bool)
    return ndimage.binary_propagation(marker, mask=mask)


def _drop_text_components(mask: np.ndarray, line_width_px: float) -> np.ndarray:
    """Keep elongated line runs; drop compact text/symbol blobs. G&W Ch. 11."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask
    objects = ndimage.find_objects(labels)
    areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    label_max = line_width_px * 8
    for label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        area = float(areas[label - 1])
        extent = area / (height * width) if height * width else 0.0
        keep[label] = max(height, width) >= label_max or extent < 0.20
    return keep[labels]


def _separate_strokes(mask: np.ndarray, line_width_px: float) -> np.ndarray:
    """Stage C: isolate parcel-boundary strokes from dimension/text ink.

    Opening-by-reconstruction removes thin dimension/text strokes while keeping
    boundaries whole; a connected-component pass drops residual compact text.
    Replaces the weaker stroke-width percentile filter (PLAN.md D20/§10-C).
    """
    radius = max(1, int(round(line_width_px * 0.4)))
    boundary = _open_by_reconstruction(mask, radius)
    return _drop_text_components(boundary, line_width_px)


def _trace_boundary_face(
    threshold_mask: np.ndarray,
    clip_polygon: list[Point] | None,
    geotransform: Geotransform,
    offset: Offset,
    line_width_px: float,
) -> list[Point] | None:
    """Extract one parcel by tracing its closed boundary loop and filling it.

    The line arrangement fragments a large/complex single parcel — one missed edge breaks the
    face loop and interior lines (subdivisions/dimensions) split it. Isolating the boundary
    strokes (Stage C), closing gaps, filling the enclosed region, and contouring it is robust
    to interior content and to a few gaps (PM6 09_256, 47 vertices: 0.38 → 0.98). Returns one
    ring (list[Point]) or None. NOT for multi-parcel groups — a single enclosing loop merges
    them, which is why the region-fill engine was rejected as the general method.
    """
    boundary = _separate_strokes(threshold_mask, line_width_px)
    if clip_polygon is not None and len(clip_polygon) >= 3:
        boundary = boundary & _dilated_polygon_mask(
            clip_polygon, geotransform, offset, cast(Shape2D, boundary.shape), line_width_px
        )
    radius = max(1, int(round(line_width_px * 0.75)))
    bridged = _bridge_gaps(boundary, line_width_px)  # same close radius as here; reuse it
    filled = ndimage.binary_fill_holes(bridged)
    interior = filled & ~bridged
    labels, count = ndimage.label(interior)
    if count == 0:
        return None
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    region = labels == (int(np.argmax(sizes)) + 1)
    # Grow the interior back across the boundary stroke to its centreline, so the polygon
    # sits on the drawn line rather than inset by half the stroke width.
    region = ndimage.binary_dilation(region, iterations=radius)
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 0.002 * cv2.arcLength(contour, True), True)
    if len(approx) < 3:
        return None
    return [
        coord_to_point(
            Coord(row=int(point[0][1]) + offset.row, col=int(point[0][0]) + offset.col),
            geotransform,
        )
        for point in approx
    ]


def _bridge_gaps(mask: np.ndarray, line_width_px: float) -> np.ndarray:
    """Close dash gaps so a dashed boundary polygonizes as one line (Q-O2)."""
    radius = max(1, int(round(line_width_px * 0.75)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def _suppress_thin_strokes(mask: np.ndarray, line_width_px: float, frac: float) -> np.ndarray:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    min_half_width = max(1.0, frac * line_width_px)
    core = distance >= min_half_width
    radius = max(1, int(round(min_half_width)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    restored = cv2.dilate(core.astype(np.uint8), kernel).astype(bool)
    return mask & restored


def _extract_from_boundary(
    boundary: np.ndarray,
    threshold_mask: np.ndarray,
    geotransform: Geotransform,
    offset: Offset,
    line_width_px: float,
    min_polygon_area: float,
    clip_polygon: list[Point] | None,
    clip_margin: float,
    recover_weak_shared_boundaries: bool,
    detector: str = "hough",
) -> BoundaryExtraction:
    mask = _bridge_gaps(boundary, line_width_px)
    lines = _detect_lines(mask, line_width_px, detector)
    if recover_weak_shared_boundaries and clip_polygon is not None and len(clip_polygon) >= 3:
        lines.extend(
            _recover_removed_thin_lines(
                threshold_mask, boundary, clip_polygon, geotransform, offset, line_width_px
            )
        )
    support_lines = _find_dominant_lines(lines, line_width_px, max_lines=24)
    merged_lines = _merge_collinear_lines(lines, line_width_px)
    polygons, world_lines = _arrangement_faces(
        merged_lines, geotransform, offset, min_polygon_area, line_width_px
    )
    if clip_polygon is not None:
        polygons = _filter_polygons_contained_in_roi(polygons, clip_polygon, clip_margin)
        polygons = _filter_polygons_by_roi_area(polygons, clip_polygon, clip_margin)
    return mask, lines, support_lines, polygons, world_lines


def _recover_removed_thin_lines(
    threshold_mask: np.ndarray,
    boundary: np.ndarray,
    clip_polygon: list[Point],
    geotransform: Geotransform,
    offset: Offset,
    line_width_px: float,
) -> list[LineSegment]:
    clip_mask = _dilated_polygon_mask(
        clip_polygon, geotransform, offset, cast(Shape2D, threshold_mask.shape), line_width_px
    )
    radius = max(1, int(round(line_width_px * 0.5)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    removed = threshold_mask & clip_mask
    removed &= ~cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool)
    min_length = max(60, int(round(line_width_px * 22.5)))
    raw_lines = cv2.HoughLinesP(
        removed.astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_length,
        maxLineGap=max(20, int(round(line_width_px * 7.5))),
    )
    if raw_lines is None:
        return []
    return [
        (Coord(row=int(y1), col=int(x1)), Coord(row=int(y2), col=int(x2)))
        for x1, y1, x2, y2 in raw_lines[:, 0]
    ]


def _merge_collinear_lines(
    segments: list[LineSegment],
    line_width_px: float,
    angle_tol_deg: float = 8.0,
    offset_tol_factor: float = 1.8,
) -> list[LineSegment]:
    """Collapse near-duplicate/collinear Hough segments into one clean line each.

    HoughLinesP returns several parallel segments per drawn edge (within ~1 stroke
    width) plus collinear pieces of a dashed edge. Feeding those straight into the
    arrangement makes ``polygonize`` pick a slightly *inset* cycle (lower IoU). This
    clusters segments by orientation + perpendicular offset with a *tight* tolerance,
    then emits ONE line per cluster at the length-weighted **average** offset spanning
    the **union** of the members' extents.

    Unlike ``_find_dominant_lines`` (which keeps only the single longest member and so
    drops a parcel's dividing edge beyond ~2 parcels), this keeps **every distinct
    edge** — so all N faces still form — while centering duplicates for accurate edges
    and corners. The tight offset tolerance (~1 stroke width) merges duplicates of one
    edge without collapsing genuinely-distinct adjacent parcel edges.
    """
    features = []
    for a, b in segments:
        dx = float(b.col - a.col)
        dy = float(b.row - a.row)
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue
        angle = math.atan2(dy, dx) % math.pi
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux  # unit normal
        offset = a.col * nx + a.row * ny
        ta = a.col * ux + a.row * uy
        tb = b.col * ux + b.row * uy
        features.append(
            {"angle": angle, "offset": offset, "length": length,
             "tmin": min(ta, tb), "tmax": max(ta, tb)}
        )

    angle_tol = math.radians(angle_tol_deg)
    offset_tol = line_width_px * offset_tol_factor
    clusters: list[dict] = []
    for feature in sorted(features, key=lambda d: d["length"], reverse=True):
        for cluster in clusters:
            angle_diff = abs(feature["angle"] - cluster["angle"])
            angle_diff = min(angle_diff, math.pi - angle_diff)
            if angle_diff <= angle_tol and abs(feature["offset"] - cluster["offset"]) <= offset_tol:
                total = cluster["weight"] + feature["length"]
                cluster["angle"] = (cluster["angle"] * cluster["weight"] + feature["angle"] * feature["length"]) / total
                cluster["offset"] = (cluster["offset"] * cluster["weight"] + feature["offset"] * feature["length"]) / total
                cluster["weight"] = total
                cluster["tmin"] = min(cluster["tmin"], feature["tmin"])
                cluster["tmax"] = max(cluster["tmax"], feature["tmax"])
                break
        else:
            clusters.append({
                "angle": feature["angle"], "offset": feature["offset"],
                "weight": feature["length"], "tmin": feature["tmin"], "tmax": feature["tmax"],
            })

    lines = []
    for cluster in clusters:
        angle, offset = cluster["angle"], cluster["offset"]
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        base_col, base_row = offset * nx, offset * ny
        p0 = Coord(
            row=int(round(base_row + cluster["tmin"] * uy)),
            col=int(round(base_col + cluster["tmin"] * ux)),
        )
        p1 = Coord(
            row=int(round(base_row + cluster["tmax"] * uy)),
            col=int(round(base_col + cluster["tmax"] * ux)),
        )
        lines.append((p0, p1))
    return lines


def _arrangement_faces(
    segments: list[LineSegment],
    geotransform: Geotransform,
    offset: Offset,
    min_area: float,
    line_width_px: float,
) -> tuple[list[PointRing], list[PointRing]]:
    """Extract parcel faces from a noded planar arrangement of detected lines.

    The planar-arrangement engine (docs/investigation.md, PLAN.md §10): each detected
    segment is extended at both ends by a bounded corner-closing margin so adjacent
    boundary lines cross and corners close; all segments are then *noded* together
    (``unary_union`` splits every crossing into shared vertices) and ``polygonize``
    extracts the bounded faces. Noding is the step ``shapely.polygonize`` needs but
    does not do itself, and the reason the old custom noder was fragile.

    Critically this consumes the *raw* detected lines, not the merged dominant set:
    the dominant cap/near-parallel merge drops a parcel's dividing edge beyond ~2
    adjacent parcels, so the third face never forms. The redundant parallel Hough
    segments raw detection produces only create slivers, which the ``min_area``
    filter removes; coincident shared edges are guaranteed because every interior
    edge bounds exactly two faces.

    The corner-closing margin scales with stroke width (~8x), since the gap Hough
    leaves at a corner scales with the vertex-marker/stroke size.

    Args:
        segments: Raw detected line segments in ROI pixel coordinates.
        geotransform: GDAL geotransform.
        offset: ROI offset within the full image.
        min_area: Minimum face area (world units) to keep; drops slivers.
        line_width_px: Estimated stroke width; sets the extension margin.

    Returns:
        ``(faces, world_lines)`` — ``faces`` are polygon rings (world Points) sorted
        by descending area; ``world_lines`` are the extended segments in world space
        for debug/polyline output.
    """
    ext = max(1, int(round(line_width_px * 8)))
    world_lines: list[PointRing] = []
    shapely_lines = []
    for a, b in segments:
        dx = float(b.col - a.col)
        dy = float(b.row - a.row)
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue
        ux, uy = dx / length, dy / length
        p0 = coord_to_point(
            Coord(
                int(round(a.row - uy * ext)) + offset.row,
                int(round(a.col - ux * ext)) + offset.col,
            ),
            geotransform,
        )
        p1 = coord_to_point(
            Coord(
                int(round(b.row + uy * ext)) + offset.row,
                int(round(b.col + ux * ext)) + offset.col,
            ),
            geotransform,
        )
        world_lines.append([p0, p1])
        shapely_lines.append(LineString([(p0.x, p0.y), (p1.x, p1.y)]))

    if not shapely_lines:
        return [], world_lines

    faces = []
    for poly in polygonize(unary_union(shapely_lines)):
        if not poly.is_valid or poly.area < min_area:
            continue
        faces.append([Point(x=x, y=y) for x, y in list(poly.exterior.coords)[:-1]])
    faces.sort(key=lambda ring: abs(_ring_area(ring)), reverse=True)
    return faces, world_lines


def _arrangement_diagnostics(
    world_lines: list[PointRing],
    clip_polygon: list[Point] | None,
    min_area: float,
) -> dict:
    """Topology breakdown of the arrangement linework via ``polygonize_full``.

    Explains *why* a case under- or over-produces instead of guessing: the same noded
    linework the engine polygonizes is also passed to ``polygonize_full``, which separates
    it into faces, **dangles** (stubs that close no face → missing boundary), **cut edges**
    (bridges removed during face-building), and **invalid rings**. Lengths are in world
    units (metres). Diagnostic only — not used for selection here.
    """
    lines = [
        LineString([(p.x, p.y) for p in line]) for line in (world_lines or []) if len(line) >= 2
    ]
    if not lines:
        return {}
    polys, cuts, dangles, invalid = polygonize_full(unary_union(lines))
    face_areas = [g.area for g in polys.geoms]
    faces_in_roi = 0
    roi = None
    if clip_polygon is not None and len(clip_polygon) >= 3:
        candidate = ShapelyPolygon([(p.x, p.y) for p in clip_polygon])
        if candidate.is_valid and candidate.area > 0:
            roi = candidate
    if roi is not None:
        faces_in_roi = sum(
            1 for g in polys.geoms if g.intersection(roi).area > 0.5 * g.area and g.area >= min_area
        )
    return {
        "faces": len(face_areas),
        "faces_kept": sum(1 for a in face_areas if a >= min_area),
        "faces_sliver": sum(1 for a in face_areas if a < min_area),
        "faces_in_roi": faces_in_roi,
        "dangle_count": len(dangles.geoms),
        "dangle_length": round(sum(g.length for g in dangles.geoms), 1),
        "cut_edge_count": len(cuts.geoms),
        "cut_edge_length": round(sum(g.length for g in cuts.geoms), 1),
        "invalid_ring_count": len(invalid.geoms),
        "invalid_ring_length": round(sum(g.length for g in invalid.geoms), 1),
    }


def _polygonize_world_lines(
    world_lines: list[PointRing], min_area: float
) -> list[PointRing]:
    shapely_lines = [
        LineString([(point.x, point.y) for point in line])
        for line in world_lines
        if len(line) >= 2
    ]
    if not shapely_lines:
        return []

    polygons = []
    for poly in polygonize(unary_union(shapely_lines)):
        if poly.area < min_area:
            continue
        polygons.append([Point(x=x, y=y) for x, y in list(poly.exterior.coords)[:-1]])
    polygons.sort(key=lambda ring: abs(_ring_area(ring)), reverse=True)
    return polygons


def _filter_polygons_contained_in_roi(
    polygons: list[PointRing], clip_polygon: list[Point], margin: float = 0.0
) -> list[PointRing]:
    if len(clip_polygon) < 3:
        return polygons

    roi = ShapelyPolygon([(point.x, point.y) for point in clip_polygon])
    if not roi.is_valid or roi.area <= 0:
        return polygons
    if margin > 0:
        roi = roi.buffer(margin)  # the drawn polygon is a guide; tolerate a small overrun

    kept = []
    for ring in polygons:
        candidate = ShapelyPolygon([(point.x, point.y) for point in ring])
        if not candidate.is_valid or candidate.area <= 0:
            continue
        outside_area = candidate.difference(roi).area
        if outside_area <= candidate.area * 0.02:
            kept.append(ring)
    kept.sort(key=lambda ring: abs(_ring_area(ring)), reverse=True)
    return kept


def _filter_fallback_by_roi(
    polygons: list[PointRing], clip_polygon: list[Point]
) -> list[PointRing]:
    if len(clip_polygon) < 3:
        return polygons

    roi = ShapelyPolygon([(point.x, point.y) for point in clip_polygon])
    if not roi.is_valid or roi.area <= 0:
        return polygons

    kept = []
    for ring in polygons:
        candidate = ShapelyPolygon([(point.x, point.y) for point in ring])
        if not candidate.is_valid or candidate.area <= 0:
            continue
        if not candidate.intersects(roi):
            continue
        clipped_area = candidate.intersection(roi).area
        if clipped_area < roi.area * 0.20:
            continue
        if candidate.area > roi.area * 2.50:
            continue
        kept.append(ring)
    kept.sort(key=lambda ring: abs(_ring_area(ring)), reverse=True)
    return kept


def _filter_polygons_by_roi_area(
    polygons: list[PointRing], clip_polygon: list[Point], margin: float = 0.0
) -> list[PointRing]:
    if len(clip_polygon) < 3:
        return polygons

    roi = ShapelyPolygon([(point.x, point.y) for point in clip_polygon])
    if not roi.is_valid or roi.area <= 0:
        return polygons
    roi_area = roi.buffer(margin).area if margin > 0 else roi.area

    kept = []
    for ring in polygons:
        area = abs(_ring_area(ring))
        if area <= roi_area * 1.10:
            kept.append(ring)
    kept.sort(key=lambda ring: abs(_ring_area(ring)), reverse=True)
    return kept


def _has_reasonable_polygons(
    polygons: list[PointRing], clip_polygon: list[Point] | None
) -> bool:
    if not polygons:
        return False
    return True


def _ring_area(ring: list[Point]) -> float:
    area = 0.0
    for a, b in zip(ring, ring[1:] + ring[:1]):
        area += a.x * b.y - b.x * a.y
    return area / 2


def _min_polygon_area(line_width_px: float, geotransform: Geotransform) -> float:
    pixel_area = abs(geotransform.pixel_width * geotransform.pixel_height)
    return max(100.0, pixel_area * (line_width_px * 10) ** 2)


def _has_fast_line_detector() -> bool:
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector")


def _detect_lines(
    mask: np.ndarray, line_width_px: float, detector: str = "hough"
) -> list[LineSegment]:
    """Detect straight line segments."""
    if detector == "fld":
        return _detect_lines_fld(mask, line_width_px)
    if detector != "hough":
        raise ValueError(detector)
    img = mask.astype(np.uint8) * 255
    min_len = int(line_width_px * 10)
    max_gap = int(line_width_px * 5)

    lines = cv2.HoughLinesP(
        img, rho=1, theta=np.pi / 180, threshold=15,
        minLineLength=min_len, maxLineGap=max_gap
    )

    segments = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            segments.append((Coord(row=y1, col=x1), Coord(row=y2, col=x2)))
    return segments


def _detect_lines_fld(mask: np.ndarray, line_width_px: float) -> list[LineSegment]:
    if not _has_fast_line_detector():
        return []
    detector = cv2.ximgproc.createFastLineDetector(
        length_threshold=max(6, int(round(line_width_px * 3))),
        distance_threshold=1.414,
        canny_th1=20,
        canny_th2=40,
        canny_aperture_size=3,
        do_merge=True,
    )
    lines = detector.detect(mask.astype(np.uint8) * 255)
    if lines is None:
        return []
    segments = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        segments.append(
            (
                Coord(row=int(round(y1)), col=int(round(x1))),
                Coord(row=int(round(y2)), col=int(round(x2))),
            )
        )
    return segments


def _find_dominant_lines(
    segments: list[LineSegment],
    line_width_px: float,
    max_lines: int = 12,
    angle_tol_deg: float = 12.0,
    offset_tol_factor: float = 5.0,
) -> list[LineSegment]:
    """Group Hough segments by orientation/offset and keep the longest members.

    ``offset_tol_factor`` (× line width) sets how close two parallel segments must be
    to merge into one dominant line. Large values collapse genuinely-distinct adjacent
    parcel edges (dropping a dividing edge → a missing face); small values keep them
    separate while still merging the redundant Hough duplicates of a single edge
    (which sit within ~1 stroke width). ``angle_tol_deg`` is the orientation bin.
    """
    if not segments:
        return []

    features = []
    for a, b in segments:
        dx = b.col - a.col
        dy = b.row - a.row
        length = math.hypot(dx, dy)
        if length < line_width_px * 10:
            continue

        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += math.pi

        # Normal-form offset for an unoriented line: x*cos(theta)+y*sin(theta).
        normal_angle = angle + math.pi / 2
        offset = a.col * math.cos(normal_angle) + a.row * math.sin(normal_angle)
        features.append((angle, offset, length, a, b))

    angle_tol = math.radians(angle_tol_deg)
    offset_tol = line_width_px * offset_tol_factor
    groups: list[list[tuple[float, float, float, Coord, Coord]]] = []

    for feature in sorted(features, key=lambda item: item[2], reverse=True):
        angle, offset, _length, _a, _b = feature
        for group in groups:
            ref_angle, ref_offset, *_ = group[0]
            angle_diff = abs(angle - ref_angle)
            if angle_diff > math.pi / 2:
                angle_diff = math.pi - angle_diff
            if angle_diff <= angle_tol and abs(offset - ref_offset) <= offset_tol:
                group.append(feature)
                break
        else:
            groups.append([feature])

    dominant = []
    for group in groups:
        best = max(group, key=lambda item: item[2])
        dominant.append((best[3], best[4]))

    dominant.sort(
        key=lambda line: math.hypot(
            line[1].col - line[0].col,
            line[1].row - line[0].row,
        ),
        reverse=True,
    )
    return dominant[:max_lines]


def _compute_intersections(
    lines: list[LineSegment], shape: Shape2D
) -> list[Coord]:
    if len(lines) < 3:
        return []

    h, w = shape
    vertices = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = _line_intersection(lines[i], lines[j])
            if intersection is None:
                continue
            row, col = intersection
            margin = max(h, w) * 0.05
            if -margin <= row < h + margin and -margin <= col < w + margin:
                vertices.append(Coord(row=int(round(row)), col=int(round(col))))

    return _dedupe_coords(vertices, min(h, w) * 0.03)


def _fallback_polylines(
    lines: list[LineSegment], shape: Shape2D
) -> list[Polyline]:
    vertices = _compute_intersections(lines, shape)
    ordered = _order_vertices(vertices) if len(vertices) >= 3 else []
    return [
        Polyline(id=i, points=[ordered[i], ordered[(i + 1) % len(ordered)]])
        for i in range(len(ordered))
    ]


def _line_intersection(
    first: LineSegment, second: LineSegment
) -> tuple[float, float] | None:
    a1, a2 = first
    b1, b2 = second

    x1, y1 = float(a1.col), float(a1.row)
    x2, y2 = float(a2.col), float(a2.row)
    x3, y3 = float(b1.col), float(b1.row)
    x4, y4 = float(b2.col), float(b2.row)

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return py, px


def _dedupe_coords(coords: list[Coord], min_dist: float) -> list[Coord]:
    deduped = []
    for coord in coords:
        if all(
            math.hypot(coord.row - other.row, coord.col - other.col) > min_dist
            for other in deduped
        ):
            deduped.append(coord)
    return deduped


def _order_vertices(vertices: list[Coord]) -> list[Coord]:
    center_row = sum(v.row for v in vertices) / len(vertices)
    center_col = sum(v.col for v in vertices) / len(vertices)
    return sorted(
        vertices,
        key=lambda vertex: math.atan2(vertex.row - center_row, vertex.col - center_col),
    )



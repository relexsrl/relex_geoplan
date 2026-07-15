"""Harvest-on-confirm flywheel (QGIS-free).

Every etiqueta the user confirms — typed at extraction or filled in the
post-extraction dialog — identifies exactly one handwritten marker on one
plano. This module crops that marker's disk and appends it to a local store,
so normal daily use grows the labeled training set that gates every measured
reader route (see docs/ocr_training_improvement_research.md VERDICT).

Store layout (the plugin puts it in the QGIS project folder, never shipped):
    etiqueta_harvest/
        <nmp>_<label>.png     marker disk crop (grayscale; overwrite = dedup)
        harvest.jsonl         one line per save: file, label, nmp, source,
                              text_angle, ts (consumers keep the last per file)
"""
import json
import os
import time

import cv2
import numpy as np

from . import char_reader as cr
from . import etiqueta_ocr as eo

# thumbnail: below this CNN read-confidence the pick is probably the wrong
# marker, so show NO thumbnail (a misleading crop is worse than none) — the fill
# dialog falls back to its Locate button.
_THUMB_MIN_SCORE = 0.30


def _pick(masked, detected, contour, reader):
    """Select this polygon's marker for a THUMBNAIL (label unknown). Prefer the
    handwriting-aware CNN selector when a reader is available; fall back to the
    Hershey selector. Returns (disk, text_angle, score) or None."""
    if reader is not None:
        got = cr.best_marker_disk(reader[0], reader[1], masked, detected, contour)
        if got is not None:
            return got
    return eo.best_marker_disk(masked, detected, contour)


def marker_disks(gray, rel_polys, reader=None):
    """Marker disk for each polygon — for UI thumbnails in the fill dialog (the
    parcel is still blank, so the label is unknown). Returns a grayscale disk
    only when the pick is confident enough (>= _THUMB_MIN_SCORE); otherwise None,
    so the dialog shows no thumbnail rather than a misleading wrong crop. Aligned
    with rel_polys. Never raises."""
    try:
        contours = [np.array([[int(c), int(r)] for c, r in poly], np.int32)
                    for poly in rel_polys]
        if not contours:
            return []
        masked = eo.mask_to_polygons(gray, contours, 60)
        detected = eo.detect_markers(masked)
        out = []
        for contour in contours:
            got = _pick(masked, detected, contour, reader) if detected else None
            out.append(got[0] if got is not None and got[2] >= _THUMB_MIN_SCORE else None)
        return out
    except Exception:
        return [None] * len(rel_polys)


def harvest_disk(store_dir, disk, label, nmp, source="click", extra=None):
    """Save one already-cropped marker disk under its confirmed label. Used by
    the fill dialog's click-to-pick: the user pointed at the number, so the disk
    is trusted directly (no detection/verification needed). ``extra`` fields are
    merged into the jsonl event — the pick handlers pass the clicked raster
    location (pick_px / pick_rect_px), which is a gold marker-location label for
    a future learned detector. Returns True on save."""
    if disk is None or not label:
        return False
    try:
        os.makedirs(store_dir, exist_ok=True)
        name = f"{nmp or 'unknown'}_{label}.png"
        if not cv2.imwrite(os.path.join(store_dir, name), disk):
            return False
        with open(os.path.join(store_dir, "harvest.jsonl"), "a", encoding="utf-8") as index:
            index.write(json.dumps({
                "file": name, "label": label, "nmp": nmp or "",
                "source": source, "ts": int(time.time()),
                **(extra or {}),
            }, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def harvest_confirmed(store_dir, crop, rel_polys, confirmed, nmp):
    """Save the marker disk of every confirmed parcel.

    crop: grayscale pixel crop covering the polygons (from the extraction
    raster); rel_polys: per-polygon [(col, row), ...] in crop coords;
    confirmed: [(poly_index, label, source)] with source in
    {"extract", "user"}; nmp: the plano's registry number (stable key).

    LABEL-SPECIFIC: because the confirmed number is known, we harvest the marker
    that actually MATCHES that number (etiqueta_ocr.best_marker_for_label), not a
    generic code-like pick. If no marker in the polygon verifies against the
    label, nothing is saved — a wrong-marker crop under the label would poison
    the training store, and a missing example only slows it.
    Returns the number of disks saved. Never raises — the flywheel must not
    break extraction.
    """
    if not confirmed or crop is None:
        return 0
    try:
        os.makedirs(store_dir, exist_ok=True)
        contours = [np.array([[int(c), int(r)] for c, r in poly], np.int32)
                    for poly in rel_polys]
        masked = eo.mask_to_polygons(crop, contours, 60)
        detected = eo.detect_markers(masked)
        if not detected:
            return 0
        saved = 0
        index_path = os.path.join(store_dir, "harvest.jsonl")
        with open(index_path, "a", encoding="utf-8") as index:
            for i, label, source in confirmed:
                if not (0 <= i < len(contours)) or not label:
                    continue
                got = eo.best_marker_for_label(masked, detected, contours[i], label)
                # only harvest a marker that VERIFIES as the confirmed number;
                # else skip (don't poison the store with a wrong-marker crop).
                if got is None:
                    continue
                disk, text_angle, _fit = got
                name = f"{nmp or 'unknown'}_{label}.png"
                if not cv2.imwrite(os.path.join(store_dir, name), disk):
                    continue
                index.write(json.dumps({
                    "file": name, "label": label, "nmp": nmp or "",
                    "source": source, "text_angle": round(float(text_angle), 1),
                    "ts": int(time.time()),
                }, ensure_ascii=False) + "\n")
                saved += 1
        return saved
    except Exception:
        return 0

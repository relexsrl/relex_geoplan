"""Etiqueta (circled parcel number) reading — candidate-constrained (QGIS-free).

The marker is an elliptical ring with the parcel number inside, mostly
HANDWRITTEN (two known sheets, 064177/064281, are machine-printed). Validated
routes: detect the elliptical markers, segment the glyphs, and match them
against a KNOWN candidate label list (the typed etiquetas) using a
nearest-neighbor glyph library + Hungarian assignment; plus a free-read path
(``read_etiquetas``) gated hard by ``FREE_ACCEPT`` — everything doubtful stays
blank, because a wrong etiqueta poisons the cca.

The glyph library ships with the plugin (``data/etiqueta_glyphs.npz``);
``load_library`` accepts extra user libraries, though the plugin currently
loads only the bundled one.
"""
import math
import os
import re

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # scipy is a plugin runtime dep; degrade to greedy
    linear_sum_assignment = None

BUNDLED_LIBRARY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "etiqueta_glyphs.npz",
)

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLYPH_SHAPE = (36, 28)
_SHIFTS = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2))

ACCEPT_COST = 4200.0   # max mean per-char NN distance for a confident match
ACCEPT_MARGIN = 1.20   # best label must beat the runner-up by this factor

# Canonical etiqueta = exactly 4 chars: 3 digits + (digit | letter). Verified on
# all 634 DB labels (zero exceptions). Display form drops leading zeros, so a
# valid READ is 1-4 digits, or 1-3 digits + one final letter — never a letter
# before the last position, never 5 chars (the old \d{1,4}[A-Z]? allowed both).
_ETIQ_RE = re.compile(r"^\d{1,4}$|^\d{1,3}[A-ZÑ]$")
DIGITS = set("0123456789")
# Free reads must be nearly exact library matches: measured on GT, correct
# reads score 0-90 while wrong ones start ~650 — a wrong etiqueta poisons the
# cca, so everything doubtful stays blank for the user to fill.
FREE_ACCEPT = 200.0


# ---------------------------------------------------------------- glyphs

def norm_glyph(img):
    """Canonical glyph: minority-class ink, stroke-closed, size-normalized,
    centered on a fixed canvas, soft edges."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = (img < 180).astype(np.uint8) * 255
    if bw.mean() > 127:
        bw = 255 - bw
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(bw)
    canvas = np.zeros(GLYPH_SHAPE, np.uint8)
    if len(xs) == 0:
        return canvas
    glyph = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(24 / max(1, glyph.shape[1]), 32 / max(1, glyph.shape[0]))
    glyph = cv2.resize(glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y0 = (canvas.shape[0] - glyph.shape[0]) // 2
    x0 = (canvas.shape[1] - glyph.shape[1]) // 2
    canvas[y0:y0 + glyph.shape[0], x0:x0 + glyph.shape[1]] = glyph
    return cv2.GaussianBlur(canvas, (5, 5), 1.5)


def _hershey_templates():
    out = {}
    for ch in CHARS:
        img = np.full((70, 50), 255, np.uint8)
        cv2.putText(img, ch, (4, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 0, 3, cv2.LINE_AA)
        out[ch] = norm_glyph(img)
    return out


TEMPLATES = _hershey_templates()


def _cluster_rows(comps):
    rows = []
    for c in sorted(comps, key=lambda c: c[1] + c[3] / 2):
        cy = c[1] + c[3] / 2
        for row in rows:
            if abs(cy - row["cy"]) < max(c[3], row["h"]) * 0.5:
                row["comps"].append(c)
                row["cy"] = sum(v[1] + v[3] / 2 for v in row["comps"]) / len(row["comps"])
                row["h"] = max(row["h"], c[3])
                break
        else:
            rows.append({"cy": cy, "h": c[3], "comps": [c]})
    rows.sort(key=lambda r: r["cy"])
    return [r["comps"] for r in rows]


def _plausible_row(boxes):
    if not boxes:
        return False
    hs = sorted(b[3] for b in boxes)
    med = hs[len(hs) // 2]
    if med < 14:
        return False
    if any(b[3] < med * 0.55 or b[3] > med * 1.6 for b in boxes):
        return False
    rows = _cluster_rows(boxes)
    if len(rows) > 2 or (len(rows) == 2 and len(rows[1]) != 1):
        return False
    for row in rows:
        row_sorted = sorted(row, key=lambda b: b[0])
        centers = [b[1] + b[3] / 2 for b in row_sorted]
        if max(centers) - min(centers) > med * 0.7:
            return False
        for (x1, _y1, w1, _h1), (x2, _y2, _w2, _h2) in zip(row_sorted, row_sorted[1:]):
            if x2 < x1 + w1 * 0.3:
                return False
    return True


def segment_chars(img, angle):
    """Rotate, contrast-stretch (some markers are printed very faint),
    binarize, split into components in reading order."""
    side = max(img.shape)
    f = max(1.0, 260.0 / side)
    big = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 35, 35, 35, 35, cv2.BORDER_CONSTANT, value=255)
    H, W = big.shape
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rot = cv2.warpAffine(big, m, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    vals = rot[rot < 250]
    if vals.size > 30:
        lo = float(np.percentile(vals, 2))
        if 255.0 - lo > 30.0:
            rot = np.clip((rot.astype(np.float32) - lo) * (255.0 / (255.0 - lo)),
                          0, 255).astype(np.uint8)
    bw = (rot < 150).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20 or h < 12 or w < 4 or h > H * 0.8 or w > W * 0.8:
            continue
        comps.append((x, y, w, h))
    ordered = [c for row in _cluster_rows(comps) for c in sorted(row, key=lambda c: c[0])]
    return [(bw[max(0, y - 2):min(H, y + h + 2), max(0, x - 2):min(W, x + w + 2)],
             (x, y, w, h))
            for x, y, w, h in ordered]


def _sweep_angles(angle_prior):
    if angle_prior is None:
        return list(range(-45, 46, 5))
    out = []
    for base in (angle_prior, angle_prior + 180):
        for d in (-10, -5, 0, 5, 10):
            a = (base + d + 180) % 360 - 180
            out.append(a)
    return out


# ---------------------------------------------------------------- markers

def detect_ellipses(gray):
    """Elliptical ring markers (the etiqueta oval): fitEllipse on ink contours
    with a fit-quality test, plus a Hough-circle fallback."""
    bw = (gray < 190).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found = []
    for c in contours:
        if len(c) < 40:
            continue
        per = cv2.arcLength(c, False)
        if not (130 < per < 1200):
            continue
        (cx, cy), (a0, a1), ang = cv2.fitEllipse(c)
        minor, major = sorted((a0, a1))
        if not (20 < minor < 180 and 34 < major < 300) or major / minor > 3.2:
            continue
        pts = c[:, 0, :].astype(np.float32)
        m = int(max(a0, a1) / 2 + 8)
        wx0, wy0 = int(cx) - m, int(cy) - m
        side = 2 * m
        mask = np.zeros((side, side), np.uint8)
        cv2.ellipse(mask, ((cx - wx0, cy - wy0), (a0, a1), ang), 255, 5)
        sx = (pts[:, 0] - wx0).astype(int)
        sy = (pts[:, 1] - wy0).astype(int)
        ok = (sx >= 0) & (sx < side) & (sy >= 0) & (sy < side)
        if len(pts) < 30:
            continue
        on_frac = float((mask[sy[ok], sx[ok]] > 0).sum()) / len(pts)
        if on_frac < 0.72:
            continue
        found.append((float(cx), float(cy), float(a0), float(a1), float(ang), on_frac))
    dedup = []
    for e in sorted(found, key=lambda v: -v[5]):
        if not any((e[0] - d[0]) ** 2 + (e[1] - d[1]) ** 2 < 30 ** 2 for d in dedup):
            dedup.append(e[:5])

    def ring_support(x, y, r):
        m = int(r) + 6
        x0, y0 = int(x) - m, int(y) - m
        win = gray[max(0, y0):y0 + 2 * m, max(0, x0):x0 + 2 * m]
        if win.size == 0:
            return 0.0
        mask = np.zeros(win.shape, np.uint8)
        cv2.circle(mask, (int(x) - max(0, x0), int(y) - max(0, y0)), int(r), 255, 4)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0.0
        return float(np.mean(win[ys, xs] < 170))

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=45,
                               param1=80, param2=28, minRadius=20, maxRadius=95)
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(float):
            if ring_support(x, y, r) < 0.18:
                continue
            if not any((x - d[0]) ** 2 + (y - d[1]) ** 2 < 30 ** 2 for d in dedup):
                dedup.append((float(x), float(y), float(r * 2), float(r * 2), 0.0))
    return dedup


def extract_disk(gray, x, y, a0, a1, ang):
    """Cut the marker interior: erase the ring, blank outside the ellipse."""
    half = int(max(a0, a1) / 2 + 8)
    y0, y1 = max(0, int(y) - half), min(gray.shape[0], int(y) + half)
    x0, x1 = max(0, int(x) - half), min(gray.shape[1], int(x) + half)
    disk = gray[y0:y1, x0:x1].copy()
    ring_w = max(5, int(min(a0, a1) * 0.09))
    cv2.ellipse(disk, ((x - x0, y - y0), (a0, a1), ang), 255, ring_w)
    outside = np.zeros_like(disk)
    cv2.ellipse(outside, ((x - x0, y - y0), (a0, a1), ang), 255, -1)
    disk[outside == 0] = 255
    text_angle = ang + 90 if a1 >= a0 else ang
    return disk, text_angle


def disk_norm_rows(disk, text_angle):
    """Plausible segmentations of a disk as normalized glyph rows. The ellipse
    angle prior is tried first; the blind sweep only runs when it yields
    nothing (it rarely does, and it triples the cost)."""
    rows, seen = [], set()

    def sweep(angles):
        for angle in angles:
            if angle in seen:
                continue
            seen.add(angle)
            glyphs = segment_chars(disk, angle)
            if not glyphs or not _plausible_row([b for _g, b in glyphs]):
                continue
            rows.append([norm_glyph(g) for g, _b in glyphs])

    sweep(_sweep_angles(text_angle))
    sweep(range(-45, 46, 5))
    return rows


def mask_to_polygons(gray, contours, margin):
    """White out everything farther than ``margin`` from the polygons, so the
    marker search runs only in the relevant section (faster, fewer false
    ellipses from dimension text and neighboring parcels)."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, contours, 255)
    if margin > 0:
        cv2.polylines(mask, contours, True, 255, thickness=int(2 * margin))
    out = np.full_like(gray, 255)
    out[mask > 0] = gray[mask > 0]
    return out


def detect_markers(gray, max_side=3600.0):
    """Ellipse markers, detected on a downscaled crop for speed; parameters
    rescaled to full resolution."""
    scale = min(1.0, max_side / max(gray.shape))
    small = (cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1.0 else gray)
    out = []
    for x, y, a0, a1, ang in detect_ellipses(small):
        x, y, a0, a1 = x / scale, y / scale, a0 / scale, a1 / scale
        if 35 <= max(a0, a1) <= 320:
            out.append((x, y, a0, a1, ang))
    return out


def marker_score(disk, text_angle):
    """Reading-free plausibility [0,1] that a marker disk holds an ETIQUETA (a
    short 1-4 char parcel code) rather than a dimension callout / bearing note /
    lot label. An etiqueta segments into 1-4 similar-height glyphs that match the
    Hershey digit/letter templates well; dimension text has a decimal comma and
    many components (fails _plausible_row / the ≤4 cap), bearings carry °/', and
    labels are wide multi-word rows. Higher = more etiqueta-like; 0 = not.
    Orientation-agnostic (sweeps angles)."""
    best = 0.0
    seen = set()
    # the ellipse angle prior is often off, so the readable orientation may only
    # appear in the blind sweep — include it (same angle set the readers use).
    for angle in _sweep_angles(text_angle) + list(range(-45, 46, 15)):
        if angle in seen:
            continue
        seen.add(angle)
        glyphs = segment_chars(disk, angle)
        if not (1 <= len(glyphs) <= 4):
            continue
        boxes = [b for _g, b in glyphs]
        if not _plausible_row(boxes):
            continue
        fits = []
        for g, _b in glyphs:
            n = norm_glyph(g).astype(np.float32)
            fits.append(min(float(np.mean((n - t.astype(np.float32)) ** 2))
                            for t in TEMPLATES.values()))
        # real etiqueta glyphs fit the nearest template at ~3000-8000 MSE
        # (the readers accept up to 8000); scale so that range stays positive.
        best = max(best, max(0.0, 1.0 - float(np.mean(fits)) / 10000.0))
        if best > 0.6:  # clearly a code at this orientation — stop sweeping
            break
    return best


def markers_in_polygon(gray, detected, contour, top=4):
    """The detected markers whose centre is INSIDE this polygon and clear of the
    boundary, most-interior first (the read_etiquetas ownership rule), capped at
    `top`. Returns [(dist, x, y, a0, a1, ang)]."""
    inside = sorted(((cv2.pointPolygonTest(contour, (m[0], m[1]), True), m)
                     for m in detected), key=lambda v: -v[0])
    out = []
    for dist, m in inside[:top]:
        if dist < 15:
            break
        out.append((dist,) + tuple(m))
    return out


def best_marker_disk(gray, detected, contour, min_score=0.2):
    """Pick this polygon's etiqueta marker. Content is a GATE, interiority is the
    SELECTOR: score each inside marker for how code-like it is (marker_score),
    keep those that plausibly hold a short code (>= min_score, i.e. glyphs that
    fit the templates about as well as the readers require), and among THOSE take
    the most-interior — the parcel's own number sits deepest inside, while a
    dimension callout / lot label either fails the gate or is shallower. Falls
    back to the most-interior marker overall when none looks like a code.
    Returns (disk, text_angle, score) or None. (markers_in_polygon returns the
    candidates most-interior first.)"""
    cands = markers_in_polygon(gray, detected, contour)
    if not cands:
        return None
    best = None  # (dist, disk, text_angle, score) among code-like markers
    fallback = None
    for dist, x, y, a0, a1, ang in cands:
        disk, text_angle = extract_disk(gray, x, y, a0, a1, ang)
        if fallback is None:
            fallback = (disk, text_angle, 0.0)  # first = most interior
        score = marker_score(disk, text_angle)
        if score >= min_score and (best is None or dist > best[0]):
            best = (dist, disk, text_angle, score)
    if best is not None:
        return best[1], best[2], best[3]
    return fallback


def crop_at_rect(gray, col0, row0, col1, row1, pad=4):
    """Literal raster crop of a user-drawn marker rectangle. QGIS-free helper for
    the fill dialog's Pick tool; returns (crop, text_angle) or None."""
    h, w = gray.shape
    c0, c1 = sorted((int(round(col0)), int(round(col1))))
    r0, r1 = sorted((int(round(row0)), int(round(row1))))
    c0, c1 = max(0, c0 - pad), min(w, c1 + pad)
    r0, r1 = max(0, r0 - pad), min(h, r1 + pad)
    if c1 - c0 < 10 or r1 - r0 < 10:
        return None
    crop = gray[r0:r1, c0:c1].copy()
    return (crop, 0.0) if crop.size else None


def disk_at_point(gray, col, row, half=95):
    """Literal centred crop around a clicked number. Earlier versions tried to
    infer the nearest ellipse; on cluttered plans that could jump to a wrong
    marker. Click/Pick should mean exactly where the user pointed."""
    return crop_at_rect(gray, col - half, row - half, col + half, row + half, pad=0)


def best_marker_for_label(gray, detected, contour, label, fit_max=8000.0):
    """The inside marker whose segmentation best matches a display variant of the
    KNOWN label (Hershey fit). For the harvest flywheel, which knows the confirmed
    number: this VERIFIES the crop is that number before it enters the training
    store, so a wrong-section grab can never be saved under the label (a wrong
    label poisons training — a missing one only slows it). Returns
    (disk, text_angle, fit) of the best match, or None when nothing matches below
    fit_max. Only harvest what this returns."""
    variants = display_variants(label)
    lens = {len(v) for v in variants}
    best = None
    for _dist, x, y, a0, a1, ang in markers_in_polygon(gray, detected, contour):
        disk, text_angle = extract_disk(gray, x, y, a0, a1, ang)
        seen = set()
        for angle in _sweep_angles(text_angle) + list(range(-45, 46, 15)):
            if angle in seen:
                continue
            seen.add(angle)
            glyphs = segment_chars(disk, angle)
            if not glyphs or len(glyphs) not in lens:
                continue
            if not _plausible_row([b for _g, b in glyphs]):
                continue
            norms = [norm_glyph(g).astype(np.float32) for g, _b in glyphs]
            for target in variants:
                if len(target) != len(norms) or any(c not in TEMPLATES for c in target):
                    continue
                fit = float(np.mean([np.mean((n - TEMPLATES[c].astype(np.float32)) ** 2)
                                     for n, c in zip(norms, target)]))
                if fit <= fit_max and (best is None or fit < best[2]):
                    best = (disk, text_angle, fit)
    return best


# ---------------------------------------------------------------- library

def load_library(extra_paths=()):
    """Glyph samples: bundled npz + optional user libraries. Returns
    [(char, norm_glyph)] or [] when nothing is available."""
    samples = []
    for path in (BUNDLED_LIBRARY, *extra_paths):
        if not path or not os.path.exists(path):
            continue
        try:
            data = np.load(path, allow_pickle=False)
            labels, glyphs = data["labels"], data["glyphs"]
        except Exception:
            continue
        for ch, g in zip(labels, glyphs):
            samples.append((str(ch), g.astype(np.uint8)))
    return samples


def save_library(path, samples):
    labels = np.array([ch for ch, _g in samples])
    glyphs = np.stack([g for _ch, g in samples]).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, labels=labels, glyphs=glyphs)


_STACKS_CACHE = {}


def _char_stacks(samples):
    key = (id(samples), len(samples))
    hit = _STACKS_CACHE.get(key)
    if hit is None:
        by = {}
        for ch, n in samples:
            by.setdefault(ch, []).append(n)
        hit = {ch: np.stack(lst).astype(np.float32) for ch, lst in by.items()}
        _STACKS_CACHE[key] = hit
    return hit


def _min_dist(norm, stack):
    norm = norm.astype(np.float32)
    best = float("inf")
    for dy, dx in _SHIFTS:
        q = np.roll(np.roll(norm, dy, axis=0), dx, axis=1)
        d = float(np.min(np.mean((stack - q[None]) ** 2, axis=(1, 2))))
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------- matching

def display_variants(label):
    """How a stored etiqueta may be drawn on the sheet: 0004 -> 04, 003A -> 03A."""
    label = str(label).strip().upper()
    out = {label, label.lstrip("0") or "0"}
    if len(label) == 4 and label[:-1].isdigit() and label[-1].isalpha():
        out.add(label[1:])
    if len(label) == 4 and label.isdigit() and label.startswith("00"):
        out.add(label[-2:])
    return sorted(out, key=len)


def label_cost(rows, label, samples):
    """Best mean per-char NN distance of any segmentation as any variant."""
    stacks = _char_stacks(samples)
    best = float("inf")
    for target in display_variants(label):
        for row in rows:
            if len(row) != len(target):
                continue
            vals = []
            for norm, ch in zip(row, target):
                stack = stacks.get(ch)
                if stack is not None:
                    vals.append(_min_dist(norm, stack))
                elif ch in TEMPLATES:
                    vals.append(float(np.mean(
                        (norm.astype(np.float32) - TEMPLATES[ch].astype(np.float32)) ** 2
                    )) * 1.35)
                else:
                    vals = []
                    break
            if vals:
                best = min(best, float(np.mean(vals)))
    return best


def _ownership_mult(dist):
    if dist < 0:
        return 1.30
    return 1.0 + 0.30 * math.exp(-dist / 100.0)


def _classify_row_free(norms, samples):
    """Free-read one glyph row: nearest sample per position (letters only in the
    final slot). Returns (text, mean_distance)."""
    stacks = _char_stacks(samples)
    digit_stacks = {c: s for c, s in stacks.items() if c in DIGITS}
    text, scores = "", []
    for i, norm in enumerate(norms):
        pool = stacks if i == len(norms) - 1 else digit_stacks
        best_c, best_d = "", float("inf")
        for ch, stack in pool.items():
            d = _min_dist(norm, stack)
            if d < best_d:
                best_c, best_d = ch, d
        if not best_c:
            return "", float("inf")
        text += best_c
        scores.append(best_d)
    return text, float(np.mean(scores)) if scores else float("inf")


def read_etiquetas(gray, polygons_px, samples, debug_scores=None):
    """Free-read the circled number of EACH polygon (no typed list needed).

    SIMPLE ownership rule (Martin's): only markers INSIDE the created polygon
    count — every measured wrong read came from a neighbor's marker outside;
    parcels whose etiqueta is drawn outside simply stay blank for the user to
    type. Pattern-valid confident reads only — a blank beats a wrong number in
    the cca. Marker glyph rows are computed lazily, per polygon.
    Returns (labels, confident) per polygon.
    """
    n = len(polygons_px)
    out, conf = [""] * n, [False] * n
    if not samples or n == 0:
        return out, conf

    contours = [np.array([[int(c), int(r)] for c, r in poly], np.int32)
                for poly in polygons_px]
    gray = mask_to_polygons(gray, contours, 60)

    detected = detect_markers(gray)
    if not detected:
        return out, conf
    rows_cache = {}

    def rows_for(k):
        if k not in rows_cache:
            x, y, a0, a1, ang = detected[k]
            disk, text_angle = extract_disk(gray, x, y, a0, a1, ang)
            rows_cache[k] = disk_norm_rows(disk, text_angle)
        return rows_cache[k]

    used_labels = set()
    for i, contour in enumerate(contours):
        inside = sorted(
            ((cv2.pointPolygonTest(contour, (m[0], m[1]), True), k)
             for k, m in enumerate(detected)),
            key=lambda v: -v[0],
        )
        best_text, best_score, best_rank = "", float("inf"), float("inf")
        for dist, k in inside[:3]:
            if dist < 15:  # inside this polygon only — and clear of the
                break      # boundary, where a neighbor's marker can straddle
            for norms in rows_for(k):
                t, s = _classify_row_free(norms, samples)
                # prefer longer reads: a lone well-matched glyph from a wrong
                # angle scores deceptively low
                rank = s - 800.0 * len(t)
                if _ETIQ_RE.match(t) and rank < best_rank:
                    best_text, best_score, best_rank = t, s, rank
        if debug_scores is not None:
            debug_scores.append((i, best_text, best_score))
        if not best_text or len(best_text) < 2 or best_score > FREE_ACCEPT:
            continue
        if best_text.upper() in used_labels:
            continue
        out[i] = best_text
        conf[i] = True
        used_labels.add(best_text.upper())
    return out, conf


def match_etiquetas(gray, polygons_px, labels, samples, outside_max=180):
    """Assign candidate labels to polygons by reading their markers.

    gray: full crop containing all polygons. polygons_px: per polygon, the
    pixel-coordinate ring [(col, row), ...]. labels: the typed etiquetas.
    Returns (assigned, confident): ``assigned[i]`` is the label for polygon i
    ("" when unresolved); ``confident[i]`` says the OCR match was decisive.
    """
    n = len(polygons_px)
    assigned, confident = [""] * n, [False] * n
    if not samples or not labels or n == 0:
        return assigned, confident

    contours = [np.array([[int(c), int(r)] for c, r in poly], np.int32)
                for poly in polygons_px]
    gray = mask_to_polygons(gray, contours, outside_max + 60)

    detected = detect_markers(gray)
    if not detected:
        return assigned, confident

    # cost per (polygon, label): best marker readable as that label, weighted
    # by how much the marker belongs to the polygon. Rows computed lazily —
    # only for markers actually near some polygon.
    big = 1e9
    cost = np.full((n, len(labels)), big)
    for mx, my, a0, a1, ang in detected:
        dists = [cv2.pointPolygonTest(c, (mx, my), True) for c in contours]
        if max(dists) < -outside_max:
            continue
        disk, text_angle = extract_disk(gray, mx, my, a0, a1, ang)
        rows = disk_norm_rows(disk, text_angle)
        if not rows:
            continue
        for j, label in enumerate(labels):
            c = label_cost(rows, label, samples)
            if c == float("inf"):
                continue
            for i, d in enumerate(dists):
                if d < -outside_max:
                    continue
                v = c * _ownership_mult(d)
                if v < cost[i, j]:
                    cost[i, j] = v

    if linear_sum_assignment is not None:
        rows_i, cols_j = linear_sum_assignment(cost)
        pairs = [(cost[i, j], i, j) for i, j in zip(rows_i, cols_j) if cost[i, j] < big]
    else:
        order = sorted(((cost[i, j], i, j) for i in range(n) for j in range(len(labels))
                        if cost[i, j] < big))
        used_i, used_j, pairs = set(), set(), []
        for c, i, j in order:
            if i in used_i or j in used_j:
                continue
            pairs.append((c, i, j))
            used_i.add(i)
            used_j.add(j)

    stacks = _char_stacks(samples)

    def _unsampled(label):
        return any(ch not in stacks for ch in str(label).strip().upper())

    def _sibling(a, b):
        a, b = str(a).strip().upper(), str(b).strip().upper()
        if a == b:
            return False
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) <= 1
        return abs(len(a) - len(b)) == 1 and (a in b or b in a)

    for c, i, j in pairs:
        others = [cost[i, k] for k in range(len(labels)) if k != j and cost[i, k] < big]
        runner = min(others) if others else float("inf")
        if c >= ACCEPT_COST or runner / max(c, 1e-9) < ACCEPT_MARGIN:
            continue
        # a sibling label whose chars have no library samples can't compete
        # fairly (its cost is template-inflated) — the margin would be false
        # confidence, so leave the polygon for the spatial assignment instead
        if any(_sibling(labels[j], l) and _unsampled(l)
               for k, l in enumerate(labels) if k != j):
            continue
        assigned[i] = str(labels[j]).strip()
        confident[i] = True
    return assigned, confident

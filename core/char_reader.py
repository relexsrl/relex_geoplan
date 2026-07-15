"""Handwriting etiqueta reader — per-character CNN via cv2.dnn (QGIS-free).

A small CharCNN (EMNIST-pretrained, fine-tuned on harvested etiqueta glyphs)
classifies each segmented glyph. Reads are GATED on per-character softmax
confidence: a wrong etiqueta poisons the cca, so anything below the gate stays
blank. At/above AUTO_GATE a read auto-assigns a REAL etiqueta (measured
zero-wrong); in the SUGGEST band it only fills a placeholder hint in the
fill-in dialog. Runs after the template matcher in ``etiqueta_ocr``, filling
polygons it left blank.

Ships as ``data/char_reader.onnx`` + ``char_reader_2..5.onnx`` (a 5-seed
ensemble; the ``char_reader.json`` sidecar lists the members) — always load via
``load_reader``, never a single ONNX directly: one member alone is NOT the
shipped behavior (single-seed was measured at 2 confident wrongs at scale where
the ensemble has 0). Inference runs on cv2.dnn (CPU) — no torch at runtime.
"""
import json
import os
import warnings

import cv2
import numpy as np

from . import etiqueta_ocr as eo

BUNDLED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "char_reader.onnx",
)

# Two gates on the min per-char softmax prob (held-out etiqueta-level sweep on
# unseen sheets):
#   AUTO   >=0.90 -> writes a REAL etiqueta/cca. Zero wrong measured (the single
#          wrong read sat at 0.59), so this respects the "a wrong etiqueta poisons
#          the cca — blank beats wrong" bar. Low coverage (~3%) is the price.
#   SUGGEST>=0.50 -> PLACEHOLDER hint the user types over (16% coverage @ 90%);
#          a wrong hint is cheap because it never becomes a value on its own.
AUTO_GATE = 0.90
SUGGEST_GATE = 0.50
JUNK = "JUNK"


class _EnsembleNet:
    """Seed-ensemble behind the cv2.dnn net interface: forward() returns the
    log of the members' mean softmax, so the downstream constrained softmax
    renormalizes the averaged probabilities. Single-seed confident flukes
    (measured: 0008->000 @0.91, 383A->38A @0.97 on one seed) average below the
    AUTO gate while consistent true reads reinforce — the 5-seed ensemble read
    68/570 train parcels at 0.90 with 0 reader wrongs vs 47/570 (2 wrongs) for
    the previous single-seed model."""

    def __init__(self, nets):
        self.nets = nets
        self._blob = None

    def setInput(self, blob):
        self._blob = blob

    def forward(self):
        probs = []
        for net in self.nets:
            net.setInput(self._blob)
            logits = net.forward()
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs.append(e / e.sum(axis=1, keepdims=True))
        return np.log(np.clip(np.mean(probs, axis=0), 1e-9, 1.0))


def load_reader(onnx_path=BUNDLED, sidecar_path=None):
    """(net, classes) or None when the model isn't bundled / cv2.dnn can't load.
    When the sidecar lists ``members`` (seed ensemble), all member ONNX files
    are loaded from the same directory and wrapped in _EnsembleNet."""
    if not os.path.exists(onnx_path):
        return None
    sidecar_path = sidecar_path or (os.path.splitext(onnx_path)[0] + ".json")
    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
        classes = sidecar["classes"]
        members = sidecar.get("members")
        if members:
            base = os.path.dirname(onnx_path)
            paths = [os.path.join(base, m) for m in members]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                # partial deploy: refuse loudly rather than silently degrading —
                # a single member is NOT the validated zero-wrong reader
                warnings.warn(f"char_reader ensemble members missing: {missing}")
                return None
            return _EnsembleNet([cv2.dnn.readNetFromONNX(p) for p in paths]), classes
        net = cv2.dnn.readNetFromONNX(onnx_path)
    except Exception:
        return None
    return net, classes


def norm_char(bw):
    """segment_chars glyph (white ink on black) -> 28x28, ~20px content, centered
    (EMNIST layout). MUST stay identical to make_char_glyphs.norm_char used in
    training, or the runtime distribution drifts from what the net learned."""
    ys, xs = np.nonzero(bw)
    canvas = np.zeros((28, 28), np.uint8)
    if len(xs) == 0:
        return canvas
    crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = min(20.0 / max(1, h), 20.0 / max(1, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = crop
    return canvas


def _logits(net, glyphs):
    """glyphs: list of 28x28 uint8 -> log-scores [n, n_classes] in one pass:
    raw logits for a single net, log-mean-softmax for _EnsembleNet. Only
    shift-invariant softmax may be applied to the result — do not treat the
    values as calibrated logits."""
    blob = np.stack([g.astype(np.float32) / 255.0 for g in glyphs])[:, None]
    net.setInput(blob)
    return net.forward()


def _classify_constrained(net, classes, glyphs):
    """Position-constrained decode (verified format: every char before the last
    is a DIGIT; the last is digit or letter). Non-final glyphs softmax over
    digits+JUNK only — a letter can't occur there, so letter lookalikes (0->C)
    are excluded rather than competing. JUNK stays allowed everywhere so junk
    segmentations are still rejected. The subset softmax is shift-invariant,
    which is what keeps it valid on _logits' ensemble log-mean-probs (it
    renormalizes the averaged probabilities). Returns (chars, confidences)."""
    logits = _logits(net, glyphs)
    n = len(glyphs)
    digit_junk = [i for i, c in enumerate(classes) if c.isdigit() or c == JUNK]
    chars, confs = [], []
    for i in range(n):
        allowed = digit_junk if i < n - 1 else range(len(classes))
        row = logits[i]
        idx = np.array(list(allowed))
        e = np.exp(row[idx] - row[idx].max())
        p = e / e.sum()
        k = int(p.argmax())
        chars.append(classes[int(idx[k])])
        confs.append(float(p[k]))
    return chars, np.array(confs)


def _collect_reads(net, classes, disk, angles):
    """All pattern-valid reads across the given angles -> list of (text, min_conf)."""
    out = []
    seen = set()
    for angle in angles:
        a = round(float(angle), 3)
        if a in seen:
            continue
        seen.add(a)
        glyphs = eo.segment_chars(disk, angle)
        if not glyphs or not eo._plausible_row([b for _g, b in glyphs]):
            continue
        chars, confs = _classify_constrained(net, classes, [norm_char(g) for g, _b in glyphs])
        if JUNK in chars:
            continue
        text = "".join(chars)
        if len(text) < 2 or not eo._ETIQ_RE.match(text):
            continue
        out.append((text, float(confs.min())))
    return out


def read_disk(net, classes, disk, text_angle, gate=0.0):
    """Best gated read of one marker disk by ANGLE CONSENSUS. Sweeps orientations
    (detected text axis + its 180° flip, plus a blind ±45° fallback), classifies
    each plausible segmentation, and groups the pattern-valid reads by text. The
    winner is the text supported by the MOST angles (then highest weakest-char
    confidence, then longer). Returns (text, min_conf) or ('', 0.0); default
    gate=0 returns the winner + its confidence so the caller thresholds (AUTO vs
    SUGGEST).

    Why consensus: the disk angle detection is often unreliable, and a single
    off-angle segmentation can read the digits in a rotated order as a spuriously
    confident WRONG etiqueta (measured: 0011 read correctly at 4 angles ~0.85 but
    as 0110 at one angle @0.94; taking the lone max wrote the wrong cca). A real
    read is stable across several degrees; a fluke is not. DISAGREEMENT DEMOTION:
    if a different read is at least as confident as the winner, orientation is
    ambiguous — cap the confidence just below AUTO_GATE so it can only ever be a
    suggestion, never an auto-write. A wrong etiqueta poisons the cca (blank beats
    wrong), so this trades a little coverage for precision."""
    angles = list(eo._sweep_angles(text_angle)) + list(range(-45, 46, 5))
    reads = _collect_reads(net, classes, disk, angles)
    if not reads:
        return "", 0.0
    groups = {}
    for text, mn in reads:
        g = groups.setdefault(text, [0, 0.0])
        g[0] += 1
        g[1] = max(g[1], mn)
    wtext, (n, conf) = max(groups.items(), key=lambda kv: (kv[1][0], kv[1][1], len(kv[0])))
    rival = max((s[1] for t, s in groups.items() if t != wtext), default=0.0)
    if rival >= conf:
        conf = min(conf, AUTO_GATE - 1e-6)
    if conf < gate:
        return "", 0.0
    return wtext, conf


# selection gate: a candidate marker must yield a valid CNN read at least this
# confident to count as "an etiqueta" (vs an annotation / empty ring). Measured
# on the user-labelled review set: negatives (annotations/empty) score ~0 under
# the CNN, real markers ~0.69, so this cleanly separates them — where the old
# Hershey marker_score could not (it can't read handwriting).
SELECT_GATE = 0.30


def best_marker_disk(net, classes, gray, detected, contour):
    """Pick this polygon's etiqueta marker by HANDWRITING readability, not
    printed-template fit. Scores each inside marker by the CNN's best valid-
    pattern read confidence; among markers that clear SELECT_GATE, takes the
    most-interior (the parcel's own number sits deepest; annotations / empty
    rings don't produce a valid read). Falls back to the most-interior marker
    when none reads. Returns (disk, text_angle, score) or None.

    This is the handwriting-aware replacement for ``etiqueta_ocr.best_marker_disk``
    — same gate-then-interior shape, but the gate is a real read, so it rejects
    the dimension/label/empty grabs the Hershey score let through."""
    if net is None:
        return None
    cands = eo.markers_in_polygon(gray, detected, contour)
    if not cands:
        return None
    best = None  # (dist, disk, text_angle, score) among readable markers
    fallback = None
    for dist, x, y, a0, a1, ang in cands:
        disk, text_angle = eo.extract_disk(gray, x, y, a0, a1, ang)
        if fallback is None:
            fallback = (disk, text_angle, 0.0)  # most interior
        text, conf = read_disk(net, classes, disk, text_angle)
        score = conf if text else 0.0
        if score >= SELECT_GATE and (best is None or dist > best[0]):
            best = (dist, disk, text_angle, score)
    if best is not None:
        return best[1], best[2], best[3]
    return fallback


def read_polygons(net, classes, gray, polygons_px):
    """CNN read per polygon, using the same inside-only ownership rule as
    ``etiqueta_ocr.read_etiquetas`` (only a marker inside the polygon and clear of
    its boundary counts). Returns a list of (text, min_conf) — the best
    pattern-valid read and its weakest-char confidence, ungated ('' / 0.0 when
    nothing segments). The caller thresholds: AUTO_GATE writes a real value,
    SUGGEST_GATE shows a placeholder hint. A read is deduped against reads that
    already cleared SUGGEST_GATE so a low-confidence guess can't steal a label."""
    n = len(polygons_px)
    out = [("", 0.0)] * n
    if net is None or n == 0:
        return out
    contours = [np.array([[int(c), int(r)] for c, r in poly], np.int32)
                for poly in polygons_px]
    gray = eo.mask_to_polygons(gray, contours, 60)
    detected = eo.detect_markers(gray)
    if not detected:
        return out
    disk_cache = {}

    def disk_for(k):
        if k not in disk_cache:
            x, y, a0, a1, ang = detected[k]
            disk_cache[k] = eo.extract_disk(gray, x, y, a0, a1, ang)
        return disk_cache[k]

    used = set()
    for i, contour in enumerate(contours):
        inside = sorted(((cv2.pointPolygonTest(contour, (m[0], m[1]), True), k)
                         for k, m in enumerate(detected)), key=lambda v: -v[0])
        best_text, best_min = "", 0.0
        for dist, k in inside[:3]:
            if dist < 15:  # inside this polygon only, clear of the boundary
                break
            disk, text_angle = disk_for(k)
            text, mn = read_disk(net, classes, disk, text_angle)
            if text and mn > best_min:
                best_text, best_min = text, mn
        if best_text and (best_min < SUGGEST_GATE or best_text.upper() not in used):
            out[i] = (best_text, best_min)
            if best_min >= SUGGEST_GATE:
                used.add(best_text.upper())
    return out

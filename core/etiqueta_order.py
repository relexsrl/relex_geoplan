"""Spatial assignment of a typed etiqueta list to extracted parcel polygons.

Parcels in a group are numbered spatially on the plano: ribbons run along the
chain of adjacent parcels, blocks in reading order. So the typed list (sorted
naturally: 220,221,...,237 / 320E,320F,320G) is mapped onto the polygons'
spatial order instead of the meaningless extraction-index order.

Validated on ground truth: chains 100 % (the 18-parcel curved ribbon
included), regular blocks high, irregular fan layouts imperfect — the user
verifies on the map either way.
"""
import re

import numpy as np
from shapely.geometry import Polygon

_LABEL_RE = re.compile(r"^(\d+)([A-Za-z]*)$")


def natural_key(label):
    m = _LABEL_RE.match(str(label).strip())
    if m:
        return (0, int(m.group(1)), m.group(2).upper())
    return (1, 0, str(label).upper())


def _chain_order(polys):
    """If the parcels form a single path of adjacent polygons, walk it end to
    end starting at the northwest endpoint. Returns index order or None."""
    n = len(polys)
    if n < 3:
        return None
    buffered = [p.buffer(1.0) for p in polys]
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if buffered[i].intersection(buffered[j]).area > 0.5:
                adj[i].add(j)
                adj[j].add(i)
    degs = {i: len(v) for i, v in adj.items()}
    ends = [i for i, d in degs.items() if d == 1]
    if len(ends) != 2 or any(d > 2 or d == 0 for d in degs.values()):
        return None
    # start at the north-most (then west-most) end
    def nw(i):
        c = polys[i].centroid
        return (-c.y, c.x)
    start = min(ends, key=nw)
    order, cur, prev = [start], start, -1
    while len(order) < n:
        nxt = [j for j in adj[cur] if j != prev]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        order.append(cur)
    return order


def _reading_order(polys):
    """Row-major order along the group's own axes (rows top to bottom, then
    left to right)."""
    pts = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    if len(pts) == 1:
        return [0]
    mean = pts.mean(0)
    d = pts - mean
    cov = d.T @ d / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    main = evecs[:, int(np.argmax(evals))]
    perp = evecs[:, int(np.argmin(evals))]
    if main[1] > 0:
        main = -main
    if perp[1] > 0:
        perp = -perp
    proj_main = d @ main
    proj_perp = d @ perp
    if float(np.ptp(proj_perp)) < 0.35 * float(np.ptp(proj_main)):
        return list(np.argsort(-proj_main))
    typical = float(np.sqrt(np.median((d ** 2).sum(1)))) or 1.0
    rows, row_vals = [], []
    for i in np.argsort(-proj_perp):
        for row, rv in zip(rows, row_vals):
            if abs(proj_perp[i] - rv) < typical * 0.5:
                row.append(int(i))
                break
        else:
            rows.append([int(i)])
            row_vals.append(proj_perp[i])
    out = []
    for row in rows:
        out.extend(sorted(row, key=lambda i: -proj_main[i]))
    return out


def assign_etiquetas(rings, labels):
    """Map the typed labels onto the polygons by spatial order.

    rings: list of point sequences [(x, y), ...] (open or closed).
    labels: the typed etiqueta strings (any order; sorted naturally here).
    Returns a list aligned with ``rings``; "" where no label is available.
    """
    out = [""] * len(rings)
    if not rings or not labels:
        return out
    if len(rings) == 1:
        out[0] = str(labels[0]).strip()
        return out
    polys = []
    for ring in rings:
        pts = [(p[0], p[1]) for p in ring]
        if len(pts) < 3:
            return [""] * len(rings)
        polys.append(Polygon(pts))
    order = _chain_order(polys) or _reading_order(polys)
    ordered_labels = sorted((str(l).strip() for l in labels), key=natural_key)
    for pos, ring_i in enumerate(order):
        if pos < len(ordered_labels):
            out[ring_i] = ordered_labels[pos]
    return out

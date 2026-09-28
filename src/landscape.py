"""Procedural low-poly valley around the arena (replaces the stadium + city skyline).

The arena sits on a flat stone plaza; around it faceted grass hills with pine trees, then a ring of
mountains (snow on the high faces) fading into the dusk haze. One static, non-indexed mesh (every
triangle has its own vertices -> flat faces), shaded by LANDSCAPE_FRAG.

Vertex layout: pos(3) + uv(2); uv.x = per-face random (0..1, same on the 3 vertices), uv.y = material:
0 plaza stone, 1 grass, 2 rock (snow by height in the shader), 3 pine foliage, 4 trunk,
5 deciduous canopy (autumn palette by per-face random), 6 lake water, 7 boulder.
"""
import math

import numpy as np

PLAZA_R = 8500.0          # flat stone disc (arena corners are at ~6600, goals reach y=6000)
PLAZA_Z = -60.0
SEGMENTS = 80
RINGS = [0.0, PLAZA_R, 9300.0, 10200.0, 11200.0, 12400.0, 13800.0, 15400.0, 17500.0, 20000.0, 23000.0, 26500.0,
         30500.0, 35000.0, 40000.0, 46000.0]
SEGMENTS = 96
# lake: an ellipse in the valley floor on one side of the arena
LAKE_C, LAKE_R, LAKE_Z = (9800.0, 9800.0), (3600.0, 2400.0), -30.0


# The settlement (maps.build_valley): the road from the plaza along the lake shore to the village, the fields
# beyond it, the castle on the hill above it. Trees and boulders stay out of these (clear_zone).
ROAD = [(6200.0, 5800.0), (10800.0, 6200.0), (14400.0, 7900.0), (16000.0, 10600.0), (15700.0, 12600.0)]
STREET_END = (15300.0, 16400.0)
VILLAGE = (15550.0, 14200.0)
FIELDS = (20300.0, 12800.0)
CASTLE = (11600.0, 20200.0)


def _seg_dist(px, py, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def clear_zone(x, y):
    if math.hypot(x - VILLAGE[0], y - VILLAGE[1]) < 3000 or math.hypot(x - FIELDS[0], y - FIELDS[1]) < 3700:
        return True
    if math.hypot(x - CASTLE[0], y - CASTLE[1]) < 3300:
        return True
    pts = ROAD + [STREET_END]
    return any(_seg_dist(x, y, a, b) < 650 for a, b in zip(pts[:-1], pts[1:]))


def _noise(rng_seed=7, n=256):
    rng = np.random.default_rng(rng_seed)
    return rng.random((n, n))


_N = _noise()


def _vnoise(x, y):
    """Smooth value noise on a wrapped 256 grid, x/y in grid cells."""
    xi, yi = math.floor(x), math.floor(y)
    fx, fy = x - xi, y - yi
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _N[yi % 256, xi % 256]; b = _N[yi % 256, (xi + 1) % 256]
    c = _N[(yi + 1) % 256, xi % 256]; d = _N[(yi + 1) % 256, (xi + 1) % 256]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _height(x, y):
    r = math.hypot(x, y)
    if r <= PLAZA_R + 1.0:
        return PLAZA_Z
    hills = _smooth(PLAZA_R, 13000.0, r) * (120.0 + 560.0 * _vnoise(x / 3000.0 + 11.3, y / 3000.0 + 5.1))
    # rolling foothills rising toward the mountain ring (the peaks themselves are separate meshes)
    foot = _smooth(16000.0, 30000.0, r) * (500.0 + 900.0 * _vnoise(x / 4500.0 + 40.0, y / 4500.0 + 17.0))
    h = PLAZA_Z + hills + foot
    # lake basin
    lx, ly = (x - LAKE_C[0]) / LAKE_R[0], (y - LAKE_C[1]) / LAKE_R[1]
    e = math.sqrt(lx * lx + ly * ly)
    if e < 1.35:
        h = min(h, LAKE_Z - 90.0 + 160.0 * _smooth(0.85, 1.35, e) + (h - LAKE_Z) * _smooth(1.0, 1.35, e))
    return h


def _cone(tris, cx, cy, z0, z1, r, sides, mat, rnd, rot):
    for k in range(sides):
        a0 = rot + 2 * math.pi * k / sides
        a1 = rot + 2 * math.pi * (k + 1) / sides
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0), z0)
        p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1), z0)
        f = rnd.random()
        tris.extend([(*p0, f, mat), (*p1, f, mat), (cx, cy, z1, f, mat)])


_ICO_V = None


def _ico():
    global _ICO_V
    if _ICO_V is None:
        t = (1 + 5 ** 0.5) / 2
        v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                      (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], "f8")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
        _ICO_V = (v, f)
    return _ICO_V


def _blob(tris, cx, cy, cz, r, rnd, mat=5, squash=0.85):
    v, f = _ico()
    jit = v * (1.0 + rnd.uniform(-0.15, 0.15, (len(v), 1)))
    col = rnd.random()                       # one palette pick per tree (same on all its faces)
    for a, b, c in f:
        pts = [(cx + jit[i, 0] * r, cy + jit[i, 1] * r, cz + jit[i, 2] * r * squash) for i in (a, b, c)]
        tris.extend([(*pts[0], col, mat), (*pts[1], col, mat), (*pts[2], col, mat)])


def _rock(tris, cx, cy, cz, r, rnd):
    v, f = _ico()
    jit = v * (1.0 + rnd.uniform(-0.3, 0.3, (len(v), 1)))
    for a, b, c in f:
        g = rnd.random()
        pts = [(cx + jit[i, 0] * r, cy + jit[i, 1] * r, cz + jit[i, 2] * r * 0.6) for i in (a, b, c)]
        tris.extend([(*pts[0], g, 7), (*pts[1], g, 7), (*pts[2], g, 7)])


def _peak(tris, cx, cy, base_z, radius, height, rnd, sides=None):
    """One mountain: 16-22 sides, 7 rings, ridges from a smooth angular noise (spurs and gullies) and a jittered
    summit. uv.x on rock faces carries the height fraction (0 foot .. 1 summit) for the snow line."""
    sides = sides or int(rnd.integers(16, 23))
    rot = rnd.uniform(0, 2 * math.pi)
    waves = [(rnd.integers(2, 6), rnd.uniform(0, 6.3), rnd.uniform(0.08, 0.18)) for _ in range(3)]

    def ridge(a):
        return 1.0 + sum(amp * math.sin(k * a + ph) for k, ph, amp in waves)
    rings = [(1.0, 0.0), (0.86, 0.14), (0.70, 0.30), (0.54, 0.47), (0.39, 0.63), (0.25, 0.78), (0.12, 0.91)]
    pts = []
    for k, (rf, hf) in enumerate(rings):
        row = []
        for i in range(sides):
            a = rot + 2 * math.pi * (i + 0.5 * (k % 2)) / sides + rnd.uniform(-0.06, 0.06)
            rr = radius * rf * ridge(a) * rnd.uniform(0.94, 1.06)
            row.append((cx + rr * math.cos(a), cy + rr * math.sin(a),
                        base_z + height * hf * (0.92 + 0.16 * ridge(a + 1.0) * 0.5) * rnd.uniform(0.97, 1.03)))
        pts.append(row)
    top = (cx + rnd.uniform(-0.05, 0.05) * radius, cy + rnd.uniform(-0.05, 0.05) * radius, base_z + height)

    def f(a, b, c):
        hf = (a[2] + b[2] + c[2]) / 3.0 - base_z
        t = min(max(hf / height, 0.0), 1.0)
        tris.extend([(*a, t, 2), (*b, t, 2), (*c, t, 2)])

    for k in range(len(rings) - 1):
        lo, hi = pts[k], pts[k + 1]
        for i in range(sides):
            j = (i + 1) % sides
            f(lo[i], lo[j], hi[i]); f(lo[j], hi[j], hi[i])
    hi = pts[-1]
    for i in range(sides):
        f(hi[i], hi[(i + 1) % sides], top)


def build(seed=3):
    rnd = np.random.default_rng(seed)
    # vertex grid (ring, segment) with jitter for an irregular low-poly look
    grid = []
    for ri, r in enumerate(RINGS):
        row = []
        for si in range(SEGMENTS):
            th = 2 * math.pi * (si + (0.5 if ri % 2 else 0.0)) / SEGMENTS
            rr = r
            if ri >= 2:
                rr += rnd.uniform(-0.18, 0.18) * (RINGS[min(ri + 1, len(RINGS) - 1)] - RINGS[ri - 1]) * 0.5
                th += rnd.uniform(-0.25, 0.25) * 2 * math.pi / SEGMENTS
            x, y = rr * math.cos(th), rr * math.sin(th)
            row.append((x, y, _height(x, y)))
        grid.append(row)

    tris = []

    def face(a, b, c):
        zc = (a[2] + b[2] + c[2]) / 3.0
        rc = math.hypot((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
        if rc < PLAZA_R + 10.0:
            mat = 0
        else:
            # slope from the face normal: steep or high faces are rock
            u = np.subtract(b, a); v = np.subtract(c, a)
            n = np.cross(u, v); n = n / (np.linalg.norm(n) + 1e-9)
            mat = 2 if (zc > 1100.0 or abs(n[2]) < 0.78) else 1
        f = rnd.random()
        tris.extend([(*a, f, mat), (*b, f, mat), (*c, f, mat)])

    centre = (0.0, 0.0, PLAZA_Z)
    for si in range(SEGMENTS):
        face(centre, grid[1][si], grid[1][(si + 1) % SEGMENTS])
    for ri in range(1, len(RINGS) - 1):
        a_row, b_row = grid[ri], grid[ri + 1]
        for si in range(SEGMENTS):
            s1 = (si + 1) % SEGMENTS
            face(a_row[si], b_row[si], b_row[s1])
            face(a_row[si], b_row[s1], a_row[s1])

    # mountain ring: a near ring of mid peaks + a taller far range, spaced around the valley
    for ring_r, n, (hmin, hmax), (rmin, rmax) in ((27000.0, 22, (3500.0, 6500.0), (4500.0, 7000.0)),
                                                  (38000.0, 26, (6500.0, 11000.0), (7000.0, 11000.0))):
        for i in range(n):
            th = 2 * math.pi * (i + rnd.uniform(-0.3, 0.3)) / n
            rr = ring_r * rnd.uniform(0.9, 1.1)
            x, y = rr * math.cos(th), rr * math.sin(th)
            _peak(tris, x, y, _height(x, y) - 200.0, rnd.uniform(rmin, rmax), rnd.uniform(hmin, hmax), rnd)

    # lake surface (a fan over the ellipse, slightly above the basin floor)
    k = 40
    for i in range(k):
        a0, a1 = 2 * math.pi * i / k, 2 * math.pi * (i + 1) / k
        p0 = (LAKE_C[0] + LAKE_R[0] * 1.08 * math.cos(a0), LAKE_C[1] + LAKE_R[1] * 1.08 * math.sin(a0), LAKE_Z)
        p1 = (LAKE_C[0] + LAKE_R[0] * 1.08 * math.cos(a1), LAKE_C[1] + LAKE_R[1] * 1.08 * math.sin(a1), LAKE_Z)
        tris.extend([(LAKE_C[0], LAKE_C[1], LAKE_Z, 0.5, 6), (*p0, 0.5, 6), (*p1, 0.5, 6)])

    def in_lake(x, y, pad=1.25):
        lx, ly = (x - LAKE_C[0]) / LAKE_R[0], (y - LAKE_C[1]) / LAKE_R[1]
        return lx * lx + ly * ly < pad * pad

    # boulders
    for _ in range(70):
        r = rnd.uniform(PLAZA_R + 400.0, 24000.0); th = rnd.uniform(0, 2 * math.pi)
        x, y = r * math.cos(th), r * math.sin(th)
        if in_lake(x, y, 1.05) or clear_zone(x, y):
            continue
        z = _height(x, y)
        sz = rnd.uniform(90.0, 320.0)
        _rock(tris, x, y, z - sz * 0.25, sz, rnd)

    # trees on the low grass ring: pines + round-canopy deciduous (autumn palette)
    placed = 0
    while placed < 420:
        r = rnd.uniform(PLAZA_R + 700.0, 21000.0)
        th = rnd.uniform(0, 2 * math.pi)
        x, y = r * math.cos(th), r * math.sin(th)
        z = _height(x, y)
        if z > 700.0 or in_lake(x, y) or clear_zone(x, y):
            continue
        if rnd.random() < 0.35:                                                   # deciduous
            h = rnd.uniform(700.0, 1150.0) * (1.0 + 0.5 * (r - PLAZA_R) / 12000.0)
            _cone(tris, x, y, z - 30.0, z + h * 0.55, h * 0.06, 5, 4, rnd, 0.0)    # trunk
            _blob(tris, x, y, z + h * 0.62, h * 0.36, rnd)
            placed += 1
            continue
        h = rnd.uniform(900.0, 1650.0) * (1.0 + 0.6 * (r - PLAZA_R) / 12000.0)
        w = h * rnd.uniform(0.26, 0.34)
        rot = rnd.uniform(0, math.pi)
        z0 = z - 30.0
        _cone(tris, x, y, z0, z0 + h * 0.22, w * 0.10, 5, 4, rnd, rot)                 # trunk
        for k, (b, t, s) in enumerate(((0.15, 0.62, 1.0), (0.40, 0.84, 0.78), (0.62, 1.0, 0.55))):
            _cone(tris, x, y, z0 + h * b, z0 + h * t, w * s, 6, 3, rnd, rot + k * 0.4)
        placed += 1
    return np.asarray(tris, "f4")


MESH_VERSION = 6


def load_or_build(cache_path):
    """build() takes ~0.5 s; cache the result (rebuilt when MESH_VERSION changes)."""
    try:
        d = np.load(cache_path, allow_pickle=False)
        if d.shape[1] == 6 and int(d[0, 5]) == MESH_VERSION:
            return np.ascontiguousarray(d[:, :5])
    except (OSError, ValueError, IndexError):
        pass
    m = build()
    try:
        np.save(cache_path, np.concatenate([m, np.full((len(m), 1), MESH_VERSION, "f4")], axis=1))
    except OSError:
        pass
    return m

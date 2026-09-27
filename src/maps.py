"""Maps: the scenery around the arena, its sky and light, the field style, and the crowd.

  valley  the original low-poly evening valley (landscape.py, unchanged)
  temple  "Forbidden Temple"-style: pink dusk, karst peaks, pagodas, a paifang gate, cherry trees, lanterns
  paris   "Parc de Paris"-style: violet dusk, the Eiffel Tower down the Champ de Mars, Haussmann blocks, the Seine,
          a two-tier stadium stand under floodlights
  space   the arena on an orbital platform: stars, a ringed gas giant, moons, the planet below, asteroids, a station

Arrow keys switch map live (main.py MAP_KEYS). Each non-valley scene is one static mesh (pos, colour, emission,
kind -> SCENE_FRAG) built here with numpy and cached in data/scene_<map>_cache.npy, plus an instanced egg crowd
(pos, colour, phase, scale -> CROWD_VERT) that jumps on its seats after goals and saves.
"""
import math
import os

import numpy as np

ORDER = ["valley", "temple", "paris", "space"]
MAP_ID = {k: i for i, k in enumerate(ORDER)}
TITLE = {"valley": "Evening Valley", "temple": "Forbidden Temple", "paris": "Parc de Paris", "space": "Orbit"}
SCENE_VERSION = 6


def _n(v):
    v = np.asarray(v, "f8")
    return tuple(float(x) for x in v / np.linalg.norm(v))


# light + sky + field per map. sun_dir/sun_col/zen/mid/hor/glow/ground -> COMMON's uniforms; cloudA/B -> SKY_FRAG;
# haze (near, far, strength) -> SCENE_FRAG / CROWD_FRAG; grass -> the turf's base colour.
THEMES = {
    "valley": dict(sun_dir=_n((-0.45, 0.75, 0.30)), sun_col=(1.00, 0.88, 0.72), zen=(0.035, 0.05, 0.13),
                   mid=(0.22, 0.15, 0.30), hor=(0.95, 0.46, 0.20), glow=(1.0, 0.55, 0.25), ground=(0.04, 0.035, 0.05),
                   amb=(1.0, 1.0, 1.0), grass=(0.080, 0.265, 0.050), cloudA=(0.95, 0.55, 0.35, 0.52),
                   cloudB=(0.35, 0.25, 0.40, 0.55), haze=(9000.0, 48000.0, 0.85), glass=1.0),
    "temple": dict(sun_dir=_n((0.62, 0.74, 0.14)), sun_col=(1.00, 0.72, 0.80), zen=(0.07, 0.035, 0.16),
                   mid=(0.44, 0.15, 0.38), hor=(1.00, 0.48, 0.55), glow=(1.0, 0.45, 0.58), ground=(0.05, 0.03, 0.05),
                   amb=(1.02, 0.88, 1.02), grass=(0.050, 0.150, 0.130), cloudA=(1.0, 0.56, 0.66, 0.44),
                   cloudB=(0.42, 0.18, 0.40, 0.72), haze=(9000.0, 60000.0, 0.72), glass=1.0),
    "paris": dict(sun_dir=_n((-0.55, -0.62, 0.16)), sun_col=(0.92, 0.80, 0.96), zen=(0.05, 0.04, 0.15),
                  mid=(0.28, 0.18, 0.42), hor=(0.82, 0.52, 0.72), glow=(0.95, 0.55, 0.75), ground=(0.04, 0.03, 0.06),
                  amb=(0.95, 0.90, 1.08), grass=(0.075, 0.25, 0.060), cloudA=(0.85, 0.60, 0.86, 0.40),
                  cloudB=(0.26, 0.18, 0.40, 0.78), haze=(9000.0, 55000.0, 0.72), glass=1.0),
    "space": dict(sun_dir=_n((0.35, -0.55, 0.55)), sun_col=(1.05, 1.00, 0.95), zen=(0.004, 0.004, 0.012),
                  mid=(0.010, 0.012, 0.030), hor=(0.030, 0.040, 0.090), glow=(0.50, 0.55, 0.80),
                  ground=(0.02, 0.05, 0.12), amb=(0.70, 0.78, 1.00), grass=(0.055, 0.060, 0.072),
                  cloudA=(0.0, 0.0, 0.0, 2.0), cloudB=(0.0, 0.0, 0.0, 0.0), haze=(30000.0, 90000.0, 0.25), glass=0.45),
}

EGG_COLS = np.array([(0.95, 0.55, 0.20), (0.55, 0.35, 0.75), (0.25, 0.65, 0.62), (0.92, 0.50, 0.62),
                     (0.45, 0.70, 0.30), (0.95, 0.82, 0.35), (0.35, 0.50, 0.85), (0.85, 0.25, 0.22),
                     (0.88, 0.82, 0.70), (0.55, 0.55, 0.58), (0.30, 0.30, 0.45), (0.95, 0.68, 0.50)], "f4")


# ------------------------------------------------------------------------------------------------ geometry
class G:
    """Accumulates triangles as rows of pos(3) colour(3) emission kind."""

    def __init__(self):
        self.parts = []

    def tris(self, P, col, emis=0.0, kind=0):
        P = np.asarray(P, "f4").reshape(-1, 3, 3)
        n = len(P)
        if n == 0:
            return
        a = np.empty((n, 3, 8), "f4")
        a[:, :, :3] = P
        c = np.asarray(col, "f4")
        a[:, :, 3:6] = c if c.ndim == 1 else c.reshape(n, 1, 3)
        e = np.asarray(emis, "f4")
        a[:, :, 6] = e if e.ndim == 0 else e.reshape(n, 1)
        a[:, :, 7] = kind
        self.parts.append(a.reshape(-1, 8))

    def quads(self, A, B, C, D, col, emis=0.0, kind=0):
        A, B, C, D = (np.asarray(x, "f4").reshape(-1, 3) for x in (A, B, C, D))
        P = np.concatenate([np.stack([A, B, C], 1), np.stack([A, C, D], 1)])
        c = np.asarray(col, "f4")
        if c.ndim == 2:
            c = np.concatenate([c, c])
        e = np.asarray(emis, "f4")
        if e.ndim == 1:
            e = np.concatenate([e, e])
        self.tris(P, c, e, kind)

    def array(self):
        return np.concatenate(self.parts).astype("f4") if self.parts else np.zeros((0, 8), "f4")


def _xf(pts, c, yaw):
    """local (x, y, z) points -> world, rotated by yaw about z, translated by c."""
    pts = np.asarray(pts, "f8")
    ca, sa = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(pts)
    out[..., 0] = c[0] + pts[..., 0] * ca - pts[..., 1] * sa
    out[..., 1] = c[1] + pts[..., 0] * sa + pts[..., 1] * ca
    out[..., 2] = c[2] + pts[..., 2]
    return out


def box(g, c, h, yaw=0.0, col=(0.5, 0.5, 0.5), emis=0.0, kind=0, top=None, top_kind=0, bottom=False):
    hx, hy, hz = h
    v = _xf([(sx * hx, sy * hy, sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], c, yaw)
    sides = [(0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    for a, b, cc, d in sides:
        g.quads(v[a], v[b], v[cc], v[d], col, emis, kind)
    g.quads(v[4], v[5], v[7], v[6], col if top is None else top, emis if top is None else 0.0,
            kind if top is None else top_kind)
    if bottom:
        g.quads(v[0], v[1], v[3], v[2], col, emis, kind)


def frustum(g, c, r0, r1, z0, z1, n, col, emis=0.0, kind=0, cap=True, rot=0.0, sy=1.0):
    a = rot + np.arange(n + 1) * 2 * math.pi / n
    ca, sa = np.cos(a), np.sin(a)
    b = np.stack([c[0] + r0 * ca, c[1] + r0 * sa * sy, np.full(n + 1, c[2] + z0)], 1)
    t = np.stack([c[0] + r1 * ca, c[1] + r1 * sa * sy, np.full(n + 1, c[2] + z1)], 1)
    g.quads(b[:-1], b[1:], t[1:], t[:-1], col, emis, kind)
    if cap and r1 > 0:
        ctr = np.tile([c[0], c[1], c[2] + z1], (n, 1))
        g.tris(np.stack([t[:-1], t[1:], ctr], 1), col, emis, kind)


def lathe(g, c, prof, n, colfn, kind=0, emis=0.0, rng=None, jitter=0.0, rot=0.0):
    """Surface of revolution through profile [(r, z)...]; colfn(normal_z, z_frac, rnd) -> colour per face."""
    prof = np.asarray(prof, "f8")
    m = len(prof)
    a = rot + np.arange(n) * 2 * math.pi / n
    P = np.zeros((m, n, 3))
    for i, (r, z) in enumerate(prof):
        rr = r * (1.0 + (rng.uniform(-jitter, jitter, n) if (rng is not None and jitter > 0 and r > 0) else 0.0))
        P[i, :, 0] = c[0] + rr * np.cos(a)
        P[i, :, 1] = c[1] + rr * np.sin(a)
        P[i, :, 2] = c[2] + z
    ztop = max(prof[-1][1], 1.0)
    for i in range(m - 1):
        for k in range(n):
            k1 = (k + 1) % n
            for tri in ((P[i, k], P[i, k1], P[i + 1, k1]), (P[i, k], P[i + 1, k1], P[i + 1, k])):
                t = np.array(tri)
                nr = np.cross(t[1] - t[0], t[2] - t[0])
                ln = np.linalg.norm(nr)
                if ln < 1e-6:
                    continue
                nz = abs(nr[2] / ln)
                zf = (t[:, 2].mean() - c[2]) / ztop
                g.tris(t[None], colfn(nz, zf, rng.random() if rng is not None else 0.5), emis, kind)


_ICO = None


def _ico_base(sub):
    global _ICO
    if _ICO is None:
        _ICO = {}
    if sub in _ICO:
        return _ICO[sub]
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, "f8") / np.linalg.norm(v) for v in V]
    for _ in range(sub):
        cache, F2 = {}, []

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                m = V[i] + V[j]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    _ICO[sub] = (np.array(V), np.array(F))
    return _ICO[sub]


def ico(g, c, r, col, rng, sub=1, jit=0.18, squash=1.0, emis=0.0, kind=0, colvar=0.0):
    V, F = _ico_base(sub)
    s = 1.0 + rng.uniform(-jit, jit, len(V)) if jit > 0 else np.ones(len(V))
    P = V * s[:, None] * r
    P[:, 2] *= squash
    P += np.asarray(c, "f8")
    T = P[F]
    col = np.asarray(col, "f4")
    if colvar > 0:
        col = np.clip(col[None, :] * (1.0 + rng.uniform(-colvar, colvar, (len(F), 1))), 0, 1)
    g.tris(T, col, emis, kind)


def roof(g, c, hx, hy, z0, rise, over, curl, yaw, col, soffit=None, top_frac=0.14):
    """Chinese flared roof: eave ring with upturned corners, a concave middle ring, a ridge on top."""
    ex, ey = hx + over, hy + over
    ridge = max(hx - hy, 0.0)                                    # long halls get a ridge along x
    r0 = [(ex, ey, curl), (0, ey, 0), (-ex, ey, curl), (-ex, 0, 0), (-ex, -ey, curl), (0, -ey, 0), (ex, -ey, curl), (ex, 0, 0)]
    r1 = [(x * 0.62 + (ridge * 0.35 * np.sign(x) if x else 0), y * 0.58, rise * 0.42 + (curl * 0.25 if x and y else 0))
          for x, y, _ in r0]
    tx, ty = ridge + hx * top_frac, hy * top_frac
    r2 = [(np.sign(x) * tx if x else 0, np.sign(y) * ty if y else 0, rise) for x, y, _ in r0]
    R0, R1, R2 = (_xf(r, (c[0], c[1], z0), yaw) for r in (r0, r1, r2))
    for A, B in ((R0, R1), (R1, R2)):
        for k in range(8):
            k1 = (k + 1) % 8
            g.quads(A[k], A[k1], B[k1], B[k], col)
    top = _xf([(0, 0, rise)], (c[0], c[1], z0), yaw)[0]
    for k in range(8):
        g.tris(np.array([[R2[k], R2[(k + 1) % 8], top]]), col)
    if soffit is not None:                                     # underside of the eaves
        w = _xf([(hx, hy, -2), (0, hy, -2), (-hx, hy, -2), (-hx, 0, -2), (-hx, -hy, -2), (0, -hy, -2), (hx, -hy, -2), (hx, 0, -2)],
                (c[0], c[1], z0), yaw)
        for k in range(8):
            k1 = (k + 1) % 8
            g.quads(R0[k], R0[k1], w[k1], w[k], soffit)


def terrain(g, radii, segs, hfun, cfun, rng):
    grid = []
    for ri, r in enumerate(radii):
        row = []
        for si in range(segs):
            th = 2 * math.pi * (si + (0.5 if ri % 2 else 0.0)) / segs
            rr = r
            if 1 < ri < len(radii) - 1:
                rr += rng.uniform(-0.15, 0.15) * (radii[ri + 1] - radii[ri - 1]) * 0.5
                th += rng.uniform(-0.2, 0.2) * 2 * math.pi / segs
            x, y = rr * math.cos(th), rr * math.sin(th)
            row.append((x, y, hfun(x, y)))
        grid.append(row)

    def face(a, b, c):
        n = np.cross(np.subtract(b, a), np.subtract(c, a))
        nz = abs(n[2]) / (np.linalg.norm(n) + 1e-9)
        cx, cy, cz = (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3
        col, kind = cfun(cx, cy, cz, nz, rng.random())
        g.tris(np.array([[a, b, c]]), col, 0.0, kind)
    ctr = (0.0, 0.0, hfun(0.0, 0.0))
    for si in range(segs):
        face(ctr, grid[0][si], grid[0][(si + 1) % segs]) if radii[0] > 0 else None
    for ri in range(len(radii) - 1):
        a_row, b_row = grid[ri], grid[ri + 1]
        for si in range(segs):
            s1 = (si + 1) % segs
            face(a_row[si], b_row[si], b_row[s1])
            face(a_row[si], b_row[s1], a_row[s1])


def _smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _vn(x, y, seed=0.0):
    return 0.5 + 0.25 * math.sin(x * 1.7 + seed) * math.cos(y * 1.3 - seed) + 0.25 * math.sin(x * 0.63 - y * 0.91 + seed * 2)


# ------------------------------------------------------------------------------------------------ crowd
def egg_mesh(seg=7, rings=4):
    """Unit-height egg (wider low, narrower top), pos + smooth normal per vertex, non-indexed triangles."""
    ts = np.linspace(0.0, math.pi, rings + 1)

    def prof(t):
        z = (1.0 - math.cos(t)) * 0.5
        r = 0.36 * math.sin(t) * (1.0 - 0.22 * (z - 0.45))
        return r, z
    pts, nrm = [], []
    for i in range(rings + 1):
        r, z = prof(ts[i])
        e = 1e-3
        r1, z1 = prof(min(ts[i] + e, math.pi)); r0, z0 = prof(max(ts[i] - e, 0.0))
        dr, dz = r1 - r0, z1 - z0                               # tangent; outward normal = (dz, -dr)
        row, nrow = [], []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            row.append((r * math.cos(a), r * math.sin(a), z))
            nn = np.array([dz * math.cos(a), dz * math.sin(a), -dr])
            if i == 0:
                nn = np.array([0.0, 0.0, -1.0])
            elif i == rings:
                nn = np.array([0.0, 0.0, 1.0])
            nrow.append(nn / (np.linalg.norm(nn) + 1e-9))
        pts.append(row); nrm.append(nrow)
    out = []
    for i in range(rings):
        for k in range(seg):
            k1 = (k + 1) % seg
            quad = [(i, k), (i, k1), (i + 1, k1), (i + 1, k)]
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                for (ii, kk) in tri:
                    out.append((*pts[ii][kk], *nrm[ii][kk]))
    return np.asarray(out, "f4")


def stand(g, eggs, origin, u, v, rows, length, depth, rise, pitch, rng, tread, riser, rail=None, empty=0.06,
          z_base=-60.0, egg_h=74.0):
    """Stepped grandstand: `rows` steps going away along v (horizontal unit), each `length` long along u, `rise`
    higher than the previous; an egg on every seat (a few empty). Appends (pos, col, phase, scale) to eggs."""
    ox, oy, oz = origin
    u = np.asarray(u, "f8"); v = np.asarray(v, "f8")
    yaw = math.atan2(v[1], v[0])
    for k in range(rows):
        top = oz + (k + 1) * rise
        c = np.array([ox, oy, 0.0]) + v * (k + 0.5) * depth
        hz = (top - z_base) * 0.5
        box(g, (c[0], c[1], z_base + hz), (depth * 0.5, length * 0.5, hz), yaw, riser, top=tread)
        ns = int(length // pitch)
        s = (np.arange(ns) - (ns - 1) * 0.5) * pitch
        keep = rng.random(ns) > empty
        s = s[keep]
        p = np.array([ox, oy, top])[None, :] + (v * (k * depth + depth * 0.42))[None, :] + u[None, :] * s[:, None]
        p[:, :2] += rng.uniform(-6, 6, (len(s), 2))
        cols = EGG_COLS[rng.integers(0, len(EGG_COLS), len(s))] * rng.uniform(0.85, 1.1, (len(s), 1))
        ph = rng.random(len(s))
        sc = egg_h * rng.uniform(0.9, 1.1, len(s))
        eggs.append(np.concatenate([p, np.clip(cols, 0, 1), ph[:, None], sc[:, None]], 1))
    if rail is not None:                                        # front railing
        c = np.array([ox, oy, 0.0]) - v * 20.0
        box(g, (c[0], c[1], oz + 60.0), (12.0, length * 0.5, 60.0), yaw, rail)


# ------------------------------------------------------------------------------------------------ temple
def _pagoda(g, x, y, z, tiers, w, th, rng, yaw=0.0):
    wall, roofc, soff, gold = (0.50, 0.09, 0.07), (0.10, 0.13, 0.16), (0.30, 0.08, 0.06), (0.95, 0.70, 0.30)
    box(g, (x, y, z + 120), (w * 0.75, w * 0.75, 120), yaw, (0.30, 0.28, 0.27))            # terrace
    z += 240
    for t in range(tiers):
        box(g, (x, y, z + th * 0.5), (w * 0.5, w * 0.5, th * 0.5), yaw, wall)
        # glowing paper windows on every side
        for side in range(4):
            a = yaw + side * math.pi / 2
            off = w * 0.5 + 3
            cx, cy = x + off * math.cos(a), y + off * math.sin(a)
            box(g, (cx, cy, z + th * 0.5), (2, w * 0.30, th * 0.26), a, (1.0, 0.55, 0.25), emis=1.3, kind=2)
        z += th
        roof(g, (x, y), w * 0.5, w * 0.5, z, th * 0.55, w * 0.30, th * 0.38, yaw, roofc, soffit=soff)
        box(g, (x, y, z + 6), (w * 0.52, w * 0.52, 6), yaw, gold, emis=0.25)             # gilded beam
        w *= 0.80
        z += th * 0.35
    frustum(g, (x, y, z), w * 0.12, w * 0.02, 0, th * 1.6, 6, gold, emis=0.35)
    for k in range(3):
        ico(g, (x, y, z + th * (0.4 + 0.35 * k)), w * 0.10, gold, rng, sub=0, jit=0.0, emis=0.4)


def _paifang(g, x, y, yaw, W=5400.0, H=3900.0):
    red, blue, gold, stone, roofc = (0.48, 0.07, 0.06), (0.10, 0.22, 0.42), (0.95, 0.72, 0.30), (0.34, 0.32, 0.31), (0.08, 0.11, 0.16)
    xs = (-W / 2, -W / 6, W / 6, W / 2)
    for i, px in enumerate(xs):
        h = H if 0 < i < 3 else H * 0.72
        p = _xf([(px, 0, 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], -60 + h / 2), (130, 130, h / 2 + 60), yaw, red)
        box(g, (p[0], p[1], 90), (230, 230, 150), yaw, stone)                            # plinth
    for (a, b, h) in ((xs[1], xs[2], H), (xs[0], xs[1], H * 0.72), (xs[2], xs[3], H * 0.72)):
        cx = (a + b) / 2
        p = _xf([(cx, 0, 0)], (x, y, 0), yaw)[0]
        half = (b - a) / 2 + 160
        box(g, (p[0], p[1], h - 380), (half, 90, 60), yaw, blue)                         # lintels
        box(g, (p[0], p[1], h - 150), (half, 100, 70), yaw, blue)
        box(g, (p[0], p[1], h - 265), (half * 0.9, 70, 45), yaw, gold, emis=0.2)
        roof(g, (p[0], p[1]), half, 260, h - 70, 520, 280, 230, yaw, roofc, soffit=(0.12, 0.10, 0.12))
    p = _xf([(0, 0, 0)], (x, y, 0), yaw)[0]
    box(g, (p[0], p[1], H - 700), (520, 40, 170), yaw, gold, emis=0.9, kind=2)            # the name plaque


def _hall(g, x, y, W, D, H, rng, yaw=0.0):
    stone, red, roofc, paper = (0.33, 0.31, 0.30), (0.50, 0.09, 0.07), (0.09, 0.12, 0.16), (1.0, 0.58, 0.28)
    box(g, (x, y, 60), (W / 2 + 700, D / 2 + 700, 120), yaw, stone)
    for k in range(4):                                                                    # front steps
        p = _xf([(0, D / 2 + 700 + 150 + k * 150, 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], -60 + (4 - k) * 30), (1600, 75, (4 - k) * 30), yaw, stone)
    box(g, (x, y, 180 + H / 2), (W / 2, D / 2, H / 2), yaw, red)
    for side in (1, -1):
        p = _xf([(0, side * (D / 2 + 3), 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], 180 + H * 0.45), (W / 2 - 250, 2, H * 0.3), yaw, paper, emis=1.0, kind=2)
    ncol = int(W // 700) + 1
    for k in range(ncol):
        px = -W / 2 + k * W / (ncol - 1)
        p = _xf([(px, D / 2 + 260, 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], 180 + H / 2), (70, 70, H / 2), yaw, red)
    roof(g, (x, y), W / 2 + 260, D / 2 + 260, 180 + H, 650, 520, 300, yaw, roofc, soffit=(0.25, 0.07, 0.05))
    box(g, (x, y, 180 + H + 650 + 180), (W / 2 * 0.62, D / 2 * 0.5, 180), yaw, red)
    roof(g, (x, y), W / 2 * 0.62, D / 2 * 0.5, 180 + H + 650 + 360, 480, 380, 240, yaw, roofc, soffit=(0.25, 0.07, 0.05))


def _cherry(g, x, y, z, s, rng):
    frustum(g, (x, y, z), s * 0.06, s * 0.035, 0, s * 0.55, 5, (0.20, 0.11, 0.09))
    pinks = [(0.96, 0.58, 0.72), (0.90, 0.46, 0.64), (1.0, 0.72, 0.84), (0.85, 0.40, 0.58)]
    for _ in range(rng.integers(5, 8)):
        a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0.05, 0.30) * s
        ico(g, (x + d * math.cos(a), y + d * math.sin(a), z + s * rng.uniform(0.55, 0.78)), s * rng.uniform(0.18, 0.28),
            pinks[rng.integers(0, 4)], rng, sub=1, jit=0.22, squash=0.8, kind=6, colvar=0.08)


def _karst(g, x, y, z, r, h, rng):
    prof = [(r, -300), (r * 1.06, h * 0.12), (r * 0.98, h * 0.35), (r * 0.88, h * 0.58), (r * 0.74, h * 0.78),
            (r * 0.52, h * 0.92), (r * 0.18, h)]

    def col(nz, zf, q):
        green = nz > 0.42 or (zf > 0.8 and q < 0.7) or (q < 0.25)
        if green:
            return (0.13 + 0.05 * q, 0.25 + 0.06 * q, 0.15)
        return (0.38 + 0.08 * q, 0.37 + 0.06 * q, 0.40 + 0.06 * q)
    lathe(g, (x, y, z), prof, 9, col, rng=rng, jitter=0.14, rot=rng.uniform(0, 1))


def build_temple(seed=11):
    rng = np.random.default_rng(seed)
    g, eggs = G(), []

    def hfun(x, y):
        r = math.hypot(x, y)
        if r < 16000:
            return -60.0
        return -60.0 + _smooth(16000, 26000, r) * (300 + 700 * _vn(x / 5000, y / 5000, 1.0))

    def cfun(x, y, z, nz, q):
        r = math.hypot(x, y)
        if r < 7600:                                          # stone plaza around the arena
            return (0.19 + 0.04 * q, 0.19 + 0.04 * q, 0.20 + 0.04 * q), 0
        if r < 8200:
            return (0.30, 0.27, 0.26), 0
        return ((0.09 + 0.04 * q, 0.19 + 0.05 * q, 0.13 + 0.03 * q) if nz > 0.8 else (0.30, 0.30, 0.31)), 0
    terrain(g, [7600, 8200, 9500, 11500, 14000, 16500, 19000, 22000, 26000, 31000, 38000, 48000], 72, hfun, cfun, rng)
    # ---- west stands (-x) with the crowd, a roofed gallery with lanterns above them, pagodas behind ----
    rows, depth, rise = 24, 115.0, 80.0
    stand(g, eggs, (-5300.0, 0.0, -60.0 + 60.0), (0, 1, 0), (-1, 0, 0), rows, 10800, depth, rise, 70.0, rng,
          tread=(0.27, 0.25, 0.25), riser=(0.42, 0.08, 0.06), rail=(0.60, 0.10, 0.07))
    top = -60 + 60 + rows * rise
    xb = -5300 - rows * depth
    box(g, (xb - 250, 0, (top - 60) / 2 + 400), (250, 5600, (top + 60) / 2 + 400), 0.0, (0.40, 0.08, 0.06))  # back wall
    for k in range(13):
        yy = -5400 + k * 900
        box(g, (xb + 300, yy, top + 450), (60, 60, 450), 0.0, (0.50, 0.09, 0.07))
    roof(g, (xb + 100, 0), 520, 5500, top + 900, 520, 420, 260, 0.0, (0.09, 0.12, 0.16), soffit=(0.30, 0.07, 0.05))
    for k in range(25):                                                                    # lantern row
        yy = -5300 + k * 440
        ico(g, (xb + 560, yy, top + 700), 55, (1.0, 0.28, 0.10), rng, sub=0, jit=0.0, squash=1.3, emis=1.6, kind=2)
    _pagoda(g, -11400, -3600, -60, 5, 2300, 900, rng)
    _pagoda(g, -12300, 4300, -60, 7, 2100, 820, rng, yaw=0.3)
    _pagoda(g, -9000, 11500, -60, 3, 1800, 800, rng, yaw=-0.4)
    # ---- temple hall behind the blue goal, gate behind the orange goal ----
    _hall(g, 0, -10300, 7200, 2400, 1400, rng, yaw=0.0)
    _paifang(g, 2400, 9800, math.pi * 0.06)
    # stone path from the gate
    for k in range(10):
        box(g, (2400 + k * 40, 11000 + k * 700, -55), (900, 300, 8), 0.0, (0.33, 0.31, 0.30))
    # ---- garden (+x): pond with an arched bridge, stone lanterns, cherry trees ----
    pc, pr = (9800.0, -1500.0), (1700.0, 3800.0)
    k = 36
    for i in range(k):
        a0, a1 = 2 * math.pi * i / k, 2 * math.pi * (i + 1) / k
        p0 = (pc[0] + pr[0] * math.cos(a0), pc[1] + pr[1] * math.sin(a0), -70)
        p1 = (pc[0] + pr[0] * math.cos(a1), pc[1] + pr[1] * math.sin(a1), -70)
        g.tris(np.array([[(pc[0], pc[1], -70), p0, p1]]), (0.10, 0.16, 0.20), 0.0, 1)
        q0 = (pc[0] + (pr[0] + 160) * math.cos(a0), pc[1] + (pr[1] + 160) * math.sin(a0), -40)
        q1 = (pc[0] + (pr[0] + 160) * math.cos(a1), pc[1] + (pr[1] + 160) * math.sin(a1), -40)
        g.quads(p0, p1, q1, q0, (0.35, 0.33, 0.32))
    for i in range(14):                                                                    # arched bridge
        t0, t1 = i / 14, (i + 1) / 14
        x0, x1 = 7700 + t0 * 4200, 7700 + t1 * 4200
        z0, z1 = -40 + 520 * math.sin(math.pi * t0), -40 + 520 * math.sin(math.pi * t1)
        for side in (-1, 1):
            yy = pc[1] + side * 260
            g.quads((x0, yy, z0), (x1, yy, z1), (x1, yy, z1 + 110), (x0, yy, z0 + 110), (0.62, 0.11, 0.08))
        g.quads((x0, pc[1] - 260, z0), (x1, pc[1] - 260, z1), (x1, pc[1] + 260, z1), (x0, pc[1] + 260, z0), (0.34, 0.24, 0.20))
    for i in range(8):                                                                     # stone lanterns
        a = -0.9 + i * 0.26
        x, y = 8000 * math.cos(a) + 400, 8000 * math.sin(a)
        box(g, (x, y, 20), (70, 70, 80), 0.0, (0.40, 0.38, 0.36))
        box(g, (x, y, 170), (55, 55, 70), 0.0, (1.0, 0.62, 0.30), emis=1.4, kind=2)
        roof(g, (x, y), 60, 60, 240, 80, 40, 20, 0.0, (0.36, 0.34, 0.33))
    placed = 0
    while placed < 55:
        r = rng.uniform(8600, 17000); a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if x < -4000 and abs(y) < 13000:                          # the stands / pagoda side
            continue
        if (x - pc[0]) ** 2 / (pr[0] + 500) ** 2 + (y - pc[1]) ** 2 / (pr[1] + 500) ** 2 < 1.0:
            continue
        if abs(x) < 4200 and y < -8500 and y > -12500:              # the hall
            continue
        _cherry(g, x, y, -60, rng.uniform(900, 1500), rng)
        placed += 1
    # lantern strings around the plaza
    poles = [(7300 * math.cos(a), 7300 * math.sin(a)) for a in np.linspace(-1.2, 1.2, 9)] + \
            [(7300 * math.cos(a), 7300 * math.sin(a)) for a in np.linspace(math.pi / 2 + 0.55, math.pi / 2 + 0.95, 2)] + \
            [(7300 * math.cos(a), 7300 * math.sin(a)) for a in np.linspace(-math.pi / 2 - 0.95, -math.pi / 2 - 0.55, 2)]
    for (x, y) in poles:
        box(g, (x, y, 700), (35, 35, 760), 0.0, (0.25, 0.08, 0.06))
        ico(g, (x, y, 1480), 70, (1.0, 0.30, 0.10), rng, sub=0, jit=0.0, squash=1.3, emis=1.6, kind=2)
    for (a, b) in zip(poles[:8], poles[1:9]):
        for t in np.linspace(0.12, 0.88, 6):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            z = 1400 - 260 * math.sin(math.pi * t)
            ico(g, (x, y, z), 40, (1.0, 0.25, 0.08), rng, sub=0, jit=0.0, squash=1.3, emis=1.5, kind=2)
    # karst peaks, near and far
    for ring_r, n, (hmin, hmax), (rmin, rmax) in ((19000.0, 16, (4500, 9000), (1400, 2600)),
                                                  (30000.0, 22, (7000, 14000), (2200, 4200)),
                                                  (44000.0, 26, (9000, 18000), (3000, 5500))):
        for i in range(n):
            a = 2 * math.pi * (i + rng.uniform(-0.35, 0.35)) / n
            rr = ring_r * rng.uniform(0.88, 1.12)
            x, y = rr * math.cos(a), rr * math.sin(a)
            _karst(g, x, y, hfun(x, y) - 100, rng.uniform(rmin, rmax), rng.uniform(hmin, hmax), rng)
    # floating sky lanterns (they bob in SCENE_VERT; the emission channel carries the phase)
    for i in range(46):
        r = rng.uniform(7500, 24000); a = rng.uniform(0, 2 * math.pi)
        x, y, z = r * math.cos(a), r * math.sin(a), rng.uniform(2600, 9000)
        box(g, (x, y, z), (22, 22, 32), rng.uniform(0, 1), (1.0, 0.55, 0.22), emis=float(rng.random()), kind=9)
    return g.array(), np.concatenate(eggs).astype("f4")


# ------------------------------------------------------------------------------------------------ paris
def _eiffel(g, x, y, H=20000.0, B=3800.0):
    iron = (0.16, 0.11, 0.08)
    # (height, centre offset of each leg from the tower axis, leg half-width)
    legs = [(-60, B, 520), (1400, B * 0.80, 470), (2900, B * 0.62, 420)]
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for (z0, o0, w0), (z1, o1, w1) in zip(legs[:-1], legs[1:]):
            c0 = np.array([x + sx * o0, y + sy * o0]); c1 = np.array([x + sx * o1, y + sy * o1])
            P0 = [(c0[0] + dx * w0, c0[1] + dy * w0, z0) for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            P1 = [(c1[0] + dx * w1, c1[1] + dy * w1, z1) for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            for k in range(4):
                g.quads(P0[k], P0[(k + 1) % 4], P1[(k + 1) % 4], P1[k], iron, 0.0, 4)
    # the arches between the legs under the first platform
    for yaw in (0, math.pi / 2, math.pi, -math.pi / 2):
        # an arch sagging from the platform edge: 2900 at the legs, ~1730 in the middle
        pts = [(-B * 0.8 + t * B * 1.6, B * 0.72, 2900 - 1170 * (1 - (2 * t - 1) ** 2)) for t in np.linspace(0, 1, 11)]
        W = _xf(pts, (x, y, 0), yaw)
        Wt = _xf([(p[0], p[1], p[2] + 160) for p in pts], (x, y, 0), yaw)
        for k in range(10):
            g.quads(W[k], W[k + 1], Wt[k + 1], Wt[k], iron, 0.0, 4)
    box(g, (x, y, 3000), (B * 0.70, B * 0.70, 110), 0.0, (0.30, 0.22, 0.15), emis=0.0, kind=0,
        top=(0.30, 0.22, 0.15))
    box(g, (x, y, 3000), (B * 0.70 + 4, B * 0.70 + 4, 40), 0.0, (1.0, 0.75, 0.35), emis=1.2, kind=2)   # lit gallery
    # second stage: 4 faces tapering to the second platform, then the shaft
    stages = [(3000, B * 0.55), (5900, B * 0.33), (9000, B * 0.20), (13000, B * 0.10), (17500, B * 0.05), (H - 900, 120)]
    for (z0, w0), (z1, w1) in zip(stages[:-1], stages[1:]):
        P0 = [(x + dx * w0, y + dy * w0, z0) for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        P1 = [(x + dx * w1, y + dy * w1, z1) for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        for k in range(4):
            g.quads(P0[k], P0[(k + 1) % 4], P1[(k + 1) % 4], P1[k], iron, 0.0, 4)
    box(g, (x, y, 5950), (B * 0.36, B * 0.36, 80), 0.0, (0.30, 0.22, 0.15))
    box(g, (x, y, 5950), (B * 0.36 + 4, B * 0.36 + 4, 30), 0.0, (1.0, 0.75, 0.35), emis=1.2, kind=2)
    box(g, (x, y, H - 900), (260, 260, 220), 0.0, (0.30, 0.22, 0.15), top=(1.0, 0.8, 0.5))
    box(g, (x, y, H - 500), (120, 120, 200), 0.0, (1.0, 0.85, 0.55), emis=2.0, kind=2)          # the beacon
    frustum(g, (x, y, H - 300), 60, 10, 0, 1100, 4, iron)


def _haussmann(g, x, y, hx, hy, H, rng):
    stone = tuple(np.clip(np.array((0.78, 0.71, 0.58)) * rng.uniform(0.85, 1.08), 0, 1))
    zinc = (0.24, 0.27, 0.34)
    box(g, (x, y, -60 + H / 2), (hx, hy, H / 2), 0.0, stone, kind=3, top=zinc)
    # mansard: a steep truncated roof
    ins = 260
    b = [(x + sx * hx, y + sy * hy, H - 60) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    t = [(x + sx * (hx - ins), y + sy * (hy - ins), H + 480) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    for k in range(4):
        g.quads(b[k], b[(k + 1) % 4], t[(k + 1) % 4], t[k], zinc)
    g.quads(t[0], t[1], t[2], t[3], (0.20, 0.22, 0.27))
    for _ in range(rng.integers(2, 5)):                                                     # chimneys
        cx = x + rng.uniform(-hx + ins, hx - ins); cy = y + rng.uniform(-hy + ins, hy - ins)
        box(g, (cx, cy, H + 600), (40, 90, 120), 0.0, (0.55, 0.42, 0.34))
    # dormer windows: small lit boxes on the mansard
    for sx in (-1, 1):
        for k in range(int(hy // 500)):
            yy = y - hy + 350 + k * 500
            box(g, (x + sx * (hx - ins * 0.45), yy, H + 200), (8, 45, 55), 0.0, (1.0, 0.75, 0.45),
                emis=0.9 if rng.random() < 0.5 else 0.0, kind=2 if rng.random() < 0.5 else 0)


def _plane_tree(g, x, y, s, rng):
    frustum(g, (x, y, -60), s * 0.05, s * 0.035, 0, s * 0.5, 5, (0.30, 0.26, 0.20))
    for _ in range(3):
        a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0, 0.12) * s
        ico(g, (x + d * math.cos(a), y + d * math.sin(a), -60 + s * rng.uniform(0.62, 0.8)), s * rng.uniform(0.24, 0.32),
            (0.12, 0.22, 0.09), rng, sub=1, jit=0.2, squash=0.85, kind=6, colvar=0.15)


def build_paris(seed=21):
    rng = np.random.default_rng(seed)
    g, eggs = G(), []
    # ground: pale paving around the arena, then asphalt streets
    terrain(g, [7600, 8000, 12000, 20000, 30000, 42000, 60000], 64, lambda x, y: -60.0,
            lambda x, y, z, nz, q: (((0.36 + 0.03 * q,) * 3) if math.hypot(x, y) < 7700 else (0.10 + 0.02 * q, 0.10 + 0.02 * q, 0.115 + 0.02 * q), 0), rng)
    for i in range(48):                                                                    # centre paving disc
        a0, a1 = 2 * math.pi * i / 48, 2 * math.pi * (i + 1) / 48
        g.tris(np.array([[(0, 0, -60), (7650 * math.cos(a0), 7650 * math.sin(a0), -60), (7650 * math.cos(a1), 7650 * math.sin(a1), -60)]]),
               (0.35, 0.34, 0.33))
    # ---- the main two-tier stand (+x), orange seats, white structure, a floodlit roof ----
    white, orange, grey = (0.78, 0.76, 0.72), (0.80, 0.33, 0.08), (0.45, 0.44, 0.44)
    stand(g, eggs, (5300.0, 0.0, 0.0), (0, 1, 0), (1, 0, 0), 18, 14000, 120.0, 80.0, 72.0, rng,
          tread=grey, riser=orange, rail=white)
    z_up = 18 * 80 + 520
    x_up = 5300 + 18 * 120 + 420
    box(g, (x_up - 200, 0, z_up / 2), (200, 7000, z_up / 2), 0.0, white)                     # fascia wall
    for k in range(7):                                                                    # LED banner boards
        cols = [(0.2, 0.5, 1.0), (1.0, 0.45, 0.1), (1.0, 1.0, 1.0), (0.9, 0.2, 0.5)]
        box(g, (x_up - 405, -6000 + k * 2000, z_up - 200), (4, 900, 150), 0.0, cols[k % 4], emis=1.2, kind=2)
    stand(g, eggs, (x_up, 0.0, z_up), (0, 1, 0), (1, 0, 0), 20, 14000, 120.0, 95.0, 72.0, rng,
          tread=grey, riser=orange, rail=white, z_base=z_up - 60)
    x_back = x_up + 20 * 120
    z_back = z_up + 20 * 95
    box(g, (x_back + 150, 0, z_back / 2 + 400), (150, 7200, z_back / 2 + 460), 0.0, white)
    for sy in (-1, 1):                                                                    # end walls
        box(g, (5300 + (x_back - 5300) / 2, sy * 7150, z_back / 2), ((x_back - 5300) / 2 + 150, 150, z_back / 2 + 60), 0.0, white)
    # cantilevered roof, orange underside, floodlight banks on its front edge
    rf = [(x_back + 300, -7400, z_back + 900), (x_back + 300, 7400, z_back + 900), (5600, 7400, z_back + 1500), (5600, -7400, z_back + 1500)]
    g.quads(*rf, (0.85, 0.40, 0.12))
    g.quads(*[(p[0], p[1], p[2] + 120) for p in rf], white)
    for k in range(10):
        box(g, (5700, -6300 + k * 1400, z_back + 1380), (60, 420, 70), 0.0, (1.0, 0.97, 0.9), emis=2.2, kind=2)
    # goal-end stands (+-y)
    for sy in (-1, 1):
        stand(g, eggs, (0.0, sy * 6350.0, 0.0), (1, 0, 0), (0, sy, 0), 12, 7600, 120.0, 80.0, 72.0, rng,
              tread=grey, riser=orange, rail=white)
        box(g, (0, sy * (6350 + 12 * 120 + 150), 12 * 80 / 2 + 200), (3900, 150, 12 * 80 / 2 + 260), 0.0, white)
    # ---- the Champ de Mars (-x) leading to the Eiffel Tower, lined with plane trees ----
    g.quads((-7400, -3600, -55), (-19000, -3600, -55), (-19000, 3600, -55), (-7400, 3600, -55), (0.10, 0.24, 0.08))
    for x in np.arange(-7800, -18500, -900):
        for side in (-1, 1):
            _plane_tree(g, x + rng.uniform(-80, 80), side * 3900, rng.uniform(1100, 1400), rng)
    for y in np.arange(-5400, 5500, 1350):
        _plane_tree(g, -6500 + rng.uniform(-100, 100), y, rng.uniform(1100, 1400), rng)
    _eiffel(g, -22500, 0)
    # ---- the Seine behind the tower, with quays and bridges ----
    xs0, xs1 = -28500, -25500
    g.quads((xs0, -60000, -260), (xs1, -60000, -260), (xs1, 60000, -260), (xs0, 60000, -260), (0.12, 0.16, 0.22), 0.0, 1)
    for xq in (xs0, xs1):
        box(g, (xq, 0, -160), (80, 60000, 100), 0.0, (0.55, 0.50, 0.42))
    for yb in (0.0, -9000.0, 11000.0):
        box(g, ((xs0 + xs1) / 2, yb, -20), ((xs1 - xs0) / 2 + 200, 700, 40), 0.0, (0.60, 0.55, 0.47))
        for k in range(3):
            xa = xs0 + (k + 0.5) * (xs1 - xs0) / 3
            box(g, (xa, yb, -150), (90, 650, 110), 0.0, (0.55, 0.50, 0.42))
    # ---- Haussmann blocks everywhere else, streetlamps along the streets ----
    step = 3300.0
    for ix in range(-12, 13):
        for iy in range(-12, 13):
            cx, cy = ix * step, iy * step
            d = math.hypot(cx, cy)
            if d < 11500 or d > 38000:
                continue
            if cx > 3500 and abs(cy) < 9500 and cx < 14000:          # the main stand
                continue
            if cx < -6200 and abs(cy) < 5200 and cx > -25000:        # Champ de Mars + tower
                continue
            if xs0 - 1800 < cx < xs1 + 1800:                          # the river
                continue
            for sx in (-1, 1):
                for sy in (-1, 1):
                    hx = 650 + rng.uniform(-60, 60); hy = 650 + rng.uniform(-60, 60)
                    _haussmann(g, cx + sx * 660, cy + sy * 660, hx, hy, rng.uniform(1900, 2500), rng)
            lx, ly = cx + step / 2, cy + step / 2
            box(g, (lx, ly, 200), (14, 14, 260), 0.0, (0.12, 0.12, 0.13))
            box(g, (lx, ly, 490), (40, 40, 45), 0.0, (1.0, 0.82, 0.52), emis=1.8, kind=2)
    # Sacre-Coeur on its hill, far behind the orange goal
    lathe(g, (7000, 40000, -60), [(9000, 0), (6500, 900), (3500, 2100), (0, 2500)], 14,
          lambda nz, zf, q: (0.12 + 0.03 * q, 0.20 + 0.03 * q, 0.10), rng=rng, jitter=0.06)
    bc = (7000, 40000, 2380)
    box(g, (bc[0], bc[1], bc[2] + 600), (1500, 900, 600), 0.0, (0.92, 0.90, 0.84), emis=0.15)
    lathe(g, (bc[0], bc[1], bc[2] + 1200), [(700, 0), (700, 500), (650, 900), (480, 1350), (220, 1650), (0, 1780)], 12,
          lambda nz, zf, q: (0.94, 0.92, 0.86), emis=0.15)
    for dx, dy in ((-1000, -500), (1000, -500), (-1000, 500), (1000, 500)):
        lathe(g, (bc[0] + dx, bc[1] + dy, bc[2] + 1200), [(280, 0), (280, 200), (220, 450), (0, 620)], 10,
              lambda nz, zf, q: (0.94, 0.92, 0.86), emis=0.15)
    return g.array(), np.concatenate(eggs).astype("f4")


# ------------------------------------------------------------------------------------------------ space
def build_space(seed=31):
    rng = np.random.default_rng(seed)
    g, eggs = G(), []
    metal, dark, cyan = (0.12, 0.13, 0.15), (0.08, 0.09, 0.11), (0.30, 0.90, 1.00)
    # the orbital platform: an octagon deck, a neon rim, a tapered underside with a glowing core
    R = 9400.0
    oct_ = [(R * math.cos(math.pi / 8 + k * math.pi / 4), R * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
    for k in range(8):
        a, b = oct_[k], oct_[(k + 1) % 8]
        # deck in 4 radial bands (panel variation)
        for i in range(4):
            t0, t1 = i / 4, (i + 1) / 4
            p = [(a[0] * t0, a[1] * t0, -60), (b[0] * t0, b[1] * t0, -60), (b[0] * t1, b[1] * t1, -60), (a[0] * t1, a[1] * t1, -60)]
            g.quads(*p, tuple(np.array(metal) * rng.uniform(0.8, 1.1)), 0.0, 5)
        g.quads((a[0], a[1], -60), (b[0], b[1], -60), (b[0], b[1], -760), (a[0], a[1], -760), dark, 0.0, 5)
        g.quads((a[0] * 1.002, a[1] * 1.002, -170), (b[0] * 1.002, b[1] * 1.002, -170),
                (b[0] * 1.002, b[1] * 1.002, -110), (a[0] * 1.002, a[1] * 1.002, -110), cyan, 2.0, 7)
        g.quads((a[0], a[1], -760), (b[0], b[1], -760), (b[0] * 0.35, b[1] * 0.35, -3400), (a[0] * 0.35, a[1] * 0.35, -3400), dark, 0.0, 5)
        g.quads((a[0] * 0.62, a[1] * 0.62, -2050), (b[0] * 0.62, b[1] * 0.62, -2050),
                (b[0] * 0.61, b[1] * 0.61, -2150), (a[0] * 0.61, a[1] * 0.61, -2150), cyan, 1.8, 7)
    frustum(g, (0, 0, -3400), R * 0.35, 900, 0, -1400, 16, (0.4, 0.95, 1.0), emis=2.5, kind=2, cap=False)
    for k in range(16):                                                                    # rim beacons
        a = 2 * math.pi * k / 16
        x, y = (R - 180) * math.cos(a), (R - 180) * math.sin(a)
        box(g, (x, y, 80), (40, 40, 140), a, (0.2, 0.22, 0.25))
        box(g, (x, y, 240), (46, 46, 22), a, cyan, emis=2.2, kind=7)
    # floating spectator decks (+-x): white metal stands with neon edges
    for sx in (-1, 1):
        stand(g, eggs, (sx * 5500.0, 0.0, 120.0), (0, 1, 0), (sx, 0, 0), 18, 9200, 115.0, 85.0, 70.0, rng,
              tread=(0.70, 0.73, 0.78), riser=(0.20, 0.22, 0.26), rail=(0.85, 0.88, 0.92))
        xe = sx * (5500 + 18 * 115)
        box(g, (xe + sx * 120, 0, 1000), (120, 4700, 1100), 0.0, (0.72, 0.75, 0.80), kind=5)
        box(g, (sx * 5480, 0, 90), (20, 4620, 16), 0.0, (0.3, 0.6, 1.0) if sx < 0 else (1.0, 0.55, 0.2), emis=2.2, kind=7)
        box(g, (xe + sx * 245, 0, 2080), (8, 4600, 20), 0.0, cyan, emis=2.0, kind=7)
    # neon gates behind the goals, in the team colours
    for sy, col in ((-1, (0.25, 0.55, 1.0)), (1, (1.0, 0.50, 0.15))):
        for i in range(24):
            a0, a1 = math.pi * i / 24, math.pi * (i + 1) / 24
            r0, r1 = 3600.0, 3800.0
            p = [(r0 * math.cos(a0), sy * 7400, -60 + r0 * math.sin(a0)), (r0 * math.cos(a1), sy * 7400, -60 + r0 * math.sin(a1)),
                 (r1 * math.cos(a1), sy * 7400, -60 + r1 * math.sin(a1)), (r1 * math.cos(a0), sy * 7400, -60 + r1 * math.sin(a0))]
            g.quads(*p, col, 2.2, 7)
            q = [(x, y + sy * 200, z) for x, y, z in p]
            g.quads(p[0], p[1], q[1], q[0], (0.2, 0.22, 0.26), 0.0, 5)
            g.quads(*q, col, 2.2, 7)
    # asteroids
    for _ in range(90):
        r = rng.uniform(14000, 60000); a = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(-16000, 18000)
        s = rng.uniform(200, 900) if rng.random() < 0.7 else rng.uniform(1200, 3600)
        col = (0.32, 0.29, 0.27) if rng.random() < 0.6 else (0.40, 0.33, 0.26)
        ico(g, (r * math.cos(a), r * math.sin(a), z), s, col, rng, sub=1, jit=0.35, squash=rng.uniform(0.6, 1.0), colvar=0.15)
    # a space station: hub, ring with lit windows, spokes, solar wings, blinking beacons
    sc = np.array([26000.0, 30000.0, 12000.0])
    frustum(g, (sc[0], sc[1], sc[2] - 3500), 1400, 1400, 0, 7000, 12, (0.72, 0.74, 0.78), kind=5)
    frustum(g, (sc[0], sc[1], sc[2] + 3500), 1400, 400, 0, 900, 12, (0.72, 0.74, 0.78), kind=5)
    frustum(g, (sc[0], sc[1], sc[2] - 3500), 400, 1400, -900, 0, 12, (0.72, 0.74, 0.78), kind=5, cap=False)
    for k in range(36):
        a = 2 * math.pi * k / 36
        box(g, (sc[0] + 7000 * math.cos(a), sc[1] + 7000 * math.sin(a), sc[2]), (420, 640, 420), a,
            (0.80, 0.82, 0.86), kind=3)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        box(g, (sc[0] + 4200 * math.cos(a), sc[1] + 4200 * math.sin(a), sc[2]), (2800, 90, 90), a, (0.6, 0.62, 0.66), kind=5)
    for sz in (-1, 1):
        for sx in (-1, 1):
            box(g, (sc[0] + sx * 6000, sc[1], sc[2] + sz * 4200), (4200, 1200, 20), 0.0, (0.10, 0.18, 0.42), kind=5)
            box(g, (sc[0] + sx * 1900, sc[1], sc[2] + sz * 4200), (500, 60, 60), 0.0, (0.6, 0.62, 0.66))
    for k in range(6):
        a = 2 * math.pi * k / 6
        box(g, (sc[0] + 7000 * math.cos(a), sc[1] + 7000 * math.sin(a), sc[2] + 480), (40, 40, 40), a, (1.0, 0.15, 0.1), emis=2.5, kind=2)
    # a relay dish
    dc = (-30000.0, -22000.0, 7000.0)
    lathe(g, dc, [(0, 0), (900, 120), (1800, 420), (2600, 950)], 14, lambda nz, zf, q: (0.78, 0.80, 0.84), kind=5)
    box(g, (dc[0], dc[1], dc[2] - 1500), (120, 120, 1500), 0.0, (0.5, 0.52, 0.56))
    box(g, (dc[0], dc[1], dc[2] + 1400), (60, 60, 60), 0.0, (1.0, 0.2, 0.1), emis=2.5, kind=2)
    return g.array(), np.concatenate(eggs).astype("f4")


BUILDERS = {"temple": build_temple, "paris": build_paris, "space": build_space}


def load_or_build(name, data_dir):
    """(scene mesh (N, 8), crowd instances (M, 8)) for a map; cached on disk, rebuilt when SCENE_VERSION changes."""
    path = os.path.join(data_dir, "scene_{}_cache.npz".format(name))
    try:
        d = np.load(path, allow_pickle=False)
        if int(d["version"]) == SCENE_VERSION:
            return d["mesh"], d["crowd"]
    except (OSError, ValueError, KeyError):
        pass
    mesh, crowd = BUILDERS[name]()
    try:
        np.savez(path, mesh=mesh, crowd=crowd, version=np.int32(SCENE_VERSION))
    except OSError:
        pass
    return mesh, crowd

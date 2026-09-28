"""Maps: the scenery around the arena, its sky and light, the field style, and the crowd.

  valley  the original low-poly evening valley (landscape.py, unchanged)
  temple  "Forbidden Temple"-style: pink dusk, karst peaks, pagodas, a paifang gate, cherry trees, lanterns
  paris   "Parc de Paris"-style: noon, the Eiffel Tower down the Champ de Mars, Haussmann blocks, the Seine,
          a two-tier stadium stand under floodlights
  space   the arena on an orbital platform: stars, a ringed gas giant, moons, the planet below, asteroids, a station
          (no crowd)

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
SCENE_VERSION = 22


def _n(v):
    v = np.asarray(v, "f8")
    return tuple(float(x) for x in v / np.linalg.norm(v))


# light + sky + field per map. sun_dir/sun_col/zen/mid/hor/glow/ground -> COMMON's uniforms; cloudA/B -> SKY_FRAG;
# haze (near, far, strength) -> SCENE_FRAG / CROWD_FRAG; grass -> the turf's base colour.
THEMES = {
    "valley": dict(sun_dir=_n((-0.45, 0.75, 0.30)), sun_col=(1.00, 0.88, 0.72), zen=(0.035, 0.05, 0.13),
                   mid=(0.22, 0.15, 0.30), hor=(0.95, 0.46, 0.20), glow=(1.0, 0.55, 0.25), ground=(0.04, 0.035, 0.05),
                   amb=(1.0, 1.0, 1.0), grass=(0.080, 0.265, 0.050), cloudA=(0.95, 0.55, 0.35, 0.52),
                   cloudB=(0.35, 0.25, 0.40, 0.55), haze=(9000.0, 48000.0, 0.85), glass=1.0, stars=0.45),
    "temple": dict(sun_dir=_n((0.62, 0.74, 0.14)), sun_col=(1.00, 0.72, 0.80), zen=(0.07, 0.035, 0.16),
                   mid=(0.44, 0.15, 0.38), hor=(1.00, 0.48, 0.55), glow=(1.0, 0.45, 0.58), ground=(0.05, 0.03, 0.05),
                   amb=(1.02, 0.88, 1.02), grass=(0.050, 0.150, 0.130), cloudA=(1.0, 0.56, 0.66, 0.44),
                   cloudB=(0.42, 0.18, 0.40, 0.80), haze=(9000.0, 60000.0, 0.72), glass=1.0, stars=0.25,
                   cloud_shape=(1.6, 1.6)),
    "paris": dict(sun_dir=_n((0.30, -0.40, 0.86)), sun_col=(1.05, 1.00, 0.94), zen=(0.13, 0.30, 0.72),
                  mid=(0.36, 0.55, 0.88), hor=(0.72, 0.82, 0.94), glow=(1.0, 0.95, 0.85), ground=(0.30, 0.30, 0.32),
                  amb=(1.12, 1.12, 1.18), grass=(0.085, 0.27, 0.065), cloudA=(1.0, 1.0, 1.0, 0.54),
                  cloudB=(0.72, 0.76, 0.86, 0.85), haze=(12000.0, 75000.0, 0.45), glass=1.0, stars=0.0,
                  cloud_shape=(1.5, 1.5), night=0.0),
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
        a[:, :, 3:6] = c if c.ndim == 1 else (c.reshape(n, 3, 3) if c.size == 9 * n else c.reshape(n, 1, 3))
        e = np.asarray(emis, "f4")
        a[:, :, 6] = e if e.ndim == 0 else (e.reshape(n, 3) if e.size == 3 * n else e.reshape(n, 1))
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


def lathe(g, c, prof, n, colfn, kind=0, emis=0.0, rng=None, jitter=0.0, rot=0.0, vertex_zf=False):
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
                e = ((t[:, 2] - c[2]) / ztop)[None, :] if vertex_zf else emis
                g.tris(t[None], colfn(nz, zf, rng.random() if rng is not None else 0.5), e, kind)


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


def roof(g, c, hx, hy, z0, rise, over, curl, yaw, col, soffit=None, top_frac=0.14, kind=0):
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
            g.quads(A[k], A[k1], B[k1], B[k], col, 0.0, kind)
    top = _xf([(0, 0, rise)], (c[0], c[1], z0), yaw)[0]
    for k in range(8):
        g.tris(np.array([[R2[k], R2[(k + 1) % 8], top]]), col, 0.0, kind)
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
    wall, roofc, soff, gold = (0.50, 0.09, 0.07), (0.20, 0.26, 0.32), (0.30, 0.08, 0.06), (0.95, 0.70, 0.30)
    box(g, (x, y, z + 120), (w * 0.75, w * 0.75, 120), yaw, (0.30, 0.28, 0.27))            # terrace
    z += 240
    for t in range(tiers):
        box(g, (x, y, z + th * 0.5), (w * 0.5, w * 0.5, th * 0.5), yaw, wall, kind=13)
        # glowing paper windows on every side
        for side in range(4):
            a = yaw + side * math.pi / 2
            off = w * 0.5 + 14
            cx, cy = x + off * math.cos(a), y + off * math.sin(a)
            box(g, (cx, cy, z + th * 0.5), (6, w * 0.30, th * 0.26), a, (1.0, 0.55, 0.25), emis=1.3, kind=16)
        z += th
        roof(g, (x, y), w * 0.5, w * 0.5, z, th * 0.55, w * 0.30, th * 0.38, yaw, roofc, soffit=soff, kind=12)
        box(g, (x, y, z - 30), (w * 0.5 + 12, w * 0.5 + 12, 18), yaw, (0.62, 0.45, 0.20))   # gilded band
        w *= 0.80
        z += th * 0.35
    frustum(g, (x, y, z), w * 0.12, w * 0.02, 0, th * 1.6, 6, gold, emis=0.35)
    for k in range(3):
        ico(g, (x, y, z + th * (0.4 + 0.35 * k)), w * 0.10, gold, rng, sub=0, jit=0.0, emis=0.4)


def _paifang(g, x, y, yaw, W=5400.0, H=3900.0):
    red, blue, gold, stone, roofc = (0.48, 0.07, 0.06), (0.10, 0.22, 0.42), (0.95, 0.72, 0.30), (0.34, 0.32, 0.31), (0.20, 0.26, 0.32)
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
        roof(g, (p[0], p[1]), half, 260, h - 70, 520, 280, 230, yaw, roofc, soffit=(0.12, 0.10, 0.12), kind=12)
    p = _xf([(0, 0, 0)], (x, y, 0), yaw)[0]
    box(g, (p[0], p[1], H - 700), (520, 40, 170), yaw, gold, emis=0.9, kind=2)            # the name plaque


def _hall(g, x, y, W, D, H, rng, yaw=0.0):
    stone, red, roofc, paper = (0.33, 0.31, 0.30), (0.50, 0.09, 0.07), (0.20, 0.26, 0.32), (1.0, 0.58, 0.28)
    box(g, (x, y, 60), (W / 2 + 700, D / 2 + 700, 120), yaw, stone)
    for k in range(4):                                                                    # front steps
        p = _xf([(0, D / 2 + 700 + 150 + k * 150, 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], -60 + (4 - k) * 30), (1600, 75, (4 - k) * 30), yaw, stone)
    box(g, (x, y, 180 + H / 2), (W / 2, D / 2, H / 2), yaw, red, kind=13)
    for side in (1, -1):
        p = _xf([(0, side * (D / 2 + 14), 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], 180 + H * 0.45), (W / 2 - 250, 6, H * 0.3), yaw, paper, emis=1.0, kind=16)
    ncol = int(W // 700) + 1
    for k in range(ncol):
        px = -W / 2 + k * W / (ncol - 1)
        p = _xf([(px, D / 2 + 260, 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], 180 + H / 2), (70, 70, H / 2), yaw, red)
    roof(g, (x, y), W / 2 + 260, D / 2 + 260, 180 + H, 650, 520, 300, yaw, roofc, soffit=(0.25, 0.07, 0.05), kind=12)
    box(g, (x, y, 180 + H + 650 + 180), (W / 2 * 0.62, D / 2 * 0.5, 180), yaw, red, kind=13)
    roof(g, (x, y), W / 2 * 0.62, D / 2 * 0.5, 180 + H + 650 + 360, 480, 380, 240, yaw, roofc, soffit=(0.25, 0.07, 0.05), kind=12)


def _cherry(g, x, y, z, s, rng):
    frustum(g, (x, y, z), s * 0.06, s * 0.035, 0, s * 0.55, 5, (0.20, 0.11, 0.09))
    pinks = [(0.96, 0.58, 0.72), (0.90, 0.46, 0.64), (1.0, 0.72, 0.84), (0.85, 0.40, 0.58)]
    for _ in range(rng.integers(5, 8)):
        a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0.05, 0.30) * s
        ico(g, (x + d * math.cos(a), y + d * math.sin(a), z + s * rng.uniform(0.55, 0.78)), s * rng.uniform(0.18, 0.28),
            pinks[rng.integers(0, 4)], rng, sub=1, jit=0.22, squash=0.8, kind=6, colvar=0.08)


def _karst(g, x, y, z, r, h, rng):
    """A karst limestone peak: a rounded tower (14 sides, 11 rings, jittered). Vegetation is drawn per pixel by
    SCENE kind 8 from a noise field and the per-vertex height fraction -- per-face green showed the triangles."""
    prof = [(r, -300), (r * 1.05, h * 0.08), (r * 1.06, h * 0.18), (r * 1.00, h * 0.30), (r * 0.95, h * 0.42),
            (r * 0.90, h * 0.54), (r * 0.83, h * 0.66), (r * 0.74, h * 0.77), (r * 0.60, h * 0.87), (r * 0.40, h * 0.95),
            (r * 0.14, h)]
    tone = rng.uniform(0.9, 1.08)
    lathe(g, (x, y, z), prof, 14, lambda nz, zf, q: (0.40 * tone, 0.40 * tone, 0.43 * tone), kind=8, rng=rng,
          jitter=0.08, rot=rng.uniform(0, 1), vertex_zf=True)


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
    box(g, (xb - 250, 0, (top - 60) / 2 + 400), (250, 5600, (top + 60) / 2 + 400), 0.0, (0.40, 0.08, 0.06), kind=13)  # back wall
    for k in range(13):
        yy = -5400 + k * 900
        box(g, (xb + 300, yy, top + 450), (60, 60, 450), 0.0, (0.50, 0.09, 0.07))
    roof(g, (xb + 100, 0), 520, 5500, top + 900, 520, 420, 260, 0.0, (0.20, 0.26, 0.32), soffit=(0.30, 0.07, 0.05), kind=12)
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


def panel(g, A, B, C, D, kind=4):
    """A lattice panel A (bottom left) B (bottom right) C (top right) D (top left): SCENE kind 4 reads the panel's
    own (u across -1..1, v up 0..1) from the vertex colour and draws its lit borders + X bracing."""
    P = np.array([[A, B, C], [A, C, D]], "f4")
    col = np.array([[(-1, 0, 0), (1, 0, 0), (1, 1, 0)], [(-1, 0, 0), (1, 1, 0), (-1, 1, 0)]], "f4")
    g.tris(P, col, 0.0, kind)


def _eiffel(g, cx, cy, H, day=False):
    """The Eiffel Tower in its real proportions (concave exponential profile; four legs joined by the big arches,
    merging at the second floor into one shaft; top floor; antenna). Every face is split into lattice panels whose
    borders and X bracing light up blue (SCENE kind 4), with warm lamps inside -- like the lit tower at night."""
    def w(z):
        return H * (0.017 + 0.176 * math.exp(-3.7 * z / H))

    def t(z):
        return H * 0.075 * (1.0 - 0.35 * z / (0.357 * H))
    z0g = -60.0
    z1f, z2 = 0.178 * H, 0.357 * H
    zs = [0.0, 0.06 * H, 0.12 * H, z1f, 0.235 * H, 0.295 * H, z2]
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for za, zb in zip(zs[:-1], zs[1:]):
            wa, wb = w(za), w(zb)
            ia, ib = max(wa - t(za), 0.0), max(wb - t(zb), 0.0)

            def P(z, a, b):
                return (cx + sx * a, cy + sy * b, z + z0g)
            panel(g, P(za, wa, ia), P(za, wa, wa), P(zb, wb, wb), P(zb, wb, ib))       # outer faces
            panel(g, P(za, ia, wa), P(za, wa, wa), P(zb, wb, wb), P(zb, ib, wb))
            panel(g, P(za, ia, ia), P(za, ia, wa), P(zb, ib, wb), P(zb, ib, ib))       # inner faces
            panel(g, P(za, ia, ia), P(za, wa, ia), P(zb, wb, ib), P(zb, ib, ib))
    blue = (0.35, 0.62, 1.0)
    iron = (0.30, 0.22, 0.16) if day else (0.05, 0.05, 0.07)
    # floors: dark decks with a warm window band and blue lit edges
    for zf_, ext in ((z1f, 0.010), (z2, 0.007)):
        hw = w(zf_) + ext * H
        box(g, (cx, cy, zf_ + z0g), (hw, hw, 0.009 * H), 0.0, iron)
        box(g, (cx, cy, zf_ + z0g), (hw + 8, hw + 8, 0.0035 * H), 0.0, (1.0, 0.72, 0.40), emis=1.3, kind=2, top=iron)
        for dz in (0.009 * H, -0.009 * H):             # lit rims only on the sides (a lit top face was a huge square)
            box(g, (cx, cy, zf_ + z0g + dz), (hw + 14, hw + 14, 0.0012 * H), 0.0, iron if day else blue, emis=0.0 if day else 2.2, kind=0 if day else 7, top=iron)
    # the shaft: panels roughly as tall as they are wide
    z = z2
    zt = 0.852 * H
    while z < zt - 1.0:
        zn = min(zt, z + max(1.7 * w(z), 0.02 * H))
        wa, wb = w(z), w(zn)
        c4a = [(cx + a * wa, cy + b * wa, z + z0g) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        c4b = [(cx + a * wb, cy + b * wb, zn + z0g) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        for k in range(4):
            panel(g, c4a[k], c4a[(k + 1) % 4], c4b[(k + 1) % 4], c4b[k])
        z = zn
    hw = w(zt) + 0.005 * H
    box(g, (cx, cy, zt + z0g), (hw, hw, 0.010 * H), 0.0, iron)
    box(g, (cx, cy, zt + z0g + 0.012 * H), (hw * 0.7, hw * 0.7, 0.012 * H), 0.0, iron if day else blue, emis=0.0 if day else 2.0, kind=0 if day else 7, top=iron)
    frustum(g, (cx, cy, zt + z0g + 0.024 * H), hw * 0.4, 0.002 * H, 0, 0.12 * H, 6, iron)
    box(g, (cx, cy, H + z0g), (0.003 * H, 0.003 * H, 0.004 * H), 0.0, (1.0, 0.3, 0.2), emis=3.0, kind=2)



def _arc_pt(C, s, r, a):
    return (C[0] + s * r * math.cos(a), C[1] + r * math.sin(a))


def _curved_stand(g, eggs, s, rng, team_col, team_dark, amax=0.235, Lc=30000.0, x0=5350.0, fans=0):
    """One long side stand (s=+1 on +x, -1 on -x): two curved tiers facing the pitch, a sweeping roof with
    floodlights and banners, end walls with graffiti. Eggs on every seat."""
    C = (-s * Lc, 0.0)
    R0 = Lc + x0
    NS = 30
    A = np.linspace(-amax, amax, NS + 1)
    tread, white = (0.36, 0.36, 0.38), (0.82, 0.82, 0.84)
    prof = []                                   # (r_in, r_out, top z) of every step, for the end walls

    def tier(r_start, z_start, rows, depth, rise, z_floor):
        for k in range(rows):
            r_in, r_out = r_start + k * depth, r_start + (k + 1) * depth
            zt, zp = z_start + (k + 1) * rise, z_start + k * rise
            prof.append((r_in, r_out, zt))
            for i in range(NS):
                a0, a1 = A[i], A[i + 1]
                p = [_arc_pt(C, s, r, a) for r, a in ((r_in, a0), (r_in, a1), (r_out, a1), (r_out, a0))]
                g.quads((*p[0], zt), (*p[1], zt), (*p[2], zt), (*p[3], zt), tread)
                g.quads((*p[0], zp), (*p[1], zp), (*p[1], zt), (*p[0], zt), team_dark)          # riser
            # eggs along the row
            rr = r_in + depth * 0.42
            n = int(2 * amax * rr // 72.0)
            ang = (np.arange(n) - (n - 1) * 0.5) * (72.0 / rr)
            keep = rng.random(n) > 0.05
            ang = ang[keep]
            px = C[0] + s * rr * np.cos(ang); py = C[1] + rr * np.sin(ang)
            pz = np.full(len(ang), zt)
            cols = EGG_COLS[rng.integers(0, len(EGG_COLS), len(ang))] * rng.uniform(0.85, 1.1, (len(ang), 1))
            eggs.append(np.concatenate([np.stack([px, py, pz], 1), np.clip(cols, 0, 1), fans + 0.999 * rng.random((len(ang), 1)),
                                        (74.0 * rng.uniform(0.9, 1.1, len(ang)))[:, None]], 1))
        # the solid body under the tier (seen from below / the ends)
        r_end = r_start + rows * depth
        for i in range(NS):
            a0, a1 = A[i], A[i + 1]
            b0, b1 = _arc_pt(C, s, r_end, a0), _arc_pt(C, s, r_end, a1)
            g.quads((*b0, z_floor), (*b1, z_floor), (*b1, z_start + rows * rise), (*b0, z_start + rows * rise), white)
        return r_end, z_start + rows * rise

    # lower tier, a walkway with a team-colour LED fascia, upper tier
    r1, z1 = tier(R0, 0.0, 16, 120.0, 78.0, -60.0)
    z_walk = z1 + 120.0
    rf = r1 + 420.0
    for i in range(NS):
        a0, a1 = A[i], A[i + 1]
        p = [_arc_pt(C, s, r, a) for r, a in ((r1, a0), (r1, a1), (rf, a1), (rf, a0))]
        g.quads((*p[0], z_walk), (*p[1], z_walk), (*p[2], z_walk), (*p[3], z_walk), (0.30, 0.30, 0.32))
        f0, f1 = _arc_pt(C, s, rf, a0), _arc_pt(C, s, rf, a1)
        g.quads((*f0, z_walk), (*f1, z_walk), (*f1, z_walk + 520), (*f0, z_walk + 520), white)
        g.quads((*_arc_pt(C, s, rf - 4, a0), z_walk + 300), (*_arc_pt(C, s, rf - 4, a1), z_walk + 300),
                (*_arc_pt(C, s, rf - 4, a1), z_walk + 380), (*_arc_pt(C, s, rf - 4, a0), z_walk + 380), team_col, 1.8, 7)
    prof.append((r1, rf, z_walk + 520.0))
    r2, z2 = tier(rf, z_walk + 520.0, 18, 125.0, 100.0, z_walk)
    # back wall
    rb = r2 + 150.0
    for i in range(NS):
        a0, a1 = A[i], A[i + 1]
        b0, b1 = _arc_pt(C, s, rb, a0), _arc_pt(C, s, rb, a1)
        g.quads((*b0, -60), (*b1, -60), (*b1, z2 + 900), (*b0, z2 + 900), white)
    # sweeping roof: front edge rising toward the middle, team-coloured underside, floodlights + banners
    zr = z2 + 700.0
    for i in range(NS):
        a0, a1 = A[i], A[i + 1]
        zf0 = zr + 950.0 * math.cos(a0 / amax * math.pi / 2); zf1 = zr + 950.0 * math.cos(a1 / amax * math.pi / 2)
        f0, f1 = _arc_pt(C, s, R0 + 700.0, a0), _arc_pt(C, s, R0 + 700.0, a1)
        b0, b1 = _arc_pt(C, s, rb + 200.0, a0), _arc_pt(C, s, rb + 200.0, a1)
        g.quads((*f0, zf0), (*f1, zf1), (*b1, z2 + 900), (*b0, z2 + 900), team_dark)                 # underside
        g.quads((*f0, zf0 + 140), (*f1, zf1 + 140), (*b1, z2 + 1040), (*b0, z2 + 1040), white)        # top
        g.quads((*f0, zf0), (*f1, zf1), (*f1, zf1 + 140), (*f0, zf0 + 140), team_col, 1.6, 7)          # lit edge
        if i % 2 == 0:                                                                                  # floodlights
            m = _arc_pt(C, s, R0 + 760.0, (a0 + a1) / 2)
            box(g, (m[0], m[1], (zf0 + zf1) / 2 + 40), (40, 260, 60), 0.0, (1.0, 0.98, 0.92), emis=2.4, kind=2)
        else:                                                                                           # banners
            m = _arc_pt(C, s, R0 + 720.0, (a0 + a1) / 2)
            bc = team_col if (i // 2) % 2 else (0.95, 0.95, 0.97)
            box(g, (m[0], m[1], (zf0 + zf1) / 2 - 420), (8, 110, 380), 0.0, bc, emis=0.35)
    prof.append((r2, rb, z2 + 900.0))
    # end walls following the stepped profile, with graffiti at street level on the front part
    for a_end in (-amax, amax):
        sg = 1.0 if a_end > 0 else -1.0
        t = np.array([-s * math.sin(a_end), math.cos(a_end)]) * sg          # outward along the arc
        for r_a, r_b, zt in prof:
            e0, e1 = _arc_pt(C, s, r_a, a_end), _arc_pt(C, s, r_b, a_end)
            g.quads((*e0, -60), (*e1, -60), (*e1, zt), (*e0, zt), white)
            if r_b < R0 + 2000.0:
                h = min(zt - 30.0, 1100.0)
                g0, g1 = np.array(e0) + t * 30.0, np.array(e1) + t * 30.0
                g.quads((*g0, -40), (*g1, -40), (*g1, h), (*g0, h), (1, 1, 1), 0.0, 11)
        if a_end < 0:                                                       # facing the garden
            er = np.array([s * math.cos(a_end), math.sin(a_end)])
            width = (6 * 7 - 1) * 44.0
            if s > 0:                     # the vis mirrors x: this wall reads right-to-left along er, so flip it
                base = np.array(_arc_pt(C, s, R0 + 180.0 + width, a_end)) + t * 55.0
                _text(g, "I<PARIS", base, -er, t, 330.0, 44.0)
            else:
                base = np.array(_arc_pt(C, s, R0 + 180.0, a_end)) + t * 55.0
                _text(g, "I<PARIS", base, er, t, 330.0, 44.0)


GLYPHS = {
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "<": ["00000", "01010", "11111", "11111", "01110", "00100", "00000"],   # a heart
}


def _text(g, txt, base, along, out, z0, px):
    """Block letters (5x7 pixel font) standing proud of a wall: `along` = reading direction, `out` = wall normal."""
    x = 0.0
    for ch in txt:
        rows = GLYPHS[ch]
        col = (0.95, 0.12, 0.20) if ch == "<" else (0.98, 0.35, 0.62)
        for j, row in enumerate(rows):
            for i, bit in enumerate(row):
                if bit != "1":
                    continue
                c = base + along * (x + (i + 0.5) * px)
                box(g, (c[0], c[1], z0 + (6 - j + 0.5) * px), (px * 0.5, px * 0.5, px * 0.5),
                    math.atan2(along[1], along[0]), col, emis=0.5)
                cb = base + along * (x + (i + 0.5) * px + px * 0.18) - out * 6.0          # dark drop shadow
                box(g, (cb[0], cb[1], z0 + (6 - j + 0.5) * px - px * 0.18), (px * 0.5, px * 0.5, px * 0.5),
                    math.atan2(along[1], along[0]), (0.05, 0.05, 0.08))
        x += 6 * px


def _topiary(g, x, y, s, rng):
    frustum(g, (x, y, -60), s * 0.32, s * 0.02, 0, s, 9, (0.10, 0.27, 0.10), kind=6)
    for _ in range(9):
        a = rng.uniform(0, 2 * math.pi); h = rng.uniform(0.1, 0.85)
        r = s * 0.32 * (1 - h) + 6
        ico(g, (x + r * math.cos(a), y + r * math.sin(a), -60 + h * s), s * 0.035, (1.0, 0.55, 0.12), rng, sub=0,
            jit=0.0, emis=0.25)


def _parterre(g, x0, y0, x1, y1, rr=260.0):
    """A lawn bed with rounded corners, raised a little, with a white stone curb."""
    pts = []
    for (cx, cy, a0) in ((x1 - rr, y1 - rr, 0.0), (x0 + rr, y1 - rr, math.pi / 2), (x0 + rr, y0 + rr, math.pi),
                         (x1 - rr, y0 + rr, 1.5 * math.pi)):
        for k in range(5):
            a = a0 + (math.pi / 2) * k / 4
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    n = len(pts)
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        g.tris(np.array([[(c[0], c[1], -42), (a[0], a[1], -42), (b[0], b[1], -42)]]), (0.13, 0.34, 0.10))
        # curb: outward normal offset
        da = np.array(a) - c; db = np.array(b) - c
        a2 = np.array(a) + da / np.linalg.norm(da) * 40; b2 = np.array(b) + db / np.linalg.norm(db) * 40
        g.quads((*a, -45), (*b, -45), (*b2, -45), (*a2, -45), (0.86, 0.85, 0.83))
        g.quads((*a2, -60), (*b2, -60), (*b2, -45), (*a2, -45), (0.86, 0.85, 0.83))


def _fountain(g, x, y, rng):
    white, silver, gold = (0.86, 0.86, 0.88), (0.70, 0.72, 0.76), (1.0, 0.74, 0.22)
    lathe(g, (x, y, -60), [(1200, 0), (1200, 90), (1100, 110), (1100, 40)], 28, lambda nz, zf, q: white)
    g.tris(np.array([[(x, y, 10), (x + 1100 * math.cos(a0), y + 1100 * math.sin(a0), 10),
                      (x + 1100 * math.cos(a1), y + 1100 * math.sin(a1), 10)]
                     for a0, a1 in zip(np.linspace(0, 2 * math.pi, 29)[:-1], np.linspace(0, 2 * math.pi, 29)[1:])]),
           (0.20, 0.35, 0.45), 0.0, 1)
    lathe(g, (x, y, -60), [(420, 0), (420, 200), (260, 320), (200, 700), (330, 780), (330, 820), (120, 900)], 16,
          lambda nz, zf, q: silver, kind=5)
    ico(g, (x, y, 1250), 360, gold, rng, sub=2, jit=0.0, emis=0.35, kind=5)
    for k in range(20):                                     # the silver swirl around the golden ball
        a0, a1 = 2 * math.pi * k / 20, 2 * math.pi * (k + 1) / 20
        r = 520
        z0 = 1250 + 180 * math.sin(a0 * 2); z1 = 1250 + 180 * math.sin(a1 * 2)
        p0 = (x + r * math.cos(a0), y + r * math.sin(a0)); p1 = (x + r * math.cos(a1), y + r * math.sin(a1))
        g.quads((*p0, z0 - 45), (*p1, z1 - 45), (*p1, z1 + 45), (*p0, z0 + 45), silver, 0.0, 5)


def _building(g, x, y, hx, hy, H, rng, yaw=0.0):
    """A Parisian block: limestone facade (lit windows), zinc mansard or a flat roof with rooftop boxes,
    chimneys. Colour, height, footprint and roof vary per building."""
    stone = [(0.80, 0.74, 0.62), (0.74, 0.66, 0.52), (0.70, 0.70, 0.69), (0.82, 0.70, 0.60), (0.77, 0.73, 0.66)]
    col = tuple(np.clip(np.array(stone[rng.integers(0, len(stone))]) * rng.uniform(0.9, 1.06), 0, 1))
    zinc = (0.24 + rng.uniform(-0.03, 0.03), 0.27, 0.34)
    box(g, (x, y, -60 + H / 2), (hx, hy, H / 2), yaw, col, kind=3, top=zinc)
    box(g, (x, y, -60 + 330), (hx + 12, hy + 12, 18), yaw, tuple(np.array(col) * 0.85))              # cornice
    if rng.random() < 0.7:
        ins = rng.uniform(200, 320)
        b = _xf([(sx * hx, sy * hy, H - 60) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))], (x, y, 0), yaw)
        t = _xf([(sx * (hx - ins), sy * (hy - ins), H - 60 + rng.uniform(380, 560)) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))], (x, y, 0), yaw)
        for k in range(4):
            g.quads(b[k], b[(k + 1) % 4], t[(k + 1) % 4], t[k], zinc)
        g.quads(t[0], t[1], t[2], t[3], (0.20, 0.22, 0.27))
    else:
        for _ in range(rng.integers(1, 3)):
            box(g, (x + rng.uniform(-hx, hx) * 0.5, y + rng.uniform(-hy, hy) * 0.5, H - 60 + 120), (rng.uniform(120, 300), rng.uniform(120, 300), 120), yaw, (0.5, 0.5, 0.52))
    for _ in range(rng.integers(1, 4)):
        box(g, (x + rng.uniform(-0.6, 0.6) * hx, y + rng.uniform(-0.6, 0.6) * hy, H + 520), (35, 80, 110), yaw, (0.55, 0.42, 0.34))


def _plane_tree(g, x, y, s, rng):
    frustum(g, (x, y, -60), s * 0.05, s * 0.035, 0, s * 0.5, 5, (0.30, 0.26, 0.20))
    for _ in range(3):
        a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0, 0.12) * s
        ico(g, (x + d * math.cos(a), y + d * math.sin(a), -60 + s * rng.uniform(0.62, 0.8)), s * rng.uniform(0.24, 0.32),
            (0.12, 0.22, 0.09), rng, sub=1, jit=0.2, squash=0.85, kind=6, colvar=0.15)


def build_paris(seed=21):
    rng = np.random.default_rng(seed)
    g, eggs = G(), []
    pink = (0.60, 0.43, 0.42)
    # ground layers >= 6 uu apart: coplanar ones z-fought (the flicker around the pitch)
    terrain(g, [7250, 9000, 13000, 20000, 30000, 42000, 60000], 64, lambda x, y: -66.0,
            lambda x, y, z, nz, q: ((0.30 + 0.03 * q, 0.29 + 0.03 * q, 0.28 + 0.03 * q), 0), rng)
    for i in range(48):                                                                    # paving round the arena
        a0, a1 = 2 * math.pi * i / 48, 2 * math.pi * (i + 1) / 48
        g.tris(np.array([[(0, 0, -60), (7250 * math.cos(a0), 7250 * math.sin(a0), -60), (7250 * math.cos(a1), 7250 * math.sin(a1), -60)]]),
               pink)
    # ---- the two long stands: blue (-x) and orange (+x) ----
    _curved_stand(g, eggs, 1, rng, (0.30, 0.55, 1.0), (0.10, 0.16, 0.40), fans=1)
    _curved_stand(g, eggs, -1, rng, (1.0, 0.50, 0.12), (0.48, 0.18, 0.05), fans=2)
    # ---- the garden behind the blue goal: pink paths, lawn beds, the golden-sphere fountain, topiaries,
    #      statues, lamps ----
    g.quads((-7500, -7250, -54), (7500, -7250, -54), (7500, -15500, -54), (-7500, -15500, -54), pink)
    for sx in (-1, 1):
        xa, xb = sorted((sx * 1900.0, sx * 6200.0))
        _parterre(g, xa, -12800, xb, -10200)
        _parterre(g, xa, -9300, xb, -7000)
        for y in (-7200, -8800, -10500, -12200):
            _topiary(g, sx * 1350, y, rng.uniform(650, 850), rng)
        for (x, y) in ((sx * 1700, -6800), (sx * 1700, -13200)):
            box(g, (x, y, 60), (140, 140, 120), 0.0, (0.84, 0.84, 0.84))
            ico(g, (x, y, 330), 150, (0.78, 0.78, 0.80), rng, sub=1, jit=0.3, squash=1.6)
        for y in np.arange(-6800, -15000, -1600):
            box(g, (sx * 6700, y, 300), (16, 16, 360), 0.0, (0.12, 0.12, 0.14))
            ico(g, (sx * 6700, y, 700), 55, (1.0, 0.88, 0.62), rng, sub=0, jit=0.0, emis=1.8, kind=2)
    _fountain(g, 0, -9950, rng)
    for x in np.arange(-7200, 7300, 1200):
        _plane_tree(g, x + rng.uniform(-100, 100), -15000 + rng.uniform(-150, 150), rng.uniform(1300, 1700), rng)
    # ---- the Champ de Mars behind the orange goal, leading to the tower ----
    g.quads((-3200, 7300, -48), (3200, 7300, -48), (3200, 24000, -48), (-3200, 24000, -48), (0.13, 0.33, 0.10))
    for sx in (-1, 1):
        g.quads((sx * 3200, 7300, -54), (sx * 4600, 7300, -54), (sx * 4600, 24000, -54), (sx * 3200, 24000, -54), (0.62, 0.55, 0.46))
        for y in np.arange(7200, 23800, 950):
            _plane_tree(g, sx * (5000 + rng.uniform(-60, 60)), y, rng.uniform(1200, 1500), rng)
            _plane_tree(g, sx * (6100 + rng.uniform(-60, 60)), y + 475, rng.uniform(1200, 1500), rng)
        for y in np.arange(7000, 24000, 2000):
            box(g, (sx * 3400, y, 280), (14, 14, 340), 0.0, (0.12, 0.12, 0.14))
            ico(g, (sx * 3400, y, 650), 50, (1.0, 0.88, 0.62), rng, sub=0, jit=0.0, emis=1.8, kind=2)
    _eiffel(g, 0.0, 28500.0, 26000.0, day=True)
    # ---- the city: varied Haussmann blocks around, a few landmarks ----
    placed, tries = [], 0
    while len(placed) < 170 and tries < 6000:
        tries += 1
        r = rng.uniform(15500, 42000); a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if abs(x) < 7500 and 5500 < y < 36000:                  # Champ de Mars + tower
            continue
        if abs(x) < 9000 and -17500 < y < 0:                    # garden
            continue
        hx, hy = rng.uniform(700, 1700), rng.uniform(600, 1400)
        if any(abs(x - px) < hx + phx + 450 and abs(y - py) < hy + phy + 450 for px, py, phx, phy in placed):
            continue
        placed.append((x, y, hx, hy))
        _building(g, x, y, hx, hy, rng.uniform(1500, 2900), rng)
    for _ in range(80):                                                                    # street trees
        r = rng.uniform(13000, 30000); a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if any(abs(x - px) < phx + 300 and abs(y - py) < phy + 300 for px, py, phx, phy in placed):
            continue
        _plane_tree(g, x, y, rng.uniform(1100, 1500), rng)
    # Les Invalides: a golden ribbed dome on a drum
    ic = (-19000.0, 16000.0)
    box(g, (ic[0], ic[1], 700), (2600, 2000, 760), 0.0, (0.80, 0.75, 0.64), kind=3, top=(0.25, 0.27, 0.32))
    frustum(g, (ic[0], ic[1], 1460), 1100, 1100, 0, 900, 16, (0.80, 0.75, 0.64))
    lathe(g, (ic[0], ic[1], 2360), [(1150, 0), (1100, 500), (900, 950), (520, 1300), (0, 1450)], 16,
          lambda nz, zf, q: (0.95, 0.72, 0.28) if q < 0.5 else (0.26, 0.28, 0.33), emis=0.12)
    frustum(g, (ic[0], ic[1], 3810), 160, 20, 0, 1100, 8, (1.0, 0.78, 0.30), emis=0.3)
    # Sacre-Coeur on its hill, far away
    lathe(g, (9000, 41000, -60), [(9000, 0), (6500, 900), (3500, 2100), (0, 2500)], 14,
          lambda nz, zf, q: (0.12 + 0.03 * q, 0.20 + 0.03 * q, 0.10), rng=rng, jitter=0.06)
    bc = (9000, 41000, 2380)
    box(g, (bc[0], bc[1], bc[2] + 600), (1500, 900, 600), 0.0, (0.92, 0.90, 0.84), emis=0.15)
    lathe(g, (bc[0], bc[1], bc[2] + 1200), [(700, 0), (700, 500), (650, 900), (480, 1350), (220, 1650), (0, 1780)], 12,
          lambda nz, zf, q: (0.94, 0.92, 0.86), emis=0.15)
    for dx, dy in ((-1000, -500), (1000, -500), (-1000, 500), (1000, 500)):
        lathe(g, (bc[0] + dx, bc[1] + dy, bc[2] + 1200), [(280, 0), (280, 200), (220, 450), (0, 620)], 10,
              lambda nz, zf, q: (0.94, 0.92, 0.86), emis=0.15)
    # the Arc de Triomphe
    ac = (21000.0, 24000.0)
    for sx in (-1, 1):
        q = _xf([(sx * 1100, 0, 0)], (ac[0], ac[1], 0), 0.4)[0]
        box(g, (q[0], q[1], 1400), (650, 1000, 1460), 0.4, (0.85, 0.80, 0.68))
    box(g, (ac[0], ac[1], 2560), (1750, 1000, 300), 0.4, (0.85, 0.80, 0.68), emis=0.1)
    return g.array(), np.concatenate(eggs).astype("f4")


# ------------------------------------------------------------------------------------------------ space
def _asteroid(g, c, r, col, rng, sub):
    V, F = _ico_base(sub)
    disp = np.ones(len(V))
    for k in range(7):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        disp += (0.16 / (1.0 + 0.6 * k)) * np.cos(rng.uniform(1.5, 3.5) * (1 + k * 0.6) * (V @ d) * 2.0 + rng.uniform(0, 6.3))
    P = V * disp[:, None] * r * rng.uniform(0.65, 1.0, 3)[None, :]
    th = rng.uniform(0, 2 * math.pi)
    ca, sa = math.cos(th), math.sin(th)
    P = np.stack([P[:, 0] * ca - P[:, 2] * sa, P[:, 1], P[:, 0] * sa + P[:, 2] * ca], 1) + np.asarray(c)
    g.tris(P[F], col, 0.0, 10)


def _cyl_y(g, cx, cz, y0, y1, r0, r1, n, col, emis=0.0, kind=0, cap=None):
    """A cylinder / cone along the y axis (engines, sensors); cap = colour of a disc at y1 (None = open)."""
    a = np.arange(n + 1) * 2 * math.pi / n
    A = np.stack([cx + r0 * np.cos(a), np.full(n + 1, y0), cz + r0 * np.sin(a)], 1)
    B = np.stack([cx + r1 * np.cos(a), np.full(n + 1, y1), cz + r1 * np.sin(a)], 1)
    g.quads(A[:-1], A[1:], B[1:], B[:-1], col, emis, kind)
    if cap is not None:
        ctr = np.tile([cx, y1, cz], (n, 1))
        g.tris(np.stack([B[:-1], B[1:], ctr], 1), cap[0], cap[1], cap[2])


def build_space(seed=31):
    """The arena on the flight deck of a star cruiser in orbit. From the pitch it reads as a ship: armoured hull
    flanks rise along both sides of the deck (ribs, running lights), the prow narrows to a point ahead past the
    orange goal, the bridge is a raked wedge at the stern with two big engine nacelles on pylons behind it."""
    rng = np.random.default_rng(seed)
    g = G()
    deck, hullc, dark, cyan = (0.10, 0.11, 0.13), (0.46, 0.48, 0.53), (0.09, 0.10, 0.12), (0.30, 0.90, 1.00)
    ys = np.array([-21000, -17000, -12000, -6000, 0, 6000, 12000, 18000, 24000, 30000, 36000, 41000], "f8")
    hws = np.array([9000, 9800, 10300, 10500, 10500, 10300, 9600, 8200, 6300, 4000, 1700, 120], "f8")
    FL_IN, FL_H, FL_OUT = 700.0, 2600.0, 250.0            # flank: inset of its foot, height, overhang of its top

    def hw_at(y):
        return float(np.interp(y, ys, hws))

    # deck + lower hull (seen from far / below)
    for i in range(len(ys) - 1):
        ya, yb = ys[i], ys[i + 1]
        wa, wb = hws[i], hws[i + 1]
        g.quads((-wa, ya, -60), (wa, ya, -60), (wb, yb, -60), (-wb, yb, -60), deck, 0.0, 15)
        for sx in (-1, 1):
            g.quads((sx * wa, ya, -60), (sx * wb, yb, -60), (sx * wb * 0.6, yb, -3200), (sx * wa * 0.6, ya, -3200), dark, 0.0, 15)
            g.quads((sx * wa * 0.6, ya, -3200), (sx * wb * 0.6, yb, -3200), (0, yb, -3900), (0, ya, -3900), dark, 0.0, 5)
            # the armoured flank: a raked wall from inside the deck edge up and out, a top rail, the outer face
            fa0, fb0 = (sx * (wa - FL_IN), ya, -60), (sx * (wb - FL_IN), yb, -60)
            fa1, fb1 = (sx * (wa + FL_OUT), ya, FL_H * min(1.0, wa / 6000.0)), (sx * (wb + FL_OUT), yb, FL_H * min(1.0, wb / 6000.0))
            g.quads(fa0, fb0, fb1, fa1, hullc, 0.0, 15)
            g.quads(fa1, fb1, (fb1[0] + sx * 260, yb, fb1[2] - 200), (fa1[0] + sx * 260, ya, fa1[2] - 200), dark, 0.0, 5)
            g.quads((fa1[0] + sx * 260, ya, fa1[2] - 200), (fb1[0] + sx * 260, yb, fb1[2] - 200),
                    (sx * wb, yb, -60), (sx * wa, ya, -60), hullc, 0.0, 15)
            # running lights along the flank top + a neon line at its foot
            g.quads((fa1[0] - sx * 6, ya, fa1[2] - 90), (fb1[0] - sx * 6, yb, fb1[2] - 90), (fb1[0] - sx * 6, yb, fb1[2] - 40),
                    (fa1[0] - sx * 6, ya, fa1[2] - 40), cyan, 2.0, 7)
            g.quads((fa0[0] - sx * 4, ya, -58), (fb0[0] - sx * 4, yb, -58), (fb0[0] - sx * 34, yb, -58), (fa0[0] - sx * 34, ya, -58), cyan, 1.6, 7)
    # livery: a painted stripe along the inside of both flanks, gun turrets on the flank tops
    for i in range(len(ys) - 1):
        ya, yb = ys[i], ys[i + 1]
        if yb > 34000:
            continue
        for sx in (-1, 1):
            wa, wb = hws[i], hws[i + 1]
            ta, tb = FL_H * min(1.0, wa / 6000.0), FL_H * min(1.0, wb / 6000.0)
            for f0, f1, col in ((0.62, 0.70, (0.72, 0.12, 0.10)), (0.72, 0.76, (0.85, 0.85, 0.88))):
                pa0 = (sx * (wa - FL_IN + (FL_IN + FL_OUT) * f0 - 14), ya, -60 + (ta + 60) * f0)
                pa1 = (sx * (wa - FL_IN + (FL_IN + FL_OUT) * f1 - 14), ya, -60 + (ta + 60) * f1)
                pb0 = (sx * (wb - FL_IN + (FL_IN + FL_OUT) * f0 - 14), yb, -60 + (tb + 60) * f0)
                pb1 = (sx * (wb - FL_IN + (FL_IN + FL_OUT) * f1 - 14), yb, -60 + (tb + 60) * f1)
                g.quads(pa0, pb0, pb1, pa1, col)
    for y in (-9000.0, -2000.0, 5000.0, 12000.0, 19000.0):
        w = hw_at(y)
        for sx in (-1, 1):
            top = FL_H * min(1.0, w / 6000.0)
            cx_ = sx * (w + FL_OUT * 0.5)
            lathe(g, (cx_, y, top - 40), [(620, 0), (600, 180), (380, 420), (0, 480)], 14, lambda nz, zf, q: (0.40, 0.42, 0.46), kind=5)
            for bx in (-150, 150):
                _cyl_y(g, cx_ + bx, top + 250, y + 200, y + 2400, 70, 55, 8, (0.28, 0.30, 0.34), kind=5)
    # ribs on the inside of the flanks, every 3000 uu, with a light at the top
    for y in np.arange(-15000, 30001, 3000):
        w = hw_at(y)
        for sx in (-1, 1):
            top = FL_H * min(1.0, w / 6000.0)
            p0 = (sx * (w - FL_IN - 30), y, -60); p1 = (sx * (w + FL_OUT - 30), y, top)
            g.quads((p0[0], y - 120, p0[2]), (p0[0], y + 120, p0[2]), (p1[0], y + 120, p1[2]), (p1[0], y - 120, p1[2]), (0.30, 0.32, 0.36), 0.0, 5)
            box(g, (p1[0] - sx * 40, y, top - 60), (30, 60, 30), 0.0, (1.0, 0.25, 0.15) if sx < 0 else (0.3, 1.0, 0.4), emis=2.4, kind=2)
    # deck markings: guide lines and chevrons toward the prow, hazard bands at the stern
    for sx in (-1, 1):
        g.quads((sx * 5200, -14000, -57), (sx * 5320, -14000, -57), (sx * 3500, 34000, -57), (sx * 3380, 34000, -57), (0.85, 0.75, 0.2), 0.6, 0)
    for y in np.arange(8500, 32000, 2200):
        for sx in (-1, 1):
            g.quads((0, y + 700, -57), (sx * 1600, y, -57), (sx * 1600, y + 180, -57), (0, y + 880, -57), cyan, 1.4, 7)
    # the bridge: a raked wedge at the stern with a wide window band
    by0, by1, bw = -19500.0, -13500.0, 4200.0
    for sx in (-1, 1):
        g.quads((sx * bw, by1, -60), (sx * bw, by0, -60), (sx * bw * 0.8, by0, 3800), (sx * bw * 0.75, by1 - 2500, 3400), hullc, 0.0, 15)
    g.quads((-bw, by1, -60), (bw, by1, -60), (bw * 0.75, by1 - 2500, 3400), (-bw * 0.75, by1 - 2500, 3400), hullc, 0.0, 15)
    g.quads((-bw * 0.75, by1 - 2500, 3400), (bw * 0.75, by1 - 2500, 3400), (bw * 0.8, by0, 3800), (-bw * 0.8, by0, 3800), dark, 0.0, 5)
    for k in range(2):                                                      # window bands on the raked front
        f0, f1 = 0.62 + k * 0.2, 0.70 + k * 0.2
        za, zb = 3400 * f0, 3400 * f1
        ya_, yb_ = by1 - 2500 * f0 + 12, by1 - 2500 * f1 + 12
        g.quads((-bw * (1 - 0.25 * f0) * 0.85, ya_, za), (bw * (1 - 0.25 * f0) * 0.85, ya_, za),
                (bw * (1 - 0.25 * f1) * 0.85, yb_, zb), (-bw * (1 - 0.25 * f1) * 0.85, yb_, zb), (0.75, 0.93, 1.0), 1.5, 2)
    for ax, h in ((-1800, 2000), (1400, 1500)):                         # antenna masts
        box(g, (ax, by0 + 800, 3800 + h / 2), (30, 30, h / 2), 0.0, (0.5, 0.52, 0.56))
        box(g, (ax, by0 + 800, 3800 + h + 30), (45, 45, 45), 0.0, (1.0, 0.15, 0.1), emis=2.6, kind=2)
    # a radar dish on the bridge roof, tilted up
    lathe(g, (0.0, by0 + 1800, 4100), [(0, 0), (500, 70), (1000, 250), (1350, 520)], 16, lambda nz, zf, q: (0.78, 0.80, 0.84), kind=5)
    frustum(g, (0.0, by0 + 1800, 3800), 160, 120, 0, 320, 8, (0.45, 0.47, 0.5))
    # two engine nacelles on pylons behind the bridge (seen past the blue goal) + the main engine block
    for sx in (-1, 1):
        ex, ez = sx * 7200.0, 2600.0
        _cyl_y(g, ex, ez, -12000, -24000, 1500, 1500, 20, hullc, kind=5)
        _cyl_y(g, ex, ez, -12000, -11000, 1500, 400, 20, hullc, kind=5, cap=((0.3, 0.32, 0.36), 0.0, 5))
        _cyl_y(g, ex, ez, -24000, -24400, 1500, 1250, 20, (0.2, 0.21, 0.24), kind=5, cap=((0.55, 0.85, 1.0), 3.0, 2))
        _cyl_y(g, ex, ez, -24380, -24420, 1600, 1600, 20, (0.45, 0.8, 1.0), emis=2.4, kind=7)
        g.quads((sx * 5000, -20000, -60), (sx * 5000, -15000, -60), (ex - sx * 900, -15000, ez), (ex - sx * 900, -20000, ez), dark, 0.0, 5)   # pylon
    for ex, ez, r in ((0.0, -1700.0, 1800.0), (-4600.0, -1500.0, 1300.0), (4600.0, -1500.0, 1300.0)):
        _cyl_y(g, ex, ez, ys[0], ys[0] - 2200, r * 0.95, r * 1.15, 20, (0.22, 0.23, 0.26), kind=5)
        _cyl_y(g, ex, ez, ys[0] - 2200, ys[0] - 2100, r * 1.15, r * 0.9, 20, (0.3, 0.32, 0.36), kind=5, cap=((0.55, 0.85, 1.0), 3.0, 2))
    A = [(hws[0], -60), (hws[0] * 0.6, -3200), (0, -3900)]
    for sx in (-1, 1):
        g.quads((sx * A[0][0], ys[0], A[0][1]), (sx * A[1][0], ys[0], A[1][1]), (0, ys[0], A[1][1]), (0, ys[0], A[0][1]), dark, 0.0, 5)
    # tail fins on the nacelles
    for sx in (-1, 1):
        base = [(sx * 7200, -23000, 4100), (sx * 7200, -18000, 4100), (sx * 7600, -21500, 8200), (sx * 7600, -24200, 8400)]
        g.quads(*base, hullc, 0.0, 15)
        g.quads((base[1][0] - sx * 20, base[1][1] + 10, base[1][2]), (base[1][0] + sx * 20, base[1][1] + 10, base[1][2]),
                (base[2][0] + sx * 20, base[2][1] + 10, base[2][2]), (base[2][0] - sx * 20, base[2][1] + 10, base[2][2]), cyan, 1.8, 7)
        box(g, (base[3][0], base[3][1], base[3][2] + 60), (60, 60, 60), 0.0, (1.0, 0.2, 0.1), emis=2.6, kind=2)
    # the prow: a sensor spike + a light at its tip
    _cyl_y(g, 0.0, 400.0, ys[-1] - 800, ys[-1] + 4500, 340, 20, 8, (0.5, 0.52, 0.56), kind=5)
    box(g, (0.0, ys[-1] + 4500, 400), (60, 60, 60), 0.0, (1.0, 1.0, 1.0), emis=3.0, kind=2)
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
    # an asteroid belt the ship flies past, and a station farther out
    tilt = math.radians(14.0)
    placed = 0
    while placed < 100:                                  # beside and below the ship, never in front of it
        th = rng.uniform(0, 2 * math.pi)
        if abs(math.cos(th)) < 0.55:
            continue
        r = rng.uniform(40000, 64000)
        x, y = r * math.cos(th), r * math.sin(th)
        z = -5000 + y * math.tan(tilt) * 0.15 + rng.normal(0, 2200)
        placed += 1
        big = rng.random() > 0.72
        sz = rng.uniform(1500, 3800) if big else rng.uniform(300, 1000)
        col = (0.34, 0.31, 0.29) if rng.random() < 0.6 else (0.42, 0.35, 0.28)
        _asteroid(g, (x, y, z), sz, col, rng, 3 if big else 2)
    sc = np.array([-36000.0, 42000.0, 16000.0])
    frustum(g, (sc[0], sc[1], sc[2] - 3500), 1400, 1400, 0, 7000, 12, (0.72, 0.74, 0.78), kind=5)
    frustum(g, (sc[0], sc[1], sc[2] + 3500), 1400, 400, 0, 900, 12, (0.72, 0.74, 0.78), kind=5)
    frustum(g, (sc[0], sc[1], sc[2] - 3500), 400, 1400, -900, 0, 12, (0.72, 0.74, 0.78), kind=5, cap=False)
    for k in range(36):
        a = 2 * math.pi * k / 36
        box(g, (sc[0] + 7000 * math.cos(a), sc[1] + 7000 * math.sin(a), sc[2]), (420, 640, 420), a, (0.80, 0.82, 0.86), kind=15)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        box(g, (sc[0] + 4200 * math.cos(a), sc[1] + 4200 * math.sin(a), sc[2]), (2800, 90, 90), a, (0.6, 0.62, 0.66), kind=5)
    for sz_ in (-1, 1):
        for sx in (-1, 1):
            box(g, (sc[0] + sx * 6000, sc[1], sc[2] + sz_ * 4200), (4200, 1200, 20), 0.0, (0.10, 0.18, 0.42), kind=5)
            box(g, (sc[0] + sx * 1900, sc[1], sc[2] + sz_ * 4200), (500, 60, 60), 0.0, (0.6, 0.62, 0.66))
    return g.array(), np.zeros((0, 8), "f4")                  # no crowd aboard


# ------------------------------------------------------------------------------------------------ valley extras
def _gable_house(g, x, y, z, hx, hy, h, yaw, rng, wall=None, roofc=None):
    wall = wall or [(0.86, 0.82, 0.72), (0.80, 0.74, 0.62), (0.90, 0.88, 0.84), (0.74, 0.60, 0.48)][rng.integers(0, 4)]
    roofc = roofc or [(0.42, 0.16, 0.12), (0.30, 0.22, 0.18), (0.25, 0.27, 0.32)][rng.integers(0, 3)]
    box(g, (x, y, z + h / 2), (hx, hy, h / 2), yaw, wall, kind=3, top=wall)
    rh = hy * 0.9
    P = _xf([(-hx - 60, -hy - 60, h), (hx + 60, -hy - 60, h), (hx + 60, 0, h + rh), (-hx - 60, 0, h + rh),
             (-hx - 60, hy + 60, h), (hx + 60, hy + 60, h)], (x, y, z), yaw)
    g.quads(P[0], P[1], P[2], P[3], roofc, 0.0, 12)
    g.quads(P[4], P[5], P[2], P[3], roofc, 0.0, 12)
    for sx in (-1, 1):                                                    # gable ends
        G3 = _xf([(sx * hx, -hy, h), (sx * hx, hy, h), (sx * hx, 0, h + rh * 0.95)], (x, y, z), yaw)
        g.tris(np.array([G3]), wall)
    c = _xf([(hx * 0.5, hy * 0.4, h + rh * 0.8)], (x, y, z), yaw)[0]
    box(g, (c[0], c[1], c[2]), (45, 45, rh * 0.5), yaw, (0.45, 0.35, 0.30))


def _balloon(g, x, y, z, s, rng):
    cols = [(0.95, 0.30, 0.20), (0.98, 0.80, 0.25), (0.25, 0.55, 0.95), (0.95, 0.95, 0.92), (0.40, 0.80, 0.40), (0.85, 0.35, 0.75)]
    ca, cb = cols[rng.integers(0, 6)], cols[rng.integers(0, 6)]
    prof = [(0.0, 0.0), (0.22, 0.10), (0.47, 0.32), (0.55, 0.58), (0.50, 0.80), (0.33, 0.95), (0.0, 1.0)]
    n = 16
    ph = float(rng.random())
    for i in range(len(prof) - 1):
        (r0, z0), (r1, z1) = prof[i], prof[i + 1]
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            p = [(x + s * r0 * math.cos(a0), y + s * r0 * math.sin(a0), z + s * z0), (x + s * r0 * math.cos(a1), y + s * r0 * math.sin(a1), z + s * z0),
                 (x + s * r1 * math.cos(a1), y + s * r1 * math.sin(a1), z + s * z1), (x + s * r1 * math.cos(a0), y + s * r1 * math.sin(a0), z + s * z1)]
            g.quads(*p, ca if k % 2 else cb, ph, 14)
    box(g, (x, y, z - s * 0.18), (s * 0.08, s * 0.08, s * 0.06), 0.0, (0.40, 0.28, 0.16), emis=ph, kind=14)   # basket
    box(g, (x, y, z - s * 0.06), (s * 0.03, s * 0.03, s * 0.03), 0.0, (1.0, 0.6, 0.2), emis=ph, kind=9)      # burner


def _road(g, pts, width, L, col=(0.42, 0.36, 0.30), step_=300.0, lift=22.0):
    """A dirt road through the polyline pts, draped on the terrain (landscape._height)."""
    P = []
    for (a, b) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // step_))
        for k in range(n):
            t = k / n
            P.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    P.append(pts[-1])
    for (a, b) in zip(P[:-1], P[1:]):
        d = np.array([b[0] - a[0], b[1] - a[1]]); d /= np.linalg.norm(d) + 1e-9
        nrm = np.array([-d[1], d[0]]) * width / 2
        za, zb = L._height(*a) + lift, L._height(*b) + lift
        g.quads((a[0] - nrm[0], a[1] - nrm[1], za), (a[0] + nrm[0], a[1] + nrm[1], za),
                (b[0] + nrm[0], b[1] + nrm[1], zb), (b[0] - nrm[0], b[1] - nrm[1], zb), col)
    return P


def build_valley(seed=41):
    """The evening valley's settlement, laid out as one place: a road leaves the plaza, follows the lake shore and
    becomes the main street of a village (houses facing the street, a church on the square, a path down to a pier);
    farm fields with a windmill and a farmhouse beyond the village; the castle on the hill above it; a few hot-air
    balloons flying together over the lake. landscape.py keeps its trees out of all of these (clear_zone)."""
    import landscape as L
    rng = np.random.default_rng(seed)
    g = G()
    walls = [(0.88, 0.84, 0.74), (0.84, 0.78, 0.66), (0.90, 0.88, 0.82)]
    roofs = [(0.44, 0.17, 0.12), (0.38, 0.15, 0.11), (0.48, 0.22, 0.14)]
    road = _road(g, L.ROAD, 420.0, L)
    for k in range(1, len(road) - 1, 4):                                   # lamps along the road
        a, b = road[k - 1], road[k + 1]
        d = np.array([b[0] - a[0], b[1] - a[1]]); d /= np.linalg.norm(d) + 1e-9
        x, y = road[k][0] - d[1] * 300, road[k][1] + d[0] * 300
        z = L._height(x, y)
        box(g, (x, y, z + 230), (12, 12, 280), 0.0, (0.15, 0.13, 0.12))
        ico(g, (x, y, z + 520), 36, (1.0, 0.80, 0.50), rng, sub=0, jit=0.0, emis=1.8, kind=2)
    # the main street: the road's last leg carried on through the village; houses on both sides face it
    s0, s1 = np.array(L.ROAD[-2]), np.array(L.STREET_END)
    sd = (s1 - s0) / np.linalg.norm(s1 - s0)
    sn = np.array([-sd[1], sd[0]])
    _road(g, [tuple(s0), tuple(s1)], 420.0, L)
    yaw_st = math.atan2(sd[1], sd[0])
    length = np.linalg.norm(s1 - s0)
    for side in (-1, 1):
        t = 500.0
        while t < length - 300:
            c = s0 + sd * t + sn * side * (720 + rng.uniform(-40, 40))
            if np.linalg.norm(c - np.array(L.VILLAGE)) < 900:            # leave the square open
                t += 700
                continue
            hy = rng.uniform(260, 330)
            _gable_house(g, c[0], c[1], L._height(*c) - 30, rng.uniform(320, 420), hy, rng.uniform(420, 560),
                         yaw_st, rng, wall=walls[rng.integers(0, 3)], roofc=roofs[rng.integers(0, 3)])
            t += rng.uniform(850, 1000)
    # the church on the square, its spire the village landmark
    vc = np.array(L.VILLAGE)
    cc = vc + sn * 1300
    cz = L._height(*cc) - 30
    _gable_house(g, cc[0], cc[1], cz, 720, 380, 900, yaw_st + math.pi / 2, rng, wall=(0.90, 0.88, 0.82), roofc=(0.28, 0.30, 0.34))
    t = _xf([(-780, 0, 0)], (cc[0], cc[1], cz), yaw_st + math.pi / 2)[0]
    box(g, (t[0], t[1], cz + 800), (230, 230, 800), yaw_st + math.pi / 2, (0.90, 0.88, 0.82), kind=3)
    frustum(g, (t[0], t[1], cz + 1600), 330, 10, 0, 1300, 4, (0.28, 0.30, 0.34), rot=yaw_st + math.pi / 2 + math.pi / 4, kind=12)
    frustum(g, (vc[0], vc[1], L._height(*vc) - 20), 160, 160, 0, 120, 10, (0.55, 0.53, 0.50))   # the well
    # a path from the square down to the pier on the lake
    LC, LR = L.LAKE_C, L.LAKE_R
    ang = math.atan2(vc[1] - LC[1], vc[0] - LC[0])
    shore = np.array([LC[0] + LR[0] * 1.02 * math.cos(ang), LC[1] + LR[1] * 1.02 * math.sin(ang)])
    _road(g, [tuple(vc), tuple(shore + (vc - shore) / np.linalg.norm(vc - shore) * 250)], 300.0, L)
    d = (np.array(LC) - shore); d /= np.linalg.norm(d)
    for k in range(10):
        c = shore + d * (k * 180)
        box(g, (c[0], c[1], L.LAKE_Z + 50), (95, 130, 12), math.atan2(d[1], d[0]), (0.42, 0.30, 0.20))
        if k % 3 == 0:
            box(g, (c[0] + d[1] * 110, c[1] - d[0] * 110, L.LAKE_Z + 140), (10, 10, 90), 0.0, (0.2, 0.15, 0.1))
            ico(g, (c[0] + d[1] * 110, c[1] - d[0] * 110, L.LAKE_Z + 250), 32, (1.0, 0.75, 0.40), rng, sub=0, jit=0.0, emis=2.0, kind=2)
    # farm fields beyond the village, the windmill and a farmhouse at their edge
    fc = np.array(L.FIELDS)
    crops = [(0.62, 0.52, 0.20), (0.26, 0.40, 0.12), (0.36, 0.26, 0.16), (0.55, 0.50, 0.22), (0.30, 0.44, 0.14), (0.40, 0.30, 0.18)]
    k = 0
    for i in (-1, 0, 1):
        for j in (-0.5, 0.5):
            c = fc + sd * (i * 1550) + sn * (j * 1150)
            hx, hy = 720, 540
            corners = [c + sd * a * hx + sn * b * hy for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            P = [(p[0], p[1], L._height(*p) + 14) for p in corners]
            g.quads(*P, crops[k % len(crops)])
            k += 1
    wm = fc + sd * 2700 + sn * 400
    wx, wy = float(wm[0]), float(wm[1])
    wz = L._height(wx, wy) - 20
    frustum(g, (wx, wy, wz), 520, 330, 0, 1700, 10, (0.86, 0.84, 0.78), kind=3)
    frustum(g, (wx, wy, wz + 1700), 420, 0, 0, 550, 10, (0.40, 0.22, 0.16), kind=12)
    face = -sn
    hub = np.array([wx, wy]) + face * 420
    e = np.array([face[1], -face[0]])
    for q in range(4):
        a = math.pi / 4 + q * math.pi / 2 + 0.2
        tip = np.array([math.cos(a), math.sin(a)])
        ww = np.array([-tip[1], tip[0]]) * 150
        p0 = (hub[0] + e[0] * tip[0] * 150, hub[1] + e[1] * tip[0] * 150, wz + 1650 + tip[1] * 150)
        p1 = (hub[0] + e[0] * tip[0] * 1500, hub[1] + e[1] * tip[0] * 1500, wz + 1650 + tip[1] * 1500)
        q0 = (p0[0] + e[0] * ww[0], p0[1] + e[1] * ww[0], p0[2] + ww[1])
        q1 = (p1[0] + e[0] * ww[0], p1[1] + e[1] * ww[0], p1[2] + ww[1])
        g.quads(p0, p1, q1, q0, (0.85, 0.82, 0.74))
    fh = fc - sd * 2600 + sn * 900
    _gable_house(g, fh[0], fh[1], L._height(*fh) - 30, 520, 320, 480, yaw_st, rng, wall=(0.80, 0.72, 0.58), roofc=(0.44, 0.17, 0.12))
    # the castle on the hill above the village, a path up to its gate
    kx, ky = L.CASTLE
    kz = L._height(kx, ky) - 30
    lathe(g, (kx, ky, kz - 200), [(3200, 0), (2500, 500), (1700, 900), (0, 1000)], 16,
          lambda nz, zf, q: (0.12 + 0.03 * q, 0.22 + 0.03 * q, 0.08), rng=rng, jitter=0.06)
    _road(g, [tuple(vc + sd * 800), ((vc[0] + kx) / 2 + 500, (vc[1] + ky) / 2), (kx, ky - 1700)], 260.0, L, lift=30.0)
    kz += 780
    stone = (0.55, 0.53, 0.50)
    for i in range(4):
        a0 = math.pi / 4 + i * math.pi / 2
        a1 = a0 + math.pi / 2
        p0 = (kx + 1400 * math.cos(a0), ky + 1400 * math.sin(a0)); p1 = (kx + 1400 * math.cos(a1), ky + 1400 * math.sin(a1))
        m = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        box(g, (m[0], m[1], kz + 350), (990, 110, 350), a0 + math.pi / 4 + math.pi / 2, stone)
        for q in range(8):
            t_ = (q + 0.5) / 8
            qq = (p0[0] + (p1[0] - p0[0]) * t_, p0[1] + (p1[1] - p0[1]) * t_)
            box(g, (qq[0], qq[1], kz + 750), (50, 50, 50), a0 + math.pi / 4, stone)
        frustum(g, (p0[0], p0[1], kz), 330, 300, 0, 1200, 12, stone)
        frustum(g, (p0[0], p0[1], kz + 1200), 380, 0, 0, 700, 12, (0.36, 0.22, 0.20), kind=12)
    box(g, (kx, ky, kz + 900), (650, 650, 900), 0.3, stone, kind=3, top=stone)
    frustum(g, (kx + 350, ky + 350, kz + 1800), 260, 240, 0, 900, 12, stone, kind=3)
    frustum(g, (kx + 350, ky + 350, kz + 2700), 320, 0, 0, 800, 12, (0.36, 0.22, 0.20), kind=12)
    box(g, (kx + 350, ky + 350, kz + 3700), (6, 6, 300), 0.0, (0.2, 0.2, 0.2))
    box(g, (kx + 450, ky + 350, kz + 3900), (100, 4, 70), 0.0, (0.85, 0.15, 0.12), emis=0.3)
    # three hot-air balloons flying together over the lake
    for (dx, dy, dz) in ((0, 0, 0), (1800, 900, 500), (-1200, 1500, -400)):
        _balloon(g, LC[0] + 800 + dx, LC[1] + 1200 + dy, 4200 + dz, 850, rng)
    return g.array(), np.zeros((0, 8), "f4")


BUILDERS = {"valley": build_valley, "temple": build_temple, "paris": build_paris, "space": build_space}


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

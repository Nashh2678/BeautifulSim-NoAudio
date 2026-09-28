"""Maps: the scenery around the arena, its sky and light, the field style, and the crowd.

  valley  the original low-poly evening valley (landscape.py, unchanged)
  temple  "Forbidden Temple"-style: pink dusk, karst peaks, pagodas, a paifang gate, cherry trees, lanterns
  paris   "Parc de Paris"-style: violet dusk, the Eiffel Tower down the Champ de Mars, Haussmann blocks, the Seine,
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
SCENE_VERSION = 12


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
                   cloudB=(0.42, 0.18, 0.40, 0.72), haze=(9000.0, 60000.0, 0.72), glass=1.0, stars=0.35),
    "paris": dict(sun_dir=_n((-0.55, -0.62, 0.16)), sun_col=(0.92, 0.80, 0.96), zen=(0.05, 0.04, 0.15),
                  mid=(0.28, 0.18, 0.42), hor=(0.82, 0.52, 0.72), glow=(0.95, 0.55, 0.75), ground=(0.04, 0.03, 0.06),
                  amb=(0.95, 0.90, 1.08), grass=(0.075, 0.25, 0.060), cloudA=(0.85, 0.60, 0.86, 0.40),
                  cloudB=(0.26, 0.18, 0.40, 0.70), haze=(9000.0, 60000.0, 0.65), glass=1.0, stars=0.75),
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
            off = w * 0.5 + 14
            cx, cy = x + off * math.cos(a), y + off * math.sin(a)
            box(g, (cx, cy, z + th * 0.5), (6, w * 0.30, th * 0.26), a, (1.0, 0.55, 0.25), emis=1.3, kind=2)
        z += th
        roof(g, (x, y), w * 0.5, w * 0.5, z, th * 0.55, w * 0.30, th * 0.38, yaw, roofc, soffit=soff)
        box(g, (x, y, z - 30), (w * 0.5 + 12, w * 0.5 + 12, 18), yaw, gold, emis=0.25)   # gilded band
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
        p = _xf([(0, side * (D / 2 + 14), 0)], (x, y, 0), yaw)[0]
        box(g, (p[0], p[1], 180 + H * 0.45), (W / 2 - 250, 6, H * 0.3), yaw, paper, emis=1.0, kind=2)
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
EIFFEL_LIGHT = (0.55, 0.72, 1.00)


def _eiffel(g, cx, cy, H, light=EIFFEL_LIGHT):
    """The Eiffel Tower in its real proportions: four legs on a concave exponential profile joined by the big
    arches under the first floor, merging at the second floor into one shaft up to the top floor and the antenna.
    Every face is see-through iron lattice (SCENE kind 4) lit in `light`; lit edges run up the four corners."""
    def w(z):                                   # half width of the outer profile
        return H * (0.017 + 0.176 * math.exp(-3.7 * z / H))

    def t(z):                                   # leg thickness (the legs merge near the second floor)
        return H * 0.075 * (1.0 - 0.35 * z / (0.357 * H))
    z2 = 0.357 * H
    zs = np.linspace(0.0, z2, 12)
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for z0, z1 in zip(zs[:-1], zs[1:]):
            w0, w1 = w(z0), w(z1)
            i0, i1 = max(w0 - t(z0), 0.0), max(w1 - t(z1), 0.0)

            def P(z, a, b):
                return (cx + sx * a, cy + sy * b, z - 60.0)
            g.quads(P(z0, w0, i0), P(z0, w0, w0), P(z1, w1, w1), P(z1, w1, i1), light, 0.0, 4)   # outer x
            g.quads(P(z0, i0, w0), P(z0, w0, w0), P(z1, w1, w1), P(z1, i1, w1), light, 0.0, 4)   # outer y
            g.quads(P(z0, i0, i0), P(z0, i0, w0), P(z1, i1, w1), P(z1, i1, i1), light, 0.0, 4)   # inner x
            g.quads(P(z0, i0, i0), P(z0, w0, i0), P(z1, w1, i1), P(z1, i1, i1), light, 0.0, 4)   # inner y
    # the arches between the legs under the first floor
    z1f = 0.178 * H
    for side in range(4):
        yaw = side * math.pi / 2
        ia = w(0.06 * H) - t(0.06 * H)
        pts, pts2 = [], []
        for k in range(17):
            u = -1.0 + 2.0 * k / 16
            zz = z1f - 0.018 * H - (z1f * 0.62) * (1.0 - u * u)
            off = w(zz) - t(zz) * 0.3
            pts.append((u * ia, off, zz - 60.0)); pts2.append((u * ia, off, zz + 0.012 * H - 60.0))
        A = _xf(pts, (cx, cy, 0.0), yaw); B = _xf(pts2, (cx, cy, 0.0), yaw)
        for k in range(16):
            g.quads(A[k], A[k + 1], B[k + 1], B[k], light, 0.0, 4)
    # floors: a dark deck with a lit gallery band
    for zf_, dw in ((z1f, 0.006), (z2, 0.004)):
        hw = w(zf_) + dw * H
        box(g, (cx, cy, zf_ - 60.0), (hw, hw, 0.008 * H), 0.0, (0.05, 0.05, 0.07))
        box(g, (cx, cy, zf_ - 60.0 + 0.004 * H), (hw + 6, hw + 6, 0.0025 * H), 0.0, light, emis=1.6, kind=7)
    # the shaft
    zs2 = np.linspace(z2, 0.852 * H, 12)
    for z0, z1 in zip(zs2[:-1], zs2[1:]):
        w0, w1 = w(z0), w(z1)
        P0 = [(cx + a * w0, cy + b * w0, z0 - 60.0) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        P1 = [(cx + a * w1, cy + b * w1, z1 - 60.0) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        for k in range(4):
            g.quads(P0[k], P0[(k + 1) % 4], P1[(k + 1) % 4], P1[k], light, 0.0, 4)
    zt = 0.852 * H
    box(g, (cx, cy, zt - 60.0), (w(zt) + 0.004 * H, w(zt) + 0.004 * H, 0.012 * H), 0.0, (0.05, 0.05, 0.07))
    box(g, (cx, cy, zt - 60.0 + 0.01 * H), (w(zt) * 0.7, w(zt) * 0.7, 0.02 * H), 0.0, light, emis=1.8, kind=7)
    frustum(g, (cx, cy, zt - 60.0 + 0.03 * H), w(zt) * 0.35, 0.002 * H, 0, 0.12 * H, 6, (0.08, 0.08, 0.1))
    box(g, (cx, cy, H - 60.0), (0.003 * H, 0.003 * H, 0.004 * H), 0.0, (1.0, 0.95, 0.9), emis=3.0, kind=2)   # beacon
    # lit corner edges (the tower's outline lights)
    ez = np.linspace(0.0, zt, 26)
    for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        d = np.array([a, b]) / math.sqrt(2)
        perp = np.array([-d[1], d[0]])
        for z0, z1 in zip(ez[:-1], ez[1:]):
            e0 = np.array([cx + a * w(z0), cy + b * w(z0)]) + d * 8; e1 = np.array([cx + a * w(z1), cy + b * w(z1)]) + d * 8
            hw = 0.0022 * H
            g.quads((*(e0 - perp * hw), z0 - 60), (*(e0 + perp * hw), z0 - 60), (*(e1 + perp * hw), z1 - 60),
                    (*(e1 - perp * hw), z1 - 60), (0.70, 0.85, 1.0), 2.0, 7)


def _arc_pt(C, s, r, a):
    return (C[0] + s * r * math.cos(a), C[1] + r * math.sin(a))


def _curved_stand(g, eggs, s, rng, team_col, team_dark, amax=0.235, Lc=30000.0, x0=5350.0):
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
            eggs.append(np.concatenate([np.stack([px, py, pz], 1), np.clip(cols, 0, 1), rng.random((len(ang), 1)),
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
        g.tris(np.array([[(c[0], c[1], -45), (a[0], a[1], -45), (b[0], b[1], -45)]]), (0.12, 0.31, 0.10))
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
    terrain(g, [7200, 9000, 13000, 20000, 30000, 42000, 60000], 64, lambda x, y: -60.0,
            lambda x, y, z, nz, q: ((0.11 + 0.02 * q, 0.11 + 0.02 * q, 0.125 + 0.02 * q), 0), rng)
    for i in range(48):                                                                    # paving round the arena
        a0, a1 = 2 * math.pi * i / 48, 2 * math.pi * (i + 1) / 48
        g.tris(np.array([[(0, 0, -60), (7250 * math.cos(a0), 7250 * math.sin(a0), -60), (7250 * math.cos(a1), 7250 * math.sin(a1), -60)]]),
               pink)
    # ---- the two long stands: blue (-x) and orange (+x) ----
    _curved_stand(g, eggs, 1, rng, (0.30, 0.55, 1.0), (0.10, 0.16, 0.40))
    _curved_stand(g, eggs, -1, rng, (1.0, 0.50, 0.12), (0.48, 0.18, 0.05))
    # ---- the garden behind the blue goal: pink paths, lawn beds, the golden-sphere fountain, topiaries,
    #      statues, lamps ----
    g.quads((-7500, -6300, -59), (7500, -6300, -59), (7500, -15500, -59), (-7500, -15500, -59), pink)
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
    g.quads((-3200, 6600, -58), (3200, 6600, -58), (3200, 24000, -58), (-3200, 24000, -58), (0.11, 0.29, 0.09))
    for sx in (-1, 1):
        g.quads((sx * 3200, 6600, -59), (sx * 4600, 6600, -59), (sx * 4600, 24000, -59), (sx * 3200, 24000, -59), (0.55, 0.47, 0.40))
        for y in np.arange(7200, 23800, 950):
            _plane_tree(g, sx * (5000 + rng.uniform(-60, 60)), y, rng.uniform(1200, 1500), rng)
            _plane_tree(g, sx * (6100 + rng.uniform(-60, 60)), y + 475, rng.uniform(1200, 1500), rng)
        for y in np.arange(7000, 24000, 2000):
            box(g, (sx * 3400, y, 280), (14, 14, 340), 0.0, (0.12, 0.12, 0.14))
            ico(g, (sx * 3400, y, 650), 50, (1.0, 0.88, 0.62), rng, sub=0, jit=0.0, emis=1.8, kind=2)
    _eiffel(g, 0.0, 28500.0, 26000.0)
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
    # asteroids: smooth lumpy shapes (low-frequency displacement, 320 triangles for the big ones) with rock detail
    # and crater-like spots drawn per pixel (SCENE kind 10)
    for _ in range(90):
        r = rng.uniform(16000, 62000); a = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(-16000, 18000)
        big = rng.random() > 0.7
        sz = rng.uniform(1300, 3600) if big else rng.uniform(250, 900)
        if big and r < 22000:
            r += 8000
        col = (0.34, 0.31, 0.29) if rng.random() < 0.6 else (0.42, 0.35, 0.28)
        _asteroid(g, (r * math.cos(a), r * math.sin(a), z), sz, col, rng, 3 if big else 2)
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
    return g.array(), np.zeros((0, 8), "f4")                  # no crowd in orbit


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

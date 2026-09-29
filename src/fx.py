"""Visual effects: GPU point-sprite particles + shockwave rings, driven by game events (events.py).

Everything is vectorised numpy on a fixed-capacity pool (no per-particle Python objects), uploaded
with ONE buffer write and drawn with ONE draw call per blend mode per frame. Rings are a handful of
quads with an analytic ring shader.

Effects (what RL shows):
  boost        team-tinted flame + hot core glow out of the exhaust while boosting
  flipreset    RL v2.66 flip-reset indicator: bright white/cyan shockwave crescent in the wheel plane at
               the contact point, a second softer ring, orange ember sparks, contact flash, wheel glow
  demo         fireball + debris sparks + smoke + ground shockwave
  goal         big team-colored burst + ring at the ball
"""
import math
import time

import numpy as np
import moderngl

from rl_shaders import PARTICLE_VERT, PARTICLE_FRAG, RING_VERT, RING_FRAG, TUBE_VERT, TUBE_FRAG

# Flip-reset indicator (RL): a disc in the car's wheel plane whose DIAMETER is the car's length, centred
# under the middle of the car, 120 ms linear fade. (Octane: ~118 uu long; wheel midpoint ~9 uu ahead of
# the origin; wheel contact plane 17 uu below it -- 14 keeps it just off the ball it sits on.)
DEMO_SCALE = 1.2          # demolition explosion size
DEMO_TIME = 1.5           # demolition explosion duration (played this much slower)
GOAL_FX_SPEED = 0.7       # goal explosion playback speed
RESET_DISC_RADIUS = 59.0
RESET_DISC_FWD = 9.0
RESET_DISC_BELOW = 14.0
JUMP_GLOW_LIFE = 0.12        # jump / flip glow
RESET_DISC_LIFE = 0.12

TRAIL_VERT = """
#version 330
uniform mat4 m_vp;
in vec3 in_pos;
in vec4 in_col;
out vec4 v_col;
void main() { v_col = in_col; gl_Position = m_vp * vec4(in_pos, 1.0); }
"""
TRAIL_FRAG = """
#version 330
in vec4 v_col;
out vec4 f_color;
void main() { f_color = vec4(v_col.rgb * v_col.a, v_col.a); }
"""

# Pad pickup dome (RL): a real 3D half sphere of glowing orange-yellow light over the pad (it used to be a flat sprite
# that the pad's own base and the grass cut in half). Additive, depth-tested, no depth write.
DOME_VERT = """
#version 330
uniform mat4 m_vp;
uniform vec3 center;
uniform float radius;
in vec3 in_pos;          // unit half sphere, z >= 0
out vec3 v_n;
out vec3 v_p;
void main() {
    v_n = in_pos;
    v_p = center + in_pos * radius;
    gl_Position = m_vp * vec4(v_p, 1.0);
}
"""
DOME_FRAG = """
#version 330
uniform vec3 camPos;
uniform float alpha;
in vec3 v_n;
in vec3 v_p;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_n);
    vec3 V = normalize(camPos - v_p);
    float ndv = dot(n, V);
    if (ndv < 0.0) discard;                              // front shell only
    // a volume of light, not a surface: brightest where the view crosses the most of it (the middle), fading to
    // NOTHING at the silhouette and softly toward the ground
    float thick = smoothstep(0.0, 1.0, ndv);
    thick *= thick;
    vec3 col = mix(vec3(1.0, 0.45, 0.04), vec3(1.0, 0.86, 0.45), thick);
    float a = alpha * thick * smoothstep(0.0, 0.4, n.z);   // no hard line where it meets the ground either
    f_color = vec4(col * a, a * 0.8);           // mostly covers the turf (additive yellow over green read as lime)
}
"""

# Alpha Boost flame puffs, rebuilt from the game's own particle system (TAGame Boost_AlphaReward_SF.upk,
# Boost_PS "Flame" emitter + LiquidGold_02 material parameters). Every curve below is the game's lookup table sampled at
# t = k/20 of the puff's 1 s life:
#   size        StartSize 75 uu x SizeMultiplyLife (pops to half size in 5% of its life, full by ~55%)
#   alpha       ColorScaleOverLife: fades IN over the first ~35%, 0.5 at 80%, gone at 100%
#   colour      CoreColor (2.5, 1.0, 0.125) x Brightness (0 -> 1 at 25% -> 0.5): over-bright orange that clips to a yellow
#               core and stays orange where the puff is thin
#   noise       NoiseAmount 1 -> 0.1 by mid-life: lumpy "popcorn" blobs early, smooth round puffs later
#   distortion  DistortionAmount 0 -> 5 at mid-life -> 0: the outline wobbles most in the middle of its life
PUFF_VERT = """
#version 330
uniform mat4 m_vp;
uniform float pxScale;
in vec3 in_pos;
in float in_t;
in float in_seed;
in float in_size;
out float v_t;
out float v_seed;
out float v_alpha;
out float v_bright;
out float v_noise;
out float v_dist;
const float SZ[20] = float[20](0.048, 0.488, 0.697, 0.776, 0.823, 0.866, 0.901, 0.931, 0.955, 0.975, 0.989, 1.0,
                               1.007, 1.012, 1.013, 1.013, 1.011, 1.008, 1.004, 1.0);
const float AL[21] = float[21](0.0, 0.052, 0.187, 0.371, 0.572, 0.757, 0.891, 0.944, 0.934, 0.906, 0.864, 0.811, 0.751,
                               0.686, 0.620, 0.557, 0.5, 0.392, 0.223, 0.068, 0.0);
const float BR[21] = float[21](0.0, 0.104, 0.352, 0.648, 0.896, 1.0, 0.994, 0.976, 0.948, 0.912, 0.870, 0.824, 0.775,
                               0.725, 0.676, 0.630, 0.588, 0.552, 0.524, 0.506, 0.5);
const float DI[21] = float[21](0.0, 0.148, 0.548, 1.136, 1.845, 2.611, 3.367, 4.05, 4.593, 4.932, 5.001, 4.789, 4.367,
                               3.79, 3.113, 2.389, 1.675, 1.024, 0.492, 0.132, 0.0);
const float NO[11] = float[11](1.0, 0.975, 0.906, 0.806, 0.683, 0.55, 0.417, 0.294, 0.194, 0.125, 0.1);
float c21(const float a[21], float t) { float x = clamp(t, 0.0, 1.0) * 20.0; int i = min(int(x), 19); return mix(a[i], a[i + 1], x - float(i)); }
void main() {
    float t = in_t;
    // the game's curves, with the first 35% of the life (fade + grow in) a bit faster: the streams show up about half
    // a car length behind the exhaust and grow from there (with the raw curve they only appeared far behind the car)
    if (t < 0.35) t = 0.35 * pow(t / 0.35, 0.75);
    float xs = clamp(t, 0.0, 1.0) * 19.0; int is = min(int(xs), 18);
    float size = in_size * mix(SZ[is], SZ[is + 1], xs - float(is));
    float xn = clamp(t, 0.0, 0.5) * 20.0; int in_ = min(int(xn), 9);
    v_noise = mix(NO[in_], NO[in_ + 1], xn - float(in_));
    v_alpha = c21(AL, t);
    v_bright = c21(BR, t);
    v_dist = c21(DI, t);
    v_t = t;
    v_seed = in_seed;
    vec4 cp = m_vp * vec4(in_pos, 1.0);
    gl_Position = cp;
    gl_PointSize = clamp(size * pxScale / max(cp.w, 1.0), 1.0, 768.0);
}
"""
PUFF_FRAG = """
#version 330
uniform sampler2D coneTex;      // "cones" noise: many small round bumps (like the game's Noise_Cones01_D)
in float v_t;
in float v_seed;
in float v_alpha;
in float v_bright;
in float v_noise;
in float v_dist;
out vec4 f_color;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float a0 = v_seed * 6.2831853;
    q = mat2(cos(a0), sin(a0), -sin(a0), cos(a0)) * q;
    // one RGBA noise texture: r = medium lumps, g = fine grain, ba = a smooth warp field -> 2 fetches per pixel
    float r0 = dot(q, q);
    if (r0 > 1.0) discard;
    vec2 o = vec2(fract(v_seed * 7.13), fract(v_seed * 3.71));
    vec4 nw = texture(coneTex, q * 0.066 + o);
    q += (nw.ba - 0.5) * 0.12 * v_dist;                  // the outline wobbles (DistortionAmount curve)
    float r = length(q);
    vec4 nd = texture(coneTex, q * 0.11 + o * 1.7);
    float nm = nd.r, nf = nd.g;
    float body = (1.0 - r) * 1.25 - (0.5 - nm) * (0.35 + 0.55 * v_noise) - (0.5 - nf) * 0.35;
    float dens = smoothstep(0.0, 0.55, body);          // soft: overlapping puffs build up into one smoky flame
    float a = min(dens * v_alpha * 1.05, 1.0);
    if (a < 0.004) discard;
    // CoreColor (2.5, 1, 0.125) x Brightness (HDR) through an exposure tone curve like the game's: the thick middle
    // over-exposes to pale yellow, the thin wisps stay deep orange
    float inner = smoothstep(0.1, 0.9, body);
    vec3 hdr = vec3(2.5, 1.0, 0.125) * v_bright * (0.25 + 1.8 * inner) * (0.8 + 0.4 * nf);
    vec3 col = 1.0 - exp(-hdr * 1.05);
    f_color = vec4(col * a, a * 0.93);                 // premultiplied "over", a touch additive (glow)
}
"""


TUBE3D_VERT = """
#version 330
uniform mat4 m_vp;
in vec3 in_pos;
in vec3 in_nrm;
in vec4 in_col;
out vec3 v_p;
out vec3 v_n;
out vec4 v_col;
void main() { v_p = in_pos; v_n = in_nrm; v_col = in_col; gl_Position = m_vp * vec4(in_pos, 1.0); }
"""
TUBE3D_FRAG = """
#version 330
uniform vec3 camPos;
in vec3 v_p;
in vec3 v_n;
in vec4 v_col;
out vec4 f_color;
void main() {
    vec3 N = normalize(v_n);
    vec3 V = normalize(camPos - v_p);
    float ndv = dot(N, V);
    if (ndv <= 0.0) discard;                              // only the near half of the tube (winding-free culling)
    // exactly the old camera-facing strip's look (s = 1 on the axis, 0 at the rim), with ndv as the cross-section:
    // the same colour ramp, highlight and soft edge, but on a real round tube that follows the ball in 3D
    float s = ndv;
    vec3 c = v_col.rgb * (0.45 + 0.65 * s) + vec3(1.0) * pow(s, 6.0) * 0.28;
    float a = v_col.a * pow(s, 1.6);
    f_color = vec4(c * a, a);
}
"""


def _cone_noise(n=256, count=2600, seed=7):
    """Tileable 'cones' noise: random round bumps with a linear (cone) falloff, max-combined."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), "f4")
    yy, xx = np.mgrid[0:n, 0:n].astype("f4")
    cx = rng.uniform(0, n, count); cy = rng.uniform(0, n, count); rad = rng.uniform(2.5, 5.5, count)
    h = rng.uniform(0.6, 1.0, count)
    for x0, y0, r0, h0 in zip(cx, cy, rad, h):
        x1, x2 = int(x0 - r0) - 1, int(x0 + r0) + 2
        y1, y2 = int(y0 - r0) - 1, int(y0 + r0) + 2
        ys = np.arange(y1, y2) % n; xs = np.arange(x1, x2) % n
        dx = (np.arange(x1, x2) - x0)[None, :]; dy = (np.arange(y1, y2) - y0)[:, None]
        v = h0 * np.clip(1.0 - np.sqrt(dx * dx + dy * dy) / r0, 0.0, 1.0)
        img[np.ix_(ys, xs)] = np.maximum(img[np.ix_(ys, xs)], v)
    return (img * 255).astype("u1")


def _puff_noise(n=256):
    """RGBA puff noise, tileable: r = cone lumps, g = the same pattern 3x finer, b / a = two smooth warp fields."""
    cones = _cone_noise(n).astype("f4") / 255.0
    fine = np.tile(cones[::3, ::3] if n % 3 == 0 else cones[::2, ::2], (3, 3))[:n, :n]
    rng = np.random.default_rng(3)

    def smooth(k):
        g = rng.random((k, k)).astype("f4")
        # bilinear upsample (tileable)
        x = np.arange(n) * k / n
        i0 = np.floor(x).astype(int) % k; i1 = (i0 + 1) % k; f = (x - np.floor(x))[None, :]
        rows = g[:, i0] * (1 - f) + g[:, i1] * f
        fy = (x - np.floor(x))[:, None]
        return rows[i0, :] * (1 - fy) + rows[i1, :] * fy

    out = np.stack([cones, fine, smooth(8), smooth(8)], -1)
    return (np.clip(out, 0, 1) * 255).astype("u1")


# Goal shock sphere: glowing rim (fresnel), clear in the middle, additive
SHELL_FRAG = """
#version 330
uniform vec3 camPos;
uniform float alpha;
uniform vec3 col;
in vec3 v_n;
in vec3 v_p;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_n);
    vec3 V = normalize(camPos - v_p);
    float ndv = abs(dot(n, V));
    // a thin bright shock front at the silhouette in the team colour, white-hot at its very edge, clear inside
    float rim = pow(1.0 - ndv, 5.0);
    vec3 c = mix(col * 1.25, col * 0.4 + 0.6, 0.5 * pow(rim, 4.0));
    float a = alpha * rim;
    f_color = vec4(c * a * 1.4, a * 0.7);        // partly "over": keeps the team colour on a bright background
}
"""


def _half_sphere(nlon=48, nlat=12):
    tris = []
    for i in range(nlat):
        a0, a1 = 0.5 * math.pi * i / nlat, 0.5 * math.pi * (i + 1) / nlat
        for j in range(nlon):
            b0, b1 = 2 * math.pi * j / nlon, 2 * math.pi * (j + 1) / nlon
            P = lambda a, b: (math.cos(a) * math.cos(b), math.cos(a) * math.sin(b), math.sin(a))
            q = (P(a0, b0), P(a0, b1), P(a1, b1), P(a1, b0))
            tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
    return np.asarray(tris, "f4")


FLAME_VERT = """
#version 330
uniform mat4 m_vp;
uniform mat4 m_model;
uniform float L;          // flame length (uu)
uniform float R0;         // radius at the nozzle
uniform vec3 nozzle;      // car-space nozzle position
uniform float time;
in vec2 in_sa;            // s = 0..1 along the flame, a = angle around it
out float v_s;
out vec3 v_n;
out vec3 v_p;
out float v_a;
void main() {
    float s = in_sa.x, a = in_sa.y;
    float wob = 1.0 + 0.04 * sin(time * 47.0 + a * 2.0) * s;
    float r = R0 * pow(1.0 - s, 0.55) * (0.75 + 0.5 * smoothstep(0.0, 0.18, s)) * wob;
    vec3 radial = vec3(0.0, cos(a), sin(a));
    vec3 lp = nozzle + vec3(-s * L, 0.0, 0.0) + radial * r;
    vec4 wp = m_model * vec4(lp, 1.0);
    v_p = wp.xyz;
    v_n = mat3(m_model) * radial;
    v_s = s; v_a = a;
    gl_Position = m_vp * wp;
}
"""
FLAME_FRAG = """
#version 330
uniform vec3 camPos;
uniform vec3 hotCol;
uniform vec3 flameCol;
uniform float time;
uniform float gain;
in float v_s;
in vec3 v_n;
in vec3 v_p;
in float v_a;
out vec4 f_color;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec3 V = normalize(camPos - v_p);
    float edge = pow(abs(dot(normalize(v_n), V)), 0.9);           // soft silhouette
    float n = noise(vec2(v_s * 7.0 - time * 26.0, v_a * 1.3)) * 0.6 + noise(vec2(v_s * 17.0 - time * 41.0, v_a * 3.1)) * 0.4;
    float body = pow(1.0 - v_s, 1.4) * (0.55 + 0.75 * n);
    vec3 c = mix(vec3(1.0), hotCol, smoothstep(0.02, 0.22, v_s));
    c = mix(c, flameCol, smoothstep(0.25, 0.75, v_s));
    float a = clamp(body * edge * gain, 0.0, 1.0);
    f_color = vec4(c * a, a);
}
"""


def _flame_mesh(ns=14, na=18):
    """Triangle list over (s, angle) for the flame cone."""
    verts = []
    for i in range(ns):
        s0, s1 = i / ns, (i + 1) / ns
        for j in range(na):
            a0, a1 = 2 * math.pi * j / na, 2 * math.pi * (j + 1) / na
            verts += [(s0, a0), (s1, a0), (s1, a1), (s0, a0), (s1, a1), (s0, a1)]
    return np.asarray(verts, "f4")


TEAM_BOOST = [
    ((0.55, 0.85, 1.00), (0.10, 0.35, 1.00)),   # blue: hot core, outer flame
    ((1.00, 0.85, 0.45), (1.00, 0.30, 0.02)),   # orange
]
TEAM_GOAL = [(0.25, 0.55, 1.0), (1.0, 0.45, 0.05)]


def _cross(a, b):
    """Row-wise cross product of (N, 3) arrays (b may be (3,) or (1, 3)) without np.cross's dispatch overhead."""
    a = np.asarray(a); b = np.asarray(b)
    if b.ndim == 1:
        b = b[None, :]
    out = np.empty((max(len(a), len(b)), 3), np.result_type(a, b))
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _perp_basis(n):
    n = np.asarray(n, "f8")
    n = n / max(np.linalg.norm(n), 1e-6)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


class ParticlePool:
    """Particles in ONE (cap, 21) float32 array (pos 3, vel 3, age, life, s0, s1, c0 4, c1 4, drag, grav): spawning,
    killing and compacting touch a single array instead of ten. col0 -> col1 and size0 -> size1 over life."""
    P, V, AGE, LIFE, S0, S1, C0, C1, DRAG, GRAV = slice(0, 3), slice(3, 6), 6, 7, 8, 9, slice(10, 14), slice(14, 18), 18, 19

    def __init__(self, cap):
        self.cap = cap
        self.n = 0
        self.a = np.zeros((cap, 21), "f4")

    @property
    def pos(self):
        return self.a[:, self.P]

    keep_frac = 1.0                                # settings: Particles quality (fraction of multi-particle bursts)

    def spawn(self, pos, vel, life, s0, s1, c0, c1, drag=0.0, grav=0.0):
        k = len(pos)
        if k == 0:
            return
        if k > 1 and self.keep_frac < 1.0:          # thin bursts (single glows / flashes always stay)
            m = np.random.random(k) < self.keep_frac
            if not m.any():
                return
            pick1 = lambda x: x[m] if (np.ndim(x) == 1 and len(x) == k) else x      # per-particle scalars
            pick2 = lambda x: x[m] if np.ndim(x) == 2 else x                           # per-particle vectors / colours
            pos, vel, c0, c1 = (pick2(np.asarray(x)) for x in (pos, vel, c0, c1))
            life, s0, s1, drag, grav = (pick1(np.asarray(x)) for x in (life, s0, s1, drag, grav))
            k = len(pos)
        if self.n + k > self.cap:                  # full: drop the oldest particles
            drop = min(self.n, self.n + k - self.cap)
            self.a[:self.n - drop] = self.a[drop:self.n]
            self.n -= drop
            k = min(k, self.cap)
        blk = self.a[self.n:self.n + k]
        blk[:, self.P] = pos[:k] if np.ndim(pos) == 2 else pos
        blk[:, self.V] = vel[:k] if np.ndim(vel) == 2 else vel
        blk[:, self.AGE] = 0.0
        blk[:, self.LIFE] = life
        blk[:, self.S0] = s0
        blk[:, self.S1] = s1
        blk[:, self.C0] = c0
        blk[:, self.C1] = c1
        blk[:, self.DRAG] = drag
        blk[:, self.GRAV] = grav
        self.n += k

    def update(self, dt):
        n = self.n
        if n == 0:
            return
        a = self.a
        a[:n, self.AGE] += dt
        alive = a[:n, self.AGE] < a[:n, self.LIFE]
        if not alive.all():
            keep = a[:n][alive]
            n = self.n = len(keep)
            a[:n] = keep
            if n == 0:
                return
        v = a[:n, self.V]
        v *= np.exp(-a[:n, self.DRAG] * dt)[:, None]
        v[:, 2] -= a[:n, self.GRAV] * dt
        a[:n, self.P] += v * dt

    def vertex_data(self, out):
        """Fill out[(n, 8)] = pos(3) col(4) size(1)."""
        n = self.n
        a = self.a[:n]
        t = (a[:, self.AGE] / a[:, self.LIFE])[:, None]
        out[:n, 0:3] = a[:, self.P]
        out[:n, 3:7] = a[:, self.C0] + (a[:, self.C1] - a[:, self.C0]) * t
        out[:n, 7] = a[:, self.S0] + (a[:, self.S1] - a[:, self.S0]) * t[:, 0]
        return n


class Ring:
    __slots__ = ("center", "u", "v", "r0", "r1", "life", "age", "width", "color", "core", "arc", "arcw", "delay",
                 "billboard", "fill", "sparkle", "ontop", "mode", "drawn")


class FX:
    CAP = 14000

    def __init__(self, ctx):
        self.ctx = ctx
        self.add = ParticlePool(self.CAP)          # additive: flames, sparks, flashes
        self.alpha = ParticlePool(9000)            # premultiplied-over: smoke
        self.rings = []
        self.reset_discs = []                      # flip-reset indicators, attached to their car
        self.car_poses = {}                        # car index -> (pos, forward, up), set each frame by main
        self.wheel_glow = {}                       # car idx -> time of last flip reset
        self.screen_flash = 0.0                    # time of last spectated flip reset (2D streaks)
        self.prog = ctx.program(vertex_shader=PARTICLE_VERT, fragment_shader=PARTICLE_FRAG)
        self.buf = np.zeros((self.CAP + 9000 + 256, 8), "f4")
        self.vbo = ctx.buffer(reserve=self.buf.nbytes, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 4f 1f", "in_pos", "in_col", "in_size")])
        self.ring_prog = ctx.program(vertex_shader=RING_VERT, fragment_shader=RING_FRAG)
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4")
        self.ring_vao = ctx.vertex_array(self.ring_prog, [(ctx.buffer(quad.tobytes()), "2f", "in_xy")])
        self._boost_accum = {}
        self._boost_last_base = {}
        self.flares = []                           # hit flashes: [pos, age, life, size, alpha]
        self.nozzle_flares = []
        self.domes = []                            # pad pickup glow domes: [pos, age, life, r0, r1, alpha]
        self.shells = []                           # goal shock spheres: [pos, age, life, r0, r1, col, alpha]
        self._rng = np.random.default_rng()
        self.trail_prog = ctx.program(vertex_shader=TRAIL_VERT, fragment_shader=TRAIL_FRAG)
        self.TRAIL_MAX = 49152
        self.trail_vbo = ctx.buffer(reserve=self.TRAIL_MAX * 7 * 4, dynamic=True)
        self.trail_vao = ctx.vertex_array(self.trail_prog, [(self.trail_vbo, "3f 4f", "in_pos", "in_col")])
        self._pad_glow = None
        self._pad_charge = []
        self._trails = []
        self._tubes = []
        self.tube_prog = ctx.program(vertex_shader=TUBE_VERT, fragment_shader=TUBE_FRAG)
        self.TUBE_MAX = 24576
        self.tube_vbo = ctx.buffer(reserve=self.TUBE_MAX * 8 * 4, dynamic=True)
        self.tube_vao = ctx.vertex_array(self.tube_prog, [(self.tube_vbo, "3f 4f 1f", "in_pos", "in_col", "in_u")])
        self._beams = []
        self._beam_blocks = []
        self._tubes3d = []
        self.tube3d_prog = ctx.program(vertex_shader=TUBE3D_VERT, fragment_shader=TUBE3D_FRAG)
        self.TUBE3D_MAX = 32768
        self.tube3d_vbo = ctx.buffer(reserve=self.TUBE3D_MAX * 10 * 4, dynamic=True)
        self.tube3d_vao = ctx.vertex_array(self.tube3d_prog, [(self.tube3d_vbo, "3f 3f 4f", "in_pos", "in_nrm", "in_col")])
        self.flame_prog = ctx.program(vertex_shader=FLAME_VERT, fragment_shader=FLAME_FRAG)
        fm = _flame_mesh()
        self.flame_vao = ctx.vertex_array(self.flame_prog, [(ctx.buffer(fm.tobytes()), "2f", "in_sa")])
        self.flame_n = len(fm)
        self._flames = []
        self.cam_right = np.array([1.0, 0.0, 0.0], "f4")
        self.cam_up = np.array([0.0, 0.0, 1.0], "f4")
        self.puff_prog = ctx.program(vertex_shader=PUFF_VERT, fragment_shader=PUFF_FRAG)
        self.PUFF_CAP = 2400
        self.puff_p = np.zeros((self.PUFF_CAP, 3), "f4")
        self.puff_age = np.zeros(self.PUFF_CAP, "f4")
        self.puff_seed = np.zeros(self.PUFF_CAP, "f4")
        self.puff_acc = np.zeros(self.PUFF_CAP, "f4")
        self.puff_size = np.zeros(self.PUFF_CAP, "f4")
        self.puff_n = 0
        self.puff_vbo = ctx.buffer(reserve=self.PUFF_CAP * 6 * 4, dynamic=True)
        self.puff_vao = ctx.vertex_array(self.puff_prog, [(self.puff_vbo, "3f 1f 1f 1f", "in_pos", "in_t", "in_seed", "in_size")])
        self.cone_tex = ctx.texture((256, 256), 4, _puff_noise().tobytes())
        self.cone_tex.build_mipmaps()
        self.cone_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._boost_dist = {}
        self.dome_prog = ctx.program(vertex_shader=DOME_VERT, fragment_shader=DOME_FRAG)
        hs = _half_sphere()
        self.dome_vao = ctx.vertex_array(self.dome_prog, [(ctx.buffer(hs.tobytes()), "3f", "in_pos")])
        self.dome_n = len(hs)
        self.shell_prog = ctx.program(vertex_shader=DOME_VERT, fragment_shader=SHELL_FRAG)
        fs = np.concatenate([hs, hs * np.array([1.0, -1.0, -1.0], "f4")], 0)      # full sphere
        self.shell_vao = ctx.vertex_array(self.shell_prog, [(ctx.buffer(fs.tobytes()), "3f", "in_pos")])
        self.shell_n = len(fs)

    def pad_pickup(self, pos, big):
        """Boost pad picked up (RL): a glowing orange-yellow half dome over the pad (a soft sprite whose lower half the
        floor hides), brief -- ~0.1 s on a small pad, longer on a big one. Small pads also puff a few dark specks
        upward; big pads throw a burst of bright golden sparks up and out."""
        rng = self._rng
        pos = np.asarray(pos, "f4").copy(); pos[2] = 0.0
        if big:
            self.domes.append([pos.copy(), 0.0, 0.30, 105.0, 175.0, 0.85])     # big shell, expanding
            self.domes.append([pos.copy(), 0.0, 0.16, 70.0, 100.0, 1.0])       # hot inner shell
            n = 46
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 2.2 + 0.6
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            p0 = pos + np.array([0, 0, 50.0], "f4")
            self.add.spawn(np.repeat(p0[None, :], n, 0), d * rng.uniform(380, 1000, (n, 1)),
                           rng.uniform(0.35, 0.70, n), rng.uniform(9, 16, n), 2.5,
                           np.array([1.0, 0.90, 0.50, 1.0], "f4"), np.array([1.0, 0.40, 0.04, 0.0], "f4"), drag=2.6, grav=500.0)
            # a quick bright flash on the orb itself
            self.add.spawn(p0[None, :], np.zeros((1, 3), "f4"), np.array([0.12], "f4"), np.array([140.0], "f4"), 60.0,
                           np.array([1.0, 0.85, 0.45, 0.9], "f4"), np.array([1.0, 0.5, 0.1, 0.0], "f4"))
        else:
            self.domes.append([pos.copy(), 0.0, 0.10, 58.0, 76.0, 0.9])
            n = 9
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 3.0 + 1.0
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            self.alpha.spawn(np.repeat((pos + np.array([0, 0, 8.0], "f4"))[None, :], n, 0) + d * 10.0,
                             d * rng.uniform(250, 520, (n, 1)), rng.uniform(0.25, 0.45, n), rng.uniform(3.0, 5.0, n), 2.0,
                             np.array([0.10, 0.07, 0.04, 0.9], "f4"), np.array([0.10, 0.07, 0.04, 0.0], "f4"), drag=2.0, grav=250.0)

    def _render_domes(self, m_vp_bytes, cam_pos):
        """The live pickup domes: real half spheres, growing a little and fading out."""
        if not self.domes:
            return
        dp = self.dome_prog
        dp["m_vp"].write(m_vp_bytes)
        dp["camPos"].value = tuple(float(c) for c in cam_pos)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        for pos, age, life, r0, r1, a0 in self.domes:
            t = min(age / life, 1.0)
            e = 1.0 - (1.0 - t) ** 2                    # ease-out growth
            dp["center"].value = (float(pos[0]), float(pos[1]), 0.5)
            dp["radius"].value = float(r0 + (r1 - r0) * e)
            dp["alpha"].value = float(a0 * (1.0 - t) ** 1.3)
            self.dome_vao.render(moderngl.TRIANGLES, vertices=self.dome_n)
        self.ctx.enable(moderngl.CULL_FACE)

    def pad_charge(self, items):
        """items: [(x, y, z, radius, progress 0..1)] for every EMPTY pad -> recharge rings drawn this frame: a ring
        on the pad's rim whose lit arc sweeps around as it recharges, deep red -> gold."""
        self._pad_charge = items

    def pad_glows(self, pads, t):
        """pads: [(x, y, is_big, active, progress, ghost)] -> soft light sprites drawn this frame (additive): a warm halo
        around an active orb + light on the floor; an empty pad's floor light goes deep red -> orange as it
        recharges."""
        if not pads:
            self._pad_glow = None
            return
        n = len(pads)
        g = np.zeros((3 * n, 8), "f4")
        k = 0
        for i, (x, y, big, active, prog, ghost) in enumerate(pads):
            pulse = 0.85 + 0.15 * math.sin(t * 3.0 + i * 1.7)
            if active:
                if big:
                    g[k] = (x, y, 74.0, 1.0, 0.62, 0.15, 0.30 * pulse, 190.0 * pulse); k += 1   # orb halo
                    g[k] = (x, y, 12.0, 1.0, 0.55, 0.10, 0.30, 300.0); k += 1                     # floor light
                else:
                    g[k] = (x, y, 10.0, 1.0, 0.62, 0.15, 0.30 * pulse, 110.0); k += 1
            else:
                p = float(prog)
                if not big:          # an empty BIG pad has no red/orange glow dome until it respawns
                    g[k] = (x, y, 12.0, 1.0, 0.12 + 0.43 * p, 0.03 + 0.07 * p, 0.10 + 0.28 * p, 110.0); k += 1
                if ghost > 0.0:
                    # the returning orb's blur: a soft whitish halo spilling past its dissolved silhouette,
                    # shrinking and fading as the orb comes into focus
                    a = 0.16 * min(1.0, ghost * 4.0)
                    g[k] = (x, y, 74.0 if big else 10.0, 0.85, 0.88, 0.95, a, (170.0 if big else 70.0)); k += 1
        self._pad_glow = g[:k]

    def add_trail(self, ribbon, lifetime, width, color, up=None, flat=False):
        """Queue a RibbonEmitter's connected points as a camera-facing, additive strip that fades
        with age and tapers at both ends (drawn batched in render())."""
        pts = [p for p in ribbon.points if p.connected]
        if len(pts) < 2:
            return
        pos = np.array([p.pos for p in pts], "f4")
        clk = ribbon.clock
        age = np.array([clk - p.t0 for p in pts], "f4")
        self._trails.append((pos, age, lifetime, width, color, None if up is None else np.asarray(up, "f4"), flat))

    def add_tube(self, ribbon, lifetime, radius, color, white_from=0.0, white_len=0.0):
        """Queue a RibbonEmitter (newest point first) as a round, shaded tube that fades with age. The first
        `white_len` uu after `white_from` (measured along the trail from its head) blend from white to `color`."""
        pts = [p for p in ribbon.points if p.connected]
        if len(pts) < 2:
            return
        base = tuple(color[:3])
        clk = ribbon.clock
        self._tubes.append((np.array([p.pos for p in pts], "f4"),
                            np.array([clk - p.t0 for p in pts], "f4"),
                            np.asarray([getattr(p, "k", 1.0) for p in pts], "f4"), lifetime, radius,
                            np.asarray([getattr(p, "col", None) or base for p in pts], "f4"),   # colour per point
                            float(color[3]) if len(color) > 3 else 1.0, white_from, white_len))

    def add_tube3d(self, ribbon, lifetime, radius, color, white_from=0.0, white_len=0.0):
        """Like add_tube, but a real 3D tube mesh (a ring of vertices round every point, lit), not a camera strip."""
        pts = [p for p in ribbon.points if p.connected]
        if len(pts) < 2:
            return
        base = tuple(color[:3])
        clk = ribbon.clock
        self._tubes3d.append((np.array([p.pos for p in pts], "f4"), np.array([clk - p.t0 for p in pts], "f4"),
                              np.asarray([getattr(p, "k", 1.0) for p in pts], "f4"), lifetime, radius,
                              np.asarray([getattr(p, "col", None) or base for p in pts], "f4"),
                              float(color[3]) if len(color) > 3 else 1.0, white_from, white_len))

    TUBE3D_SIDES = 10

    def _render_tubes3d(self, m_vp_bytes, cam):
        if not self._tubes3d:
            return
        T = self._tubes3d
        self._tubes3d = []
        S = self.TUBE3D_SIDES
        ang = np.arange(S + 1, dtype="f4") * (2.0 * math.pi / S)
        ca, sa = np.cos(ang), np.sin(ang)
        out = []
        for pos, age, kpt, life, radius, rgb, a0, w_from, w_len in T:
            n = len(pos)
            seg = np.empty_like(pos)
            seg[:-1] = pos[1:] - pos[:-1]
            seg[-1] = seg[-2]
            sl = np.sqrt((seg * seg).sum(1))
            tan = seg / (sl[:, None] + 1e-6)
            # parallel-transported frame (no twisting where the path turns or goes vertical)
            t0 = tan[0]
            ref = (0.0, 0.0, 1.0) if abs(float(t0[2])) < 0.9 else (1.0, 0.0, 0.0)
            nx = np.empty((n, 3), "f4")
            v = np.cross(t0, ref); v /= (np.linalg.norm(v) + 1e-9)
            tl = tan.tolist()
            vx, vy, vz = (float(c) for c in v)
            for i in range(n):
                tx, ty, tz = tl[i]
                d = vx * tx + vy * ty + vz * tz
                vx -= d * tx; vy -= d * ty; vz -= d * tz
                inv = 1.0 / (math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9)
                vx *= inv; vy *= inv; vz *= inv
                nx[i] = (vx, vy, vz)
            bx = np.cross(tan, nx)
            nrm = nx[:, None, :] * ca[None, :, None] + bx[:, None, :] * sa[None, :, None]      # (n, S+1, 3)
            t = np.clip(age / life, 0.0, 1.0)
            dist = np.concatenate([[0.0], np.cumsum(sl[:-1])]).astype("f4")
            w = 1.0 - np.clip((dist - w_from) / max(w_len, 1e-3), 0.0, 1.0) if w_len > 0 else np.zeros(n, "f4")
            rad = radius * (1.0 - 0.35 * t)
            V = np.empty((n, S + 1, 10), "f4")
            V[:, :, 0:3] = pos[:, None, :] + nrm * rad[:, None, None]
            V[:, :, 3:6] = nrm
            V[:, :, 6:9] = (rgb * (1.0 - w[:, None]) + w[:, None])[:, None, :]
            V[:, :, 9] = (a0 * (1.0 - t) * kpt)[:, None]
            a, b, c, d = V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:]
            out.append(np.stack([a, b, c, b, d, c], 2).reshape(-1, 10))
        data = np.concatenate(out, 0)[:self.TUBE3D_MAX]
        self.tube3d_vbo.write(data.tobytes())
        p = self.tube3d_prog
        p["m_vp"].write(m_vp_bytes)
        p["camPos"].value = tuple(float(c) for c in cam)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        self.tube3d_vao.render(moderngl.TRIANGLES, vertices=len(data))
        self.ctx.enable(moderngl.CULL_FACE)

    def _render_tubes(self, m_vp_bytes, cam):
        """All queued tubes in ONE vectorised build (no per-tube numpy calls) -> one triangle-list draw."""
        if not self._tubes:
            return
        cam = np.asarray(cam, "f4")
        T = self._tubes
        self._tubes = []
        lens = np.array([len(t[0]) for t in T])
        pos = np.concatenate([t[0] for t in T], 0)
        age = np.concatenate([t[1] for t in T], 0)
        kpt = np.concatenate([t[2] for t in T], 0)
        rgb = np.concatenate([t[5] for t in T], 0)
        life = np.repeat([t[3] for t in T], lens).astype("f4")
        radius = np.repeat([t[4] for t in T], lens).astype("f4")
        a0 = np.repeat([t[6] for t in T], lens).astype("f4")
        w_from = np.repeat([t[7] for t in T], lens).astype("f4")
        w_len = np.repeat([t[8] for t in T], lens).astype("f4")
        starts = np.cumsum(lens) - lens
        ends = np.cumsum(lens) - 1
        seg = np.empty_like(pos)
        seg[:-1] = pos[1:] - pos[:-1]
        seg[ends] = seg[ends - 1]
        sl = np.sqrt((seg * seg).sum(1))
        cs = np.concatenate([[0.0], np.cumsum(sl[:-1])]).astype("f4")
        dist = cs - np.repeat(cs[starts], lens)                  # distance along each tube from its head
        t = np.clip(age / life, 0.0, 1.0)
        w = np.where(w_len > 0, 1.0 - np.clip((dist - w_from) / np.maximum(w_len, 1e-3), 0.0, 1.0), 0.0)
        rgba = np.empty((len(pos), 4), "f4")
        rgba[:, 0:3] = rgb * (1.0 - w[:, None]) + w[:, None]
        rgba[:, 3] = a0 * (1.0 - t) * kpt
        side = _cross(seg, cam[None, :] - pos)
        side *= (radius * (1.0 - 0.35 * t) / (np.sqrt((side * side).sum(1)) + 1e-6))[:, None]
        n = len(pos)
        VL = np.empty((n, 8), "f4"); VH = np.empty((n, 8), "f4")
        VL[:, 0:3] = pos - side; VH[:, 0:3] = pos + side
        VL[:, 3:7] = rgba; VH[:, 3:7] = rgba
        VL[:, 7] = -1.0; VH[:, 7] = 1.0
        sid = np.repeat(np.arange(len(lens)), lens)
        i = np.nonzero(sid[:-1] == sid[1:])[0]
        j = i + 1
        data = np.stack([VL[i], VH[i], VL[j], VH[i], VH[j], VL[j]], 1).reshape(-1, 8)[:self.TUBE_MAX]
        self.tube_vbo.write(data.tobytes())
        self.tube_prog["m_vp"].write(m_vp_bytes)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        self.tube_vao.render(moderngl.TRIANGLES, vertices=len(data))
        self.ctx.enable(moderngl.CULL_FACE)

    def add_beam(self, p0, p1, w0, w1, c0, c1, mid=None):
        """Camera-facing tapered quad strip p0 -> (mid) -> p1 with per-end width/colour (flames)."""
        # plain lists -> one array each (np.linspace + broadcasting cost ~0.05 ms a beam)
        if mid is None:
            self._beams.append((np.array((p0, p1), "f4"), np.array((w0, w1), "f4"), np.array((c0, c1), "f4")))
        else:
            cm = [0.5 * (a + b) for a, b in zip(c0, c1)]
            self._beams.append((np.array((p0, mid, p1), "f4"), np.array((w0, 0.5 * (w0 + w1), w1), "f4"),
                                np.array((c0, cm, c1), "f4")))

    # ---------------------------------------------------------------------------------------- #
    def _rand_dirs(self, k):
        d = self._rng.normal(size=(k, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    NOZZLE = (-57.0, 0.0, 10.0)     # Octane exhaust, car space
    ALPHA_HOT = (1.0, 0.90, 0.55)    # Alpha Boost: golden-yellow streams, white-hot orange glow at the nozzle. One look for
    ALPHA_FLAME = (1.0, 0.62, 0.08)  # both teams.
    STREAM_OFFSET = 24.0             # the two streams leave the exhaust this far to each side
    PUFF_SPACING = 32.0              # the game's SpawnPerUnit: one puff per 32 uu the exhaust travels

    def set_quality(self, level):
        """Particles quality 0 / 1 / 2 -> fraction of burst particles and boost puffs kept."""
        f = (0.4, 0.7, 1.0)[max(0, min(2, int(level)))]
        self.add.keep_frac = self.alpha.keep_frac = f
        self.puff_keep = (0.55, 0.8, 1.0)[max(0, min(2, int(level)))]

    def _add_puffs(self, pts):
        if getattr(self, "puff_keep", 1.0) < 1.0 and len(pts) > 1:
            pts = pts[np.random.random(len(pts)) < self.puff_keep]
        k = len(pts)
        if k == 0:
            return
        n = self.puff_n
        if n + k > self.PUFF_CAP:                      # drop the oldest
            drop = n + k - self.PUFF_CAP
            keep = slice(drop, n)
            self.puff_p[:n - drop] = self.puff_p[keep]; self.puff_age[:n - drop] = self.puff_age[keep]
            self.puff_seed[:n - drop] = self.puff_seed[keep]; self.puff_acc[:n - drop] = self.puff_acc[keep]
            self.puff_size[:n - drop] = self.puff_size[keep]
            n -= drop
        self.puff_p[n:n + k] = pts
        self.puff_age[n:n + k] = 0.0
        self.puff_seed[n:n + k] = self._rng.uniform(0.0, 1.0, k)
        self.puff_acc[n:n + k] = self._rng.uniform(15.0, 30.0, k)    # the game's Acceleration: z 15..30 uu/s^2
        # the boost actor's ParticleSize 35..50 uu (x0.95)
        self.puff_size[n:n + k] = self._rng.uniform(35.0, 50.0, k) * 0.95
        self.puff_n = n + k

    def boost(self, key, pos, fwd, up, car_vel, team, dt, model_bytes=None):
        """Boosting car this frame -- Alpha Boost as the game defines it (see PUFF_VERT): a flame puff left in the
        world every 32 uu the exhaust travels (two exhausts), which then hangs there on its own 1 s life cycle; at the
        nozzle a small hot glow blown back out of the exhaust (the game's Drive_PS emitter: 10 sprites/s, 0.5 s, growing
        4x, colour scale 2 -> 0) and the lens-flare streak. Same for both teams."""
        fwd = np.asarray(fwd, "f4"); up = np.asarray(up, "f4")
        pos = np.asarray(pos, "f4")
        base = pos + fwd * self.NOZZLE[0] + up * self.NOZZLE[2]
        cv = np.asarray(car_vel, "f4")
        rng = self._rng
        fl = float(rng.uniform(0.9, 1.1))
        # (no glow sprite / lens flare / flame cones at the exhaust: the Alpha Boost is just its smoke + sparkles)
        right = np.array([fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2],
                          fwd[0] * up[1] - fwd[1] * up[0]], "f4")
        # SpawnPerUnit (UnitScalar 32 uu) x the boost actor's SpawnRate, a random 0..4 drawn per 32 uu step: on average
        # 2 puffs per 32 uu, but bunched -- some steps drop none, others three -> the clumps and gaps of the game
        prev = self._boost_last_base.get(key)
        self._boost_last_base[key] = base.copy()
        if prev is None or float(np.linalg.norm(base - prev)) > 400.0:
            prev = base
        seg = base - prev
        L = float(np.linalg.norm(seg))
        d0 = self._boost_dist.get(key, self.PUFF_SPACING)       # distance since the last step (first one at once)
        total = d0 + L
        k = int(total // self.PUFF_SPACING)
        self._boost_dist[key] = total - k * self.PUFF_SPACING
        if k <= 0:
            return
        k = min(k, 12)
        f = ((np.arange(1, k + 1, dtype="f4") * self.PUFF_SPACING - d0) / max(L, 1e-3)).clip(0.0, 1.0)
        pts = []
        for side in (-1.0, 1.0):
            cnt = rng.integers(1, 3, k)                          # 1..2 per 32 uu step: separate puffs, a clear stream
            ff = np.repeat(f, cnt)
            if len(ff) == 0:
                continue
            ff = (ff + rng.uniform(-0.5, 0.5, len(ff)) * self.PUFF_SPACING / max(L, 1e-3)).clip(0.0, 1.0)[:, None]
            pts.append(prev[None, :] + seg[None, :] * ff + right[None, :] * (side * self.STREAM_OFFSET)
                       + rng.normal(0.0, 2.0, (len(ff), 3)).astype("f4"))
        if pts:
            pts = np.concatenate(pts, 0)
            self._add_puffs(pts)
            # embers: tiny bright sparks sprinkled through the flame, drifting out and up, twinkling out
            ke = min(len(pts) * 3, 60)
            if ke:
                src = pts[rng.integers(0, len(pts), ke)] + rng.normal(0.0, 12.0, (ke, 3)).astype("f4")
                d = self._rand_dirs(ke)
                self.add.spawn(src, d * rng.uniform(15.0, 80.0, (ke, 1)).astype("f4") + np.array([0, 0, 25.0], "f4"),
                               rng.uniform(0.25, 0.8, ke).astype("f4"), rng.uniform(2.0, 3.8, ke).astype("f4"), 1.0,
                               np.array([1.0, 0.97, 0.75, 1.0], "f4"), np.array([1.0, 0.6, 0.15, 0.0], "f4"), drag=1.5)

    def boost_end(self, key):
        """The car stopped boosting: the next boost starts a new puff trail right at the nozzle."""
        self._boost_last_base.pop(key, None)
        self._boost_dist.pop(key, None)

    def _render_puffs(self, m_vp_bytes, px_scale, cam_pos):
        n = self.puff_n
        if n == 0:
            return
        cam = np.asarray(cam_pos, "f4")
        d2 = ((self.puff_p[:n] - cam[None, :]) ** 2).sum(1)
        order = np.argsort(-d2)                          # back to front (the game sorts by distance to view)
        buf = np.empty((n, 6), "f4")
        buf[:, 0:3] = self.puff_p[order]
        buf[:, 3] = self.puff_age[order]
        buf[:, 4] = self.puff_seed[order]
        buf[:, 5] = self.puff_size[order]
        self.puff_vbo.write(buf.tobytes())
        pp = self.puff_prog
        pp["m_vp"].write(m_vp_bytes)
        pp["pxScale"].value = float(px_scale)
        self.cone_tex.use(location=12)
        pp["coneTex"].value = 12
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.puff_vao.render(moderngl.POINTS, vertices=n)

    def _nozzle_flare_beams(self):
        for p, fl in self.nozzle_flares:
            r = self.cam_right.astype("f4") * 95.0 * fl
            self._beams.append((np.asarray([p - r, p, p + r], "f4"), np.array([0.8, 2.2, 0.8], "f4"),
                                np.array([[1.0, 0.55, 0.15, 0.0], [1.0, 0.80, 0.45, 0.8], [1.0, 0.55, 0.15, 0.0]], "f4")))
        self.nozzle_flares = []

    def sparkle(self, key, pos, dt):
        """Supersonic trail speckle: a few tiny bluish-white sparkles left along the short violet streak."""
        rng = self._rng
        acc = self._boost_accum.get(("sp", key), 0.0) + 70.0 * dt
        k = int(acc)
        self._boost_accum[("sp", key)] = acc - k
        k = min(k, 3)
        if k <= 0:
            return
        j = rng.normal(0.0, 1.0, (k, 3)).astype("f4")
        self.add.spawn(np.asarray(pos, "f4")[None, :] + j * 3.0, j * 30.0, rng.uniform(0.10, 0.22, k),
                       rng.uniform(2.0, 3.4, k), 1.0, np.array([0.90, 0.88, 1.0, 0.9], "f4"),
                       np.array([0.55, 0.55, 1.0, 0.0], "f4"), drag=1.0)

    def _render_flames(self, m_vp_bytes, cam_pos):
        if not self._flames:
            return
        fp = self.flame_prog
        fp["m_vp"].write(m_vp_bytes)
        fp["camPos"].value = tuple(float(c) for c in cam_pos)
        t = time.time() % 1000.0
        fp["time"].value = t
        fp["nozzle"].value = self.NOZZLE
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.disable(moderngl.CULL_FACE)
        for model_bytes, hot, flame in self._flames:
            fp["m_model"].write(model_bytes)
            fl = 0.92 + 0.16 * math.sin(t * 53.0) * math.sin(t * 31.0)
            for L, R0, gain, h, f in ((95.0 * fl, 12.0, 0.85, hot, flame), (48.0 * fl, 6.0, 1.3, (1.0, 1.0, 1.0), hot)):
                fp["L"].value = L
                fp["R0"].value = R0
                fp["gain"].value = gain
                fp["hotCol"].value = tuple(h)
                fp["flameCol"].value = tuple(f)
                self.flame_vao.render(moderngl.TRIANGLES, vertices=self.flame_n)
        self.ctx.enable(moderngl.CULL_FACE)
        self._flames = []

    def ring(self, center, normal, r0, r1, life, width, color, core=(1, 1, 1), arc_dir=None, arcw=-1.0, delay=0.0,
             billboard=False, fill=0.0, sparkle=0.0, ontop=False, mode=0):
        """mode 0 = expanding shockwave ring; 1 = crisp fixed-size badge disc (flip reset);
        2 = soft fixed-size glow disc (jump). Modes 1/2 never grow; they only fade out linearly."""
        g = Ring()
        g.mode = mode
        g.drawn = 0
        g.billboard = billboard
        g.fill = fill
        g.sparkle = sparkle
        g.ontop = ontop
        g.center = np.asarray(center, "f4")
        u, v = _perp_basis(normal)
        g.u, g.v = u.astype("f4"), v.astype("f4")
        g.r0, g.r1, g.life, g.age, g.width = r0, r1, life, -delay, width
        g.color, g.core = color, core
        g.arc = (0.0, 0.0)
        if arc_dir is not None:
            ad = np.asarray(arc_dir, "f8")
            a2 = np.array([ad.dot(u), ad.dot(v)])
            if np.linalg.norm(a2) > 1e-4:
                g.arc = tuple(a2 / np.linalg.norm(a2))
        g.arcw = arcw
        self.rings.append(g)

    def _render_pad_charge(self, m_vp_bytes):
        ctx, rp = self.ctx, self.ring_prog
        rp["m_vp"].write(m_vp_bytes)
        ctx.disable(moderngl.CULL_FACE)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        rp["mode"].value = 4.0
        rp["axisU"].write(np.array([1.0, 0.0, 0.0], "f4").tobytes())
        rp["axisV"].write(np.array([0.0, 1.0, 0.0], "f4").tobytes())
        rp["coreCol"].value = (1.0, 0.92, 0.65)
        for x, y, z, rad, prog in self._pad_charge:
            p = 0.0 if prog is None else float(prog)
            rp["center"].write(np.array([x, y, z], "f4").tobytes())
            rp["radius"].value = float(rad)
            rp["width"].value = 0.15
            rp["fill"].value = p
            rp["color"].value = (0.85 + 0.15 * p, 0.10 + 0.50 * p, 0.03 + 0.07 * p, 0.95)
            self.ring_vao.render(moderngl.TRIANGLES)
        ctx.enable(moderngl.CULL_FACE)
        self._pad_charge = []

    def _render_reset_discs(self, m_vp_bytes):
        """Flip-reset discs on their car's wheel plane, depth-tested: whatever the car or the ball hides
        from the camera is not drawn."""
        ctx, rp = self.ctx, self.ring_prog
        rp["m_vp"].write(m_vp_bytes)
        ctx.disable(moderngl.CULL_FACE)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        rp["mode"].value = 3.0
        rp["coreCol"].value = (1.0, 1.0, 1.0)
        rp["width"].value = 0.07
        rp["radius"].value = RESET_DISC_RADIUS
        for d in self.reset_discs:
            pose = self.car_poses.get(d["car"])
            if pose is None:
                continue
            pos, fwd, up = (np.asarray(v, "f4") for v in pose)
            right = np.cross(up, fwd).astype("f4")
            center = pos + fwd * RESET_DISC_FWD - up * RESET_DISC_BELOW
            d["drawn"] += 1
            k = 1.0 - min(d["age"] / RESET_DISC_LIFE, 1.0)          # linear fade over RESET_DISC_LIFE
            rp["center"].write(center.astype("f4").tobytes())
            rp["axisU"].write(fwd.astype("f4").tobytes())
            rp["axisV"].write(right.tobytes())
            rp["color"].value = (0.93, 0.96, 1.0, 0.95 * k)
            self.ring_vao.render(moderngl.TRIANGLES)
        ctx.enable(moderngl.CULL_FACE)

    # ---------------------------------------------------------------------------------------- #
    def on_event(self, ev, spectated):
        k = ev["kind"]
        pos = np.asarray(ev["pos"], "f4")
        if k == "flipreset":
            # RL v2.66 indicator: a frosted white disc lying TANGENT to the ball at the point where the
            # wheels got the reset (between ball and car), bright rim + sparkles, pops in then fades.
            nrm = np.asarray(ev.get("normal", ev.get("up", (0, 0, 1))), "f4")
            # a crisp white frosted circle at full size from the first frame, then a linear fade: 200 ms total
            # RL's indicator: a white disc in the car's wheel plane, as long as the car, that stays under
            # the car (follows it) for 120 ms -- drawn in render() from the car's current pose.
            car = ev.get("car", -1)
            if car is not None and car >= 0:
                self.reset_discs = [d for d in self.reset_discs if d["car"] != car]
                self.reset_discs.append({"car": car, "age": 0.0, "drawn": 0})
        elif k in ("jump", "doublejump", "dodge"):
            # RL jump burst (jumps, double jumps and flips; not stalls), 120 ms, full size at once, holds then
            # fades. Off a surface: a flat warm glow in the wheel plane at the take-off point, wider than the
            # car so it shows around the body and opens up as the car leaves. In the air (double jump / flip):
            # a soft camera-facing orange ball of light just under the car, hot in the middle (fill=1 selects
            # that profile in the shader), partly hidden by the car like RL's.
            up = np.asarray(ev.get("up", (0, 0, 1)), "f4")
            if k == "jump":
                c = pos - up * RESET_DISC_BELOW                  # wheel-bottom level: just above the surface
                self.ring(c, up, 64.0, 64.0, JUMP_GLOW_LIFE, 0.0, (1.0, 0.32, 0.07, 0.95),
                          core=(1.0, 0.72, 0.38), mode=2)
            else:
                c = pos - up * 24.0
                self.ring(c, up, 50.0, 50.0, JUMP_GLOW_LIFE, 0.0, (1.0, 0.36, 0.08, 1.0),
                          core=(1.0, 0.80, 0.45), mode=2, billboard=True, fill=1.0)
        elif k == "demo":
            # DEMO_SCALE: the whole explosion 20% bigger (sizes, spread speeds, offsets, gravity, ring radii);
            # same timing, same look
            S = DEMO_SCALE
            T = DEMO_TIME

            def _sp(sys_, p, v, life, s0, s1, c0, c1, drag=0.0, grav=0.0):
                # DEMO_TIME: the same explosion played 1.5x slower -- same shapes and reach, lasts 1.5x as long
                sys_.spawn(p, v / T, life * T, s0, s1, c0, c1, drag=drag / T, grav=grav / (T * T))
            vel = np.asarray(ev.get("vel", (0, 0, 0)), "f4") * 0.25
            pos = pos.copy(); pos[2] = max(float(pos[2]), 60.0)
            rng = self._rng
            # 1) blinding flash
            _sp(self.add, pos[None, :], vel[None, :], np.array([0.14], "f4"), np.array([650.0 * S], "f4"),
                           1200.0 * S, np.array([1, 0.97, 0.85, 1.0], "f4"), np.array([1, 0.6, 0.2, 0.0], "f4"))
            # 2) hot core: bright yellow -> orange puffs that STAY fire-coloured while they fade
            n = 70
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 0.8 + 0.2
            _sp(self.alpha, np.repeat(pos[None, :], n, 0) + d * rng.uniform(0, 50 * S, (n, 1)),
                             d * rng.uniform(300 * S, 950 * S, (n, 1)) + vel,
                             rng.uniform(0.35, 0.65, n), rng.uniform(170 * S, 250 * S, n), 360.0 * S,
                             np.array([1.0, 0.86, 0.45, 1.0], "f4"), np.array([1.0, 0.38, 0.06, 0.0], "f4"),
                             drag=4.0, grav=-220.0 * S)
            # 3) outer fire: orange -> deep red, a little longer
            n = 60
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 0.9 + 0.1
            _sp(self.alpha, np.repeat(pos[None, :], n, 0) + d * rng.uniform(20 * S, 90 * S, (n, 1)),
                             d * rng.uniform(450 * S, 1100 * S, (n, 1)) + vel,
                             rng.uniform(0.5, 0.9, n), rng.uniform(140 * S, 200 * S, n), 320.0 * S,
                             np.array([1.0, 0.55, 0.12, 0.9], "f4"), np.array([0.55, 0.10, 0.03, 0.0], "f4"),
                             drag=3.5, grav=-180.0 * S)
            # 4) embers: long-lived bright sparks that arc and fall
            n = 80
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 1.2 + 0.2
            _sp(self.add, np.repeat(pos[None, :], n, 0), d * rng.uniform(700 * S, 2200 * S, (n, 1)) + vel,
                           rng.uniform(0.6, 1.4, n), rng.uniform(7 * S, 13 * S, n), 3.0 * S,
                           np.array([1, 0.92, 0.6, 1.0], "f4"), np.array([1, 0.35, 0.05, 0.0], "f4"),
                           drag=0.9, grav=900.0 * S)
            # 5) debris chunks: dark bits of car flung out
            n = 18
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) + 0.3
            _sp(self.alpha, np.repeat(pos[None, :], n, 0), d * rng.uniform(600 * S, 1500 * S, (n, 1)) + vel,
                             rng.uniform(0.7, 1.2, n), rng.uniform(16 * S, 26 * S, n), 12.0 * S,
                             np.array([0.10, 0.10, 0.11, 1.0], "f4"), np.array([0.08, 0.08, 0.09, 0.0], "f4"),
                             drag=0.6, grav=1300.0 * S)
            # 6) smoke: light grey, thin, rising and spreading, after the fire
            n = 28
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2])
            _sp(self.alpha, np.repeat(pos[None, :], n, 0) + d * 80 * S, d * 220 * S + np.array([0, 0, 260 * S], "f4"),
                             rng.uniform(1.2, 2.0, n), 140.0 * S, 420.0 * S, np.array([0.50, 0.47, 0.45, 0.30], "f4"),
                             np.array([0.42, 0.42, 0.44, 0.0], "f4"), drag=1.6)
        elif k == "goal":
            self._goal_explosion(pos, TEAM_GOAL[int(ev.get("team", 0)) & 1])

    def _goal_explosion(self, pos, col):
        """Goal -- one theme, a SUPERNOVA in the scoring team's colour: a white-hot core flash, two 3D shock spheres
        racing out (glowing at their rims, clear in the middle), a shock ring along the ground, energy sparks riding
        the shock front, and a soft afterglow where it was.
        GOAL_FX_SPEED slows the whole thing down (speeds x v, lifetimes / v)."""
        v = GOAL_FX_SPEED
        rng = self._rng
        col = np.asarray(col, "f4")
        hot = col * 0.35 + 0.65                                   # the colour, nearly white
        pos = np.asarray(pos, "f4")
        # core flash (white-hot) and the afterglow it leaves
        self.add.spawn(pos[None, :], np.zeros((1, 3), "f4"), np.array([0.22 / v], "f4"), np.array([700.0], "f4"),
                       1300.0, np.array([1.0, 1.0, 1.0, 1.0], "f4"), np.array([*hot, 0.0], "f4"))
        self.add.spawn(pos[None, :], np.zeros((1, 3), "f4"), np.array([1.0 / v], "f4"), np.array([420.0], "f4"),
                       650.0, np.array([*col, 0.35], "f4"), np.array([*col, 0.0], "f4"))
        # shock spheres (drawn in _render_shells) + the ground ring
        self.shells.append([pos.copy(), -0.0, 0.75 / v, 80.0, 1500.0, col.copy(), 1.0])
        self.shells.append([pos.copy(), -0.10 / v, 0.85 / v, 60.0, 1150.0, col.copy(), 0.7])
        # the plasma cloud: big soft glowing billows in the team colour thrown out of the core, braking hard and
        # swelling, so the burst has a body that lingers and slowly fades after the shock has passed
        nb = 70
        d = self._rand_dirs(nb)
        d[:, 1] -= 1.1 * np.sign(pos[1])                      # out of the goal mouth, into the field
        d[:, 2] = np.abs(d[:, 2]) * 0.8 + 0.25
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.add.spawn(np.repeat(pos[None, :], nb, 0) + d * 60.0, d * rng.uniform(900 * v, 2400 * v, (nb, 1)),
                       rng.uniform(1.2 / v, 2.0 / v, nb), rng.uniform(220, 380, nb), rng.uniform(650, 950, nb),
                       np.array([*(0.25 * hot + 0.75 * col), 0.20], "f4"), np.array([*col, 0.0], "f4"), drag=1.7 * v)
        # a second, hotter, inner billow layer (white-hot heart of the cloud)
        ni = 24
        d = self._rand_dirs(ni)
        self.add.spawn(np.repeat(pos[None, :], ni, 0), d * rng.uniform(150 * v, 500 * v, (ni, 1)),
                       rng.uniform(0.5 / v, 0.9 / v, ni), rng.uniform(120, 200, ni), rng.uniform(300, 420, ni),
                       np.array([*hot, 0.35], "f4"), np.array([*col, 0.0], "f4"), drag=3.0 * v)
        # glittering embers left hanging in the cloud, drifting down and twinkling out after the burst
        ne = 140
        d = self._rand_dirs(ne)
        d[:, 1] -= 0.9 * np.sign(pos[1])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.add.spawn(np.repeat(pos[None, :], ne, 0) + d * 50.0, d * rng.uniform(700 * v, 1700 * v, (ne, 1)),
                       rng.uniform(1.3 / v, 2.3 / v, ne), rng.uniform(16, 26, ne), 5.0,
                       np.array([*hot, 1.0], "f4"), np.array([*col, 0.0], "f4"), drag=2.0 * v, grav=-120.0 * v * v)
        gp = pos.copy(); gp[2] = 4.0
        self.ring(gp, (0, 0, 1), 80.0, 1700.0, 0.8 / v, 0.05, (*col, 0.9), core=tuple(hot))
        # light rays out of the core
        # energy sparks riding the shock front: one speed band so they stay on the sphere, white-hot -> team colour
        n = 260
        d = self._rand_dirs(n)
        self.add.spawn(np.repeat(pos[None, :], n, 0) + d * 60.0, d * rng.uniform(2300 * v, 2700 * v, (n, 1)),
                       rng.uniform(0.60 / v, 0.80 / v, n), rng.uniform(26, 38, n), 8.0,
                       np.array([*hot, 1.0], "f4"), np.array([*col, 0.0], "f4"), drag=1.3 * v)

    def _render_shells(self, m_vp_bytes, cam_pos):
        if not self.shells:
            return
        sp = self.shell_prog
        sp["m_vp"].write(m_vp_bytes)
        sp["camPos"].value = tuple(float(x) for x in cam_pos)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        for c_, age, life, r0, r1, col, a0 in self.shells:
            if age < 0.0:
                continue
            t = min(age / life, 1.0)
            e = 1.0 - (1.0 - t) ** 3                               # fast out, slowing down
            sp["center"].value = tuple(float(x) for x in c_)
            sp["radius"].value = float(r0 + (r1 - r0) * e)
            sp["col"].value = tuple(float(x) for x in col)
            sp["alpha"].value = float(a0 * (1.0 - t) ** 1.6)
            self.shell_vao.render(moderngl.TRIANGLES, vertices=self.shell_n)
        self.ctx.enable(moderngl.CULL_FACE)

    def sparks(self, pos, normal, strength):
        """A hit: a small, short orange puff with a faint horizontal streak; how big / bright scales with the impact
        (the ball's momentum change for touches). A light touch barely shows."""
        rng = self._rng
        pos = np.asarray(pos, "f4")
        nrm = np.asarray(normal, "f4")
        ln = float(np.linalg.norm(nrm))
        nrm = nrm / ln if ln > 1e-4 else np.array([0, 0, 1], "f4")
        s = float(np.clip((strength - 150.0) / 2400.0, 0.0, 1.0))
        a = 0.18 + 0.47 * s
        self.add.spawn(pos[None, :] + nrm[None, :] * 4.0, nrm[None, :] * 40.0, np.array([0.09 + 0.05 * s], "f4"),
                       np.array([14.0 + 26.0 * s], "f4"), 40.0 + 50.0 * s,
                       np.array([1.0, 0.55, 0.14, a], "f4"), np.array([0.95, 0.25, 0.04, 0.0], "f4"), drag=4.0)
        self.add.spawn(pos[None, :] + nrm[None, :] * 2.0, nrm[None, :] * 20.0, np.array([0.06 + 0.03 * s], "f4"),
                       np.array([7.0 + 12.0 * s], "f4"), 16.0 + 16.0 * s,
                       np.array([1.0, 0.88, 0.55, a], "f4"), np.array([1.0, 0.5, 0.1, 0.0], "f4"), drag=4.0)
        if s > 0.25:
            self.flares.append([pos.copy(), 0.0, 0.08 + 0.04 * s, 40.0 + 60.0 * s, 0.25 + 0.35 * s])

    def _flare_beams(self):
        """The horizontal streak of every live hit flash: a thin bright camera-facing beam along the screen's x axis,
        transparent at both ends and brightest in the middle."""
        for pos, age, life, size, a0 in self.flares:
            t = age / life
            a = a0 * (1.0 - t) ** 1.5
            w = 1.2 + 1.6 * (1.0 - t)
            L = size * (0.55 + 0.6 * t)
            r = self.cam_right.astype("f4") * L
            self._beams.append((np.asarray([pos - r, pos, pos + r], "f4"), np.array([w * 0.6, w, w * 0.6], "f4"),
                                np.array([[1.0, 0.75, 0.35, 0.0], [1.0, 0.92, 0.65, 0.9 * a], [1.0, 0.75, 0.35, 0.0]], "f4")))

    # ---------------------------------------------------------------------------------------- #
    def wheel_glow_for(self, idx, now):
        t = self.wheel_glow.get(idx)
        if t is None:
            return 0.0
        a = now - t
        if a > 0.45:
            return 0.0
        return max(0.0, 1.0 - a / 0.45) ** 1.5

    def update(self, dt):
        dt = min(dt, 0.1)
        self.add.update(dt)
        self.alpha.update(dt)
        n = self.puff_n
        if n:
            self.puff_age[:n] += dt                       # life is exactly 1 s: age == t
            self.puff_p[:n, 2] += self.puff_acc[:n] * self.puff_age[:n] * dt
            alive = self.puff_age[:n] < 1.0
            if not alive.all():
                m = int(alive.sum())
                self.puff_p[:m] = self.puff_p[:n][alive]; self.puff_age[:m] = self.puff_age[:n][alive]
                self.puff_seed[:m] = self.puff_seed[:n][alive]; self.puff_acc[:m] = self.puff_acc[:n][alive]
                self.puff_size[:m] = self.puff_size[:n][alive]
                self.puff_n = m
        for f in self.flares:
            f[1] += dt
        for g_ in self.shells:
            g_[1] += dt
        self.shells = [g_ for g_ in self.shells if g_[1] < g_[2]]
        for d_ in self.domes:
            d_[1] += dt
        self.domes = [d_ for d_ in self.domes if d_[1] < d_[2]]
        self.flares = [f for f in self.flares if f[1] < f[2]]
        for g in self.rings:
            # Fixed-size indicators (flip reset / jump) only start aging once they have been SEEN:
            # with a 50 ms life, one slow frame used to age them out before their first draw (the
            # sound played, the circle never showed). They also always get >= 2 drawn frames.
            if g.mode and g.drawn == 0:
                continue
            g.age += dt
        self.rings = [g for g in self.rings if g.age < g.life or (g.mode and g.drawn < 2)]
        for d in self.reset_discs:
            if d["drawn"]:                           # ages only once it has been on screen
                d["age"] += dt
        self.reset_discs = [d for d in self.reset_discs if d["age"] < RESET_DISC_LIFE or d["drawn"] < 2]

    @staticmethod
    def _strip(pos, cam, width, rgba):
        seg = np.diff(pos, axis=0)
        seg = np.concatenate([seg, seg[-1:]], 0)
        side = _cross(seg, cam[None, :] - pos)
        side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-6
        w = width[:, None]
        v = np.empty((2 * len(pos), 7), "f4")
        v[0::2, 0:3] = pos - side * w
        v[1::2, 0:3] = pos + side * w
        v[0::2, 3:7] = rgba; v[1::2, 3:7] = rgba
        return v

    def _render_trails(self, m_vp_bytes, cam):
        """All queued trails + flame beams -> ONE vectorised strip build and ONE draw."""
        if not self._trails and not self._beams and not self._beam_blocks:
            return
        cam = np.asarray(cam, "f4")
        P, Wd, C, UP = [], [], [], []
        FL = []
        for pos, age, life, width, col, up, flat in self._trails:
            t = np.clip(age / life, 0.0, 1.0)
            rgba = np.empty((len(pos), 4), "f4"); rgba[:, 0:3] = col[:3]
            rgba[:, 3] = col[3] * (1.0 - t) * np.clip(age / 0.03, 0.0, 1.0)
            if up is None or flat:
                wd = width * (1.0 - 0.6 * t)
            else:
                # upright flame wall: jagged spike heights, stable per point (hashed from its position)
                h = np.sin(pos[:, 0] * 12.9898 + pos[:, 1] * 78.233 + pos[:, 2] * 37.719) * 43758.5453
                h = h - np.floor(h)
                spike = np.where(h > 0.78, 1.0 + 1.6 * (h - 0.78) / 0.22, 0.35 + 0.5 * h)
                wd = width * spike * (1.0 - 0.75 * t)
            P.append(pos); Wd.append(wd); C.append(rgba)
            UP.append(None if up is None else np.repeat(up[None, :], len(pos), 0))
            FL.append(bool(flat))
        lens = [len(p) for p in P]
        for pos, width, rgba in self._beams:
            P.append(pos); Wd.append(np.broadcast_to(np.asarray(width, "f4"), (len(pos),))); C.append(rgba)
            lens.append(len(pos))
        for pos, width, rgba in self._beam_blocks:           # (n, m, ...) = n strips of m points at once
            n_, m_ = pos.shape[0], pos.shape[1]
            P.append(pos.reshape(-1, 3)); Wd.append(width.reshape(-1)); C.append(rgba.reshape(-1, 4))
            lens.extend([m_] * n_)
        self._trails = []
        self._beams = []
        self._beam_blocks = []
        lens = np.array(lens)
        pos = np.concatenate(P, 0); wid = np.concatenate(Wd, 0).astype("f4"); rgba = np.concatenate(C, 0)
        # per-point segment direction (last point of each strip reuses its previous segment)
        seg = np.empty_like(pos)
        seg[:-1] = pos[1:] - pos[:-1]
        ends = np.cumsum(lens) - 1
        seg[ends] = seg[ends - 1]
        view = cam[None, :] - pos
        side = np.stack([seg[:, 1] * view[:, 2] - seg[:, 2] * view[:, 1],
                         seg[:, 2] * view[:, 0] - seg[:, 0] * view[:, 2],
                         seg[:, 0] * view[:, 1] - seg[:, 1] * view[:, 0]], 1)
        side *= (wid / (np.sqrt((side * side).sum(1)) + 1e-6))[:, None]
        lo = pos - side; hi = pos + side
        rgba_hi = rgba.copy()
        off = 0
        for k, u in enumerate(UP):                   # (trails only; beams follow) upright: base on surface, tip along up
            nk = lens[k]
            if u is not None and FL[k]:          # flat on the surface: side = seg x up
                sg = seg[off:off + nk]
                sd = _cross(sg, u)
                sd *= (wid[off:off + nk] / (np.sqrt((sd * sd).sum(1)) + 1e-6))[:, None]
                lo[off:off + nk] = pos[off:off + nk] - sd
                hi[off:off + nk] = pos[off:off + nk] + sd
            elif u is not None:
                lo[off:off + nk] = pos[off:off + nk]
                hi[off:off + nk] = pos[off:off + nk] + u * wid[off:off + nk, None]
                rgba_hi[off:off + nk, 3] *= 0.15    # fades toward the tips
            off += nk
        # one triangle list: two triangles per segment between consecutive points of the same strip
        n = len(pos)
        VL = np.empty((n, 7), "f4"); VH = np.empty((n, 7), "f4")
        VL[:, 0:3] = lo; VH[:, 0:3] = hi
        VL[:, 3:7] = rgba; VH[:, 3:7] = rgba_hi
        sid = np.repeat(np.arange(len(lens)), lens)
        i = np.nonzero(sid[:-1] == sid[1:])[0]
        j = i + 1
        data = np.stack([VL[i], VH[i], VL[j], VH[i], VH[j], VL[j]], 1).reshape(-1, 7)[:self.TRAIL_MAX]
        self.trail_vbo.write(data.tobytes())
        self.trail_prog["m_vp"].write(m_vp_bytes)
        # premultiplied "over" (not additive): a blue trail stays blue on green turf instead of turning teal
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        self.trail_vao.render(moderngl.TRIANGLES, vertices=len(data))
        self.ctx.enable(moderngl.CULL_FACE)

    def render(self, m_vp_bytes, px_scale, cam_pos=(0.0, 0.0, 0.0)):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.fbo.depth_mask = False                     # particles test depth but never write it
        self._flare_beams()
        self._nozzle_flare_beams()
        self._render_trails(m_vp_bytes, cam_pos)
        self._render_tubes(m_vp_bytes, cam_pos)
        self._render_tubes3d(m_vp_bytes, cam_pos)
        self._render_flames(m_vp_bytes, cam_pos)
        self._render_domes(m_vp_bytes, cam_pos)
        self._render_shells(m_vp_bytes, cam_pos)
        if self.alpha.n or self.add.n or self._pad_glow is not None or self.puff_n:
            self.prog["m_vp"].write(m_vp_bytes)
            self.prog["pxScale"].value = float(px_scale)
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            na = self.alpha.vertex_data(self.buf)
            nb = self.add.vertex_data(self.buf[na:])
            if self._pad_glow is not None and na + nb + len(self._pad_glow) <= len(self.buf):
                self.buf[na + nb:na + nb + len(self._pad_glow)] = self._pad_glow
                nb += len(self._pad_glow)
            self.vbo.write(self.buf[:na + nb].tobytes())
            if na:
                ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
                self.vao.render(moderngl.POINTS, vertices=na, first=0)
            self._render_puffs(m_vp_bytes, px_scale, cam_pos)
            self.prog["m_vp"].write(m_vp_bytes)
            if nb:
                ctx.blend_func = moderngl.ONE, moderngl.ONE
                self.vao.render(moderngl.POINTS, vertices=nb, first=na)
        if self.rings:
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            rp = self.ring_prog
            rp["m_vp"].write(m_vp_bytes)
            ctx.disable(moderngl.CULL_FACE)
            for g in self.rings:
                if g.age < 0:
                    continue
                g.drawn += 1
                t = min(g.age / g.life, 0.999)
                e = 1.0 - (1.0 - t) ** 3                 # ease-out expansion
                rp["center"].write(g.center.tobytes())
                if g.billboard:
                    rp["axisU"].write(self.cam_right.astype("f4").tobytes())
                    rp["axisV"].write(self.cam_up.astype("f4").tobytes())
                else:
                    rp["axisU"].write(g.u.tobytes()); rp["axisV"].write(g.v.tobytes())
                if g.ontop:
                    ctx.disable(moderngl.DEPTH_TEST)
                # frosted discs are white glass: "over" blending (additive turned them green on turf)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA) if (g.sparkle > 0 or g.mode) else (moderngl.ONE, moderngl.ONE)
                rad = g.r0 + (g.r1 - g.r0) * e
                if g.mode == 1:        # flip-reset badge: never smaller than ~1.3% of the screen height
                    rad = max(rad, 0.018 * float(np.linalg.norm(g.center - np.asarray(cam_pos, "f4"))))
                rp["radius"].value = float(rad)
                rp["mode"].value = float(g.mode)
                if g.mode == 2:                  # jump glow: holds, then fades (reads as its full 150 ms)
                    a = g.color[3] * (1.0 - t * t)
                else:
                    a = g.color[3] * ((1.0 - t) if g.mode else (1.0 - t) ** 1.3)
                rp["color"].value = (g.color[0], g.color[1], g.color[2], a)
                rp["coreCol"].value = tuple(g.core)
                rp["width"].value = float(g.width if g.mode else g.width * (1.0 - 0.5 * t))
                rp["arcDir"].value = g.arc
                rp["arcWidth"].value = float(g.arcw)
                rp["fill"].value = float(g.fill)
                rp["sparkle"].value = float(g.sparkle)
                rp["seed"].value = float(t)
                self.ring_vao.render(moderngl.TRIANGLES)
                if g.ontop:
                    ctx.enable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.CULL_FACE)
        if self._pad_charge:
            self._render_pad_charge(m_vp_bytes)
        if self.reset_discs:
            self._render_reset_discs(m_vp_bytes)
        ctx.fbo.depth_mask = True
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

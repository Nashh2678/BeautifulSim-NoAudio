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

    def spawn(self, pos, vel, life, s0, s1, c0, c1, drag=0.0, grav=0.0):
        k = len(pos)
        if k == 0:
            return
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
        self.domes = []                            # pad pickup glow domes: [pos, age, life, r0, r1]
        self._rng = np.random.default_rng()
        self.trail_prog = ctx.program(vertex_shader=TRAIL_VERT, fragment_shader=TRAIL_FRAG)
        self.trail_vbo = ctx.buffer(reserve=8192 * 7 * 4, dynamic=True)
        self.trail_vao = ctx.vertex_array(self.trail_prog, [(self.trail_vbo, "3f 4f", "in_pos", "in_col")])
        self._pad_glow = None
        self._pad_charge = []
        self._trails = []
        self._tubes = []
        self.tube_prog = ctx.program(vertex_shader=TUBE_VERT, fragment_shader=TUBE_FRAG)
        self.tube_vbo = ctx.buffer(reserve=4096 * 8 * 4, dynamic=True)
        self.tube_vao = ctx.vertex_array(self.tube_prog, [(self.tube_vbo, "3f 4f 1f", "in_pos", "in_col", "in_u")])
        self._beams = []
        self.flame_prog = ctx.program(vertex_shader=FLAME_VERT, fragment_shader=FLAME_FRAG)
        fm = _flame_mesh()
        self.flame_vao = ctx.vertex_array(self.flame_prog, [(ctx.buffer(fm.tobytes()), "2f", "in_sa")])
        self.flame_n = len(fm)
        self._flames = []
        self.cam_right = np.array([1.0, 0.0, 0.0], "f4")
        self.cam_up = np.array([0.0, 0.0, 1.0], "f4")

    def pad_pickup(self, pos, big):
        """Boost pad picked up (RL): a glowing orange-yellow half dome over the pad (a soft sprite whose lower half the
        floor hides), brief -- ~0.1 s on a small pad, longer on a big one. Small pads also puff a few dark specks
        upward; big pads throw a burst of bright golden sparks up and out."""
        rng = self._rng
        pos = np.asarray(pos, "f4").copy(); pos[2] = 0.0
        if big:
            self.domes.append([pos.copy(), 0.0, 0.32, 150.0, 230.0])
            self.domes.append([pos.copy(), 0.0, 0.16, 70.0, 110.0])
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
            self.domes.append([pos.copy(), 0.0, 0.10, 70.0, 95.0])
            n = 9
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 3.0 + 1.0
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            self.alpha.spawn(np.repeat((pos + np.array([0, 0, 8.0], "f4"))[None, :], n, 0) + d * 10.0,
                             d * rng.uniform(250, 520, (n, 1)), rng.uniform(0.25, 0.45, n), rng.uniform(3.0, 5.0, n), 2.0,
                             np.array([0.10, 0.07, 0.04, 0.9], "f4"), np.array([0.10, 0.07, 0.04, 0.0], "f4"), drag=2.0, grav=250.0)

    def _dome_sprites(self):
        """The live pickup domes as additive sprites (appended to the additive particle batch this frame)."""
        out = []
        for pos, age, life, r0, r1 in self.domes:
            t = age / life
            r = r0 + (r1 - r0) * t
            a = (1.0 - t) ** 1.2
            out.append((pos[0], pos[1], pos[2], 1.0, 0.55, 0.08, 0.95 * a, 2.0 * r))
            out.append((pos[0], pos[1], pos[2], 1.0, 0.88, 0.45, 0.85 * a, 1.2 * r))
        return np.asarray(out, "f4") if out else None

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
        pos = np.asarray([tuple(p.pos) for p in pts], "f4")
        age = np.asarray([p.time_active for p in pts], "f4")
        self._trails.append((pos, age, lifetime, width, color, None if up is None else np.asarray(up, "f4"), flat))

    def add_tube(self, ribbon, lifetime, radius, color, white_from=0.0, white_len=0.0):
        """Queue a RibbonEmitter (newest point first) as a round, shaded tube that fades with age. The first
        `white_len` uu after `white_from` (measured along the trail from its head) blend from white to `color`."""
        pts = [p for p in ribbon.points if p.connected]
        if len(pts) < 2:
            return
        base = tuple(color[:3])
        self._tubes.append((np.asarray([tuple(p.pos) for p in pts], "f4"),
                            np.asarray([p.time_active for p in pts], "f4"),
                            np.asarray([getattr(p, "k", 1.0) for p in pts], "f4"), lifetime, radius,
                            np.asarray([getattr(p, "col", None) or base for p in pts], "f4"),   # colour per point
                            float(color[3]) if len(color) > 3 else 1.0, white_from, white_len))

    def _render_tubes(self, m_vp_bytes, cam):
        if not self._tubes:
            return
        cam = np.asarray(cam, "f4")
        out = []
        for pos, age, kpt, life, radius, rgb, a0, w_from, w_len in self._tubes:
            n = len(pos)
            seg = np.empty_like(pos)
            seg[:-1] = pos[1:] - pos[:-1]
            seg[-1] = seg[-2]
            dist = np.concatenate([[0.0], np.cumsum(np.sqrt((seg[:-1] ** 2).sum(1)))]).astype("f4")
            t = np.clip(age / life, 0.0, 1.0)
            w = 1.0 - np.clip((dist - w_from) / max(w_len, 1e-3), 0.0, 1.0) if w_len > 0 else np.zeros(n, "f4")
            rgba = np.empty((n, 4), "f4")
            rgba[:, 0:3] = rgb * (1.0 - w[:, None]) + w[:, None]
            rgba[:, 3] = a0 * (1.0 - t) * kpt              # kpt: per-point strength at emission
            view = cam[None, :] - pos
            side = _cross(seg, view)
            side *= (radius * (1.0 - 0.35 * t) / (np.sqrt((side * side).sum(1)) + 1e-6))[:, None]
            v = np.empty((2 * n, 8), "f4")
            v[0::2, 0:3] = pos - side; v[1::2, 0:3] = pos + side
            v[0::2, 3:7] = rgba; v[1::2, 3:7] = rgba
            v[0::2, 7] = -1.0; v[1::2, 7] = 1.0
            if out:                                  # degenerate bridge between tubes
                out.append(out[-1][-1:]); out.append(v[:1])
            out.append(v)
        self._tubes = []
        data = np.concatenate(out, 0)[:4096]
        self.tube_vbo.write(data.tobytes())
        self.tube_prog["m_vp"].write(m_vp_bytes)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        self.tube_vao.render(moderngl.TRIANGLE_STRIP, vertices=len(data))
        self.ctx.enable(moderngl.CULL_FACE)

    def add_beam(self, p0, p1, w0, w1, c0, c1, mid=None):
        """Camera-facing tapered quad strip p0 -> (mid) -> p1 with per-end width/colour (flames)."""
        pts = [p0, mid, p1] if mid is not None else [p0, p1]
        n = len(pts)
        t = np.linspace(0.0, 1.0, n, dtype="f4")[:, None]
        self._beams.append((np.asarray(pts, "f4"), w0 + (w1 - w0) * t[:, 0],
                            np.asarray(c0, "f4")[None, :] * (1 - t) + np.asarray(c1, "f4")[None, :] * t))

    # ---------------------------------------------------------------------------------------- #
    def _rand_dirs(self, k):
        d = self._rng.normal(size=(k, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    NOZZLE = (-57.0, 0.0, 10.0)     # Octane exhaust, car space
    ALPHA_HOT = (1.0, 0.90, 0.55)    # Alpha Boost: golden-yellow streams, white-hot orange glow at the nozzle. One look for
    ALPHA_FLAME = (1.0, 0.62, 0.08)  # both teams.
    STREAM_OFFSET = 10.0             # the two streams leave the exhaust this far to each side
    STREAM_RATE = 330.0              # particles per second per stream

    def boost(self, key, pos, fwd, up, car_vel, team, dt, model_bytes=None):
        """Boosting car this frame (Alpha Boost look, both teams): at the exhaust a hot orange glow with a thin
        horizontal lens streak; behind it two long, billowing golden flame streams made of ragged fire puffs that stay
        in the air where they were emitted (the car leaves them behind), drifting apart and burning out orange over
        ~1 s. Emission is spread along the path the car took since the last frame (no gaps at speed / low fps)."""
        fwd = np.asarray(fwd, "f4"); up = np.asarray(up, "f4")
        pos = np.asarray(pos, "f4")
        base = pos + fwd * self.NOZZLE[0] + up * self.NOZZLE[2]
        cv = np.asarray(car_vel, "f4")
        rng = self._rng
        fl = float(rng.uniform(0.85, 1.15))
        # nozzle: orange glow + white-hot core (additive, re-spawned every frame = attached to the car)
        self.add.spawn(base[None, :], cv[None, :], np.array([0.02], "f4"), np.array([70.0 * fl], "f4"), 60.0,
                       np.array([1.0, 0.50, 0.08, 0.55], "f4"), np.array([1.0, 0.35, 0.02, 0.0], "f4"))
        self.add.spawn(base[None, :], cv[None, :], np.array([0.02], "f4"), np.array([24.0 * fl], "f4"), 20.0,
                       np.array([1.0, 0.95, 0.75, 0.95], "f4"), np.array([1.0, 0.75, 0.30, 0.0], "f4"))
        self.nozzle_flares.append((base.copy(), fl))
        prev = self._boost_last_base.get(key)
        self._boost_last_base[key] = base.copy()
        if prev is None or float(np.linalg.norm(base - prev)) > 400.0:
            prev = base
        right = np.array([fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2],
                          fwd[0] * up[1] - fwd[1] * up[0]], "f4")
        acc = self._boost_accum.get(key, 0.0) + self.STREAM_RATE * dt
        k = int(acc)
        self._boost_accum[key] = acc - k
        k = min(k, 24)
        if k <= 0:
            return
        tw = self._boost_accum.get(("t", key), 0.0) + dt
        self._boost_accum[("t", key)] = tw
        for side in (-1.0, 1.0):
            f = rng.uniform(0.0, 1.0, (k, 1)).astype("f4")               # where along this frame's path
            wob = np.sin(tw * 9.0 + side * 1.7 + f[:, 0] * dt * 9.0).astype("f4")[:, None] * 3.0
            src = prev[None, :] + (base - prev)[None, :] * f + right[None, :] * (side * self.STREAM_OFFSET) + up[None, :] * wob
            j = rng.normal(0.0, 1.0, (k, 3)).astype("f4")
            p = src + j * 2.5
            # blown back out of the exhaust, then drifting slowly outward / up
            v = cv[None, :] * 0.05 - fwd[None, :] * rng.uniform(80.0, 260.0, (k, 1)).astype("f4") + j * 22.0 \
                + right[None, :] * side * 18.0 + np.array([0, 0, 18.0], "f4")
            life = rng.uniform(0.70, 1.15, k).astype("f4")
            c0 = np.empty((k, 4), "f4"); c1 = np.empty((k, 4), "f4")
            m = rng.uniform(0.0, 1.0, (k, 1)).astype("f4")
            c0[:, :3] = np.array([1.0, 0.80, 0.16], "f4")[None, :] * m + np.array([1.0, 0.52, 0.04], "f4")[None, :] * (1.0 - m)
            c0[:, 3] = rng.uniform(0.85, 1.0, k)
            c1[:, :3] = np.array([0.72, 0.16, 0.02], "f4")[None, :]
            c1[:, 3] = 0.0
            # negative size = flame puff (ragged, noisy) in PARTICLE_FRAG; "over" blended (alpha pool) so the fire keeps
            # its golden colour instead of adding up to white
            self.alpha.spawn(p, v, life, -rng.uniform(12.0, 19.0, k).astype("f4"), -34.0, c0, c1, drag=1.6)
            # hot glowing tongues inside the stream (additive, shorter-lived): the fire glows instead of reading as smoke
            kh = max(1, k // 2)
            self.add.spawn(p[:kh], v[:kh], rng.uniform(0.18, 0.35, kh).astype("f4"), -rng.uniform(8.0, 13.0, kh).astype("f4"),
                           -18.0, np.array([1.0, 0.85, 0.40, 0.45], "f4"), np.array([1.0, 0.45, 0.05, 0.0], "f4"), drag=1.6)

    def _nozzle_flare_beams(self):
        for p, fl in self.nozzle_flares:
            r = self.cam_right.astype("f4") * 95.0 * fl
            self._beams.append((np.asarray([p - r, p, p + r], "f4"), np.array([0.8, 2.2, 0.8], "f4"),
                                np.array([[1.0, 0.55, 0.15, 0.0], [1.0, 0.80, 0.45, 0.8], [1.0, 0.55, 0.15, 0.0]], "f4")))
        self.nozzle_flares = []

    def sparkle(self, key, pos, dt):
        """Supersonic trail speckle: a few tiny white dots left along the thin trail."""
        rng = self._rng
        acc = self._boost_accum.get(("sp", key), 0.0) + 70.0 * dt
        k = int(acc)
        self._boost_accum[("sp", key)] = acc - k
        k = min(k, 3)
        if k <= 0:
            return
        j = rng.normal(0.0, 1.0, (k, 3)).astype("f4")
        self.add.spawn(np.asarray(pos, "f4")[None, :] + j * 2.5, j * 25.0, rng.uniform(0.18, 0.34, k),
                       rng.uniform(2.0, 3.6, k), 1.2, np.array([1.0, 1.0, 1.0, 0.9], "f4"),
                       np.array([0.85, 0.9, 1.0, 0.0], "f4"), drag=1.0)

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
            for L, R0, gain, h, f in ((150.0 * fl, 13.0, 1.0, hot, flame), (65.0 * fl, 6.0, 1.2, (1.0, 1.0, 1.0), hot)):
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
            # GOAL_FX_SPEED: the goal explosion plays at 70% speed -- same shapes and extent, just slower
            # (speeds and growth x0.7, lifetimes / 0.7, drag x0.7, gravity x0.7^2)
            v = GOAL_FX_SPEED
            col = TEAM_GOAL[int(ev.get("team", 0)) & 1]
            self.add.spawn(pos[None, :], np.zeros((1, 3), "f4"), np.array([0.35 / v], "f4"), np.array([1500.0], "f4"),
                           2600.0 * v, np.array([*col, 1.0], "f4"), np.array([*col, 0.0], "f4"))
            n = 220
            d = self._rand_dirs(n)
            self.add.spawn(np.repeat(pos[None, :], n, 0), d * self._rng.uniform(600 * v, 2600 * v, (n, 1)),
                           self._rng.uniform(0.7 / v, 1.8 / v, n), self._rng.uniform(45, 110, n), 8.0 * v,
                           np.array([1, 1, 1, 1.0], "f4"), np.array([*col, 0.0], "f4"), drag=1.4 * v, grav=300.0 * v * v)
            self.ring(pos, (0, 1, 0), 60.0, 1600.0, 0.7 / v, 0.07, (*col, 1.0), core=(1, 1, 1))
            self.ring(pos, (0, 0, 1), 60.0, 1300.0, 0.8 / v, 0.05, (*col, 0.8), core=(1, 1, 1), delay=0.08 / v)

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
        for f in self.flares:
            f[1] += dt
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
        if not self._trails and not self._beams:
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
        for pos, width, rgba in self._beams:
            P.append(pos); Wd.append(np.broadcast_to(np.asarray(width, "f4"), (len(pos),)).copy()); C.append(rgba); UP.append(None)
            FL.append(False)
        self._trails = []
        self._beams = []
        lens = np.array([len(p) for p in P])
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
        for k, u in enumerate(UP):                   # upright strips: base on the surface, tip along car-up
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
        v = np.empty((2 * len(pos), 7), "f4")
        v[0::2, 0:3] = lo; v[1::2, 0:3] = hi
        v[0::2, 3:7] = rgba; v[1::2, 3:7] = rgba_hi
        # stitch strips with degenerate bridges: repeat last vertex of strip k and first of strip k+1
        starts2 = 2 * (np.cumsum(lens) - lens)
        ends2 = 2 * np.cumsum(lens) - 1
        pieces = []
        for k in range(len(lens)):
            if k:
                pieces.append(v[ends2[k - 1]:ends2[k - 1] + 1]); pieces.append(v[starts2[k]:starts2[k] + 1])
            pieces.append(v[starts2[k]:ends2[k] + 1])
        data = np.concatenate(pieces, 0)[:8192]
        self.trail_vbo.write(data.tobytes())
        self.trail_prog["m_vp"].write(m_vp_bytes)
        # premultiplied "over" (not additive): a blue trail stays blue on green turf instead of turning teal
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        self.trail_vao.render(moderngl.TRIANGLE_STRIP, vertices=len(data))
        self.ctx.enable(moderngl.CULL_FACE)

    def render(self, m_vp_bytes, px_scale, cam_pos=(0.0, 0.0, 0.0)):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.fbo.depth_mask = False                     # particles test depth but never write it
        self._flare_beams()
        self._nozzle_flare_beams()
        self._render_trails(m_vp_bytes, cam_pos)
        self._render_tubes(m_vp_bytes, cam_pos)
        self._render_flames(m_vp_bytes, cam_pos)
        if self.alpha.n or self.add.n or self._pad_glow is not None or self.domes:
            self.prog["m_vp"].write(m_vp_bytes)
            self.prog["pxScale"].value = float(px_scale)
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            na = self.alpha.vertex_data(self.buf)
            nb = self.add.vertex_data(self.buf[na:])
            if self._pad_glow is not None and na + nb + len(self._pad_glow) <= len(self.buf):
                self.buf[na + nb:na + nb + len(self._pad_glow)] = self._pad_glow
                nb += len(self._pad_glow)
            dm = self._dome_sprites()
            if dm is not None and na + nb + len(dm) <= len(self.buf):
                self.buf[na + nb:na + nb + len(dm)] = dm
                nb += len(dm)
            self.vbo.write(self.buf[:na + nb].tobytes())
            if na:
                ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
                self.vao.render(moderngl.POINTS, vertices=na, first=0)
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

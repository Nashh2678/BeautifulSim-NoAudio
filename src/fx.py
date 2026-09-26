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
RESET_DISC_RADIUS = 59.0
RESET_DISC_FWD = 9.0
RESET_DISC_BELOW = 14.0
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


def _perp_basis(n):
    n = np.asarray(n, "f8")
    n = n / max(np.linalg.norm(n), 1e-6)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


class ParticlePool:
    """Structure-of-arrays pool. col0 -> col1 and size0 -> size1 are interpolated over life."""

    def __init__(self, cap):
        self.cap = cap
        self.n = 0
        z3 = lambda: np.zeros((cap, 3), "f4")
        self.pos, self.vel = z3(), z3()
        self.age = np.zeros(cap, "f4"); self.life = np.ones(cap, "f4")
        self.s0 = np.zeros(cap, "f4"); self.s1 = np.zeros(cap, "f4")
        self.c0 = np.zeros((cap, 4), "f4"); self.c1 = np.zeros((cap, 4), "f4")
        self.drag = np.zeros(cap, "f4"); self.grav = np.zeros(cap, "f4")

    def spawn(self, pos, vel, life, s0, s1, c0, c1, drag=0.0, grav=0.0):
        k = len(pos)
        if k == 0:
            return
        if self.n + k > self.cap:                  # full: drop the oldest particles
            drop = self.n + k - self.cap
            self._compact(np.arange(drop, self.n))
        i0, i1 = self.n, self.n + k
        self.pos[i0:i1] = pos; self.vel[i0:i1] = vel
        self.age[i0:i1] = 0.0; self.life[i0:i1] = life
        self.s0[i0:i1] = s0; self.s1[i0:i1] = s1
        self.c0[i0:i1] = c0; self.c1[i0:i1] = c1
        self.drag[i0:i1] = drag; self.grav[i0:i1] = grav
        self.n = i1

    def _compact(self, keep_idx):
        m = len(keep_idx)
        for a in (self.pos, self.vel, self.age, self.life, self.s0, self.s1, self.c0, self.c1, self.drag, self.grav):
            a[:m] = a[keep_idx]
        self.n = m

    def update(self, dt):
        n = self.n
        if n == 0:
            return
        self.age[:n] += dt
        alive = self.age[:n] < self.life[:n]
        if not alive.all():
            self._compact(np.nonzero(alive)[0])
            n = self.n
            if n == 0:
                return
        v = self.vel[:n]
        v *= np.exp(-self.drag[:n] * dt)[:, None]
        v[:, 2] -= self.grav[:n] * dt
        self.pos[:n] += v * dt

    def vertex_data(self, out):
        """Fill out[(n, 8)] = pos(3) col(4) size(1)."""
        n = self.n
        t = (self.age[:n] / self.life[:n])[:, None]
        out[:n, 0:3] = self.pos[:n]
        out[:n, 3:7] = self.c0[:n] + (self.c1[:n] - self.c0[:n]) * t
        out[:n, 7] = self.s0[:n] + (self.s1[:n] - self.s0[:n]) * t[:, 0]
        return n


class Ring:
    __slots__ = ("center", "u", "v", "r0", "r1", "life", "age", "width", "color", "core", "arc", "arcw", "delay",
                 "billboard", "fill", "sparkle", "ontop", "mode", "drawn")


class FX:
    CAP = 6000

    def __init__(self, ctx):
        self.ctx = ctx
        self.add = ParticlePool(self.CAP)          # additive: flames, sparks, flashes
        self.alpha = ParticlePool(2000)            # premultiplied-over: smoke
        self.rings = []
        self.reset_discs = []                      # flip-reset indicators, attached to their car
        self.car_poses = {}                        # car index -> (pos, forward, up), set each frame by main
        self.wheel_glow = {}                       # car idx -> time of last flip reset
        self.screen_flash = 0.0                    # time of last spectated flip reset (2D streaks)
        self.prog = ctx.program(vertex_shader=PARTICLE_VERT, fragment_shader=PARTICLE_FRAG)
        self.buf = np.zeros((self.CAP + 2000 + 128, 8), "f4")
        self.vbo = ctx.buffer(reserve=self.buf.nbytes, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 4f 1f", "in_pos", "in_col", "in_size")])
        self.ring_prog = ctx.program(vertex_shader=RING_VERT, fragment_shader=RING_FRAG)
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4")
        self.ring_vao = ctx.vertex_array(self.ring_prog, [(ctx.buffer(quad.tobytes()), "2f", "in_xy")])
        self._boost_accum = {}
        self._rng = np.random.default_rng()
        self.trail_prog = ctx.program(vertex_shader=TRAIL_VERT, fragment_shader=TRAIL_FRAG)
        self.trail_vbo = ctx.buffer(reserve=8192 * 7 * 4, dynamic=True)
        self.trail_vao = ctx.vertex_array(self.trail_prog, [(self.trail_vbo, "3f 4f", "in_pos", "in_col")])
        self._pad_glow = None
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
        """Big boost pad picked up: a short, subtle spray of golden sparks (no floor ring).
        Small pads get no pickup effect at all (RL only flashes the big canisters)."""
        if not big:
            return
        pos = np.asarray(pos, "f4").copy(); pos[2] = 40.0
        n = 10
        d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 2.5 + 0.8
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.add.spawn(np.repeat(pos[None, :], n, 0), d * self._rng.uniform(200, 520, (n, 1)),
                       self._rng.uniform(0.2, 0.38, n), self._rng.uniform(5, 9, n), 1.5,
                       np.array([1.0, 0.85, 0.4, 0.9], "f4"), np.array([1.0, 0.45, 0.05, 0.0], "f4"), drag=3.0, grav=400.0)

    def pad_glows(self, pads, t):
        """pads: [(x, y, is_big)] for ACTIVE pads -> glow sprites drawn this frame (additive)."""
        if not pads:
            self._pad_glow = None
            return
        n = len(pads)
        g = np.zeros((2 * n, 8), "f4")
        k = 0
        for i, (x, y, big) in enumerate(pads):
            pulse = 0.85 + 0.15 * math.sin(t * 3.0 + i * 1.7)
            if big:
                g[k] = (x, y, 72.0, 1.0, 0.62, 0.15, 0.85 * pulse, 170.0 * pulse); k += 1   # orb
                g[k] = (x, y, 12.0, 1.0, 0.55, 0.10, 0.45, 300.0); k += 1                     # floor glow
            else:
                g[k] = (x, y, 10.0, 1.0, 0.62, 0.15, 0.55 * pulse, 110.0); k += 1
        self._pad_glow = g[:k]

    def add_trail(self, ribbon, lifetime, width, color, up=None):
        """Queue a RibbonEmitter's connected points as a camera-facing, additive strip that fades
        with age and tapers at both ends (drawn batched in render())."""
        pts = [p for p in ribbon.points if p.connected]
        if len(pts) < 2:
            return
        pos = np.asarray([tuple(p.pos) for p in pts], "f4")
        age = np.asarray([p.time_active for p in pts], "f4")
        self._trails.append((pos, age, lifetime, width, color, None if up is None else np.asarray(up, "f4")))

    def add_tube(self, ribbon, lifetime, radius, color, white_from=0.0, white_len=0.0):
        """Queue a RibbonEmitter (newest point first) as a round, shaded tube that fades with age. The first
        `white_len` uu after `white_from` (measured along the trail from its head) blend from white to `color`."""
        pts = [p for p in ribbon.points if p.connected]
        if len(pts) < 2:
            return
        self._tubes.append((np.asarray([tuple(p.pos) for p in pts], "f4"),
                            np.asarray([p.time_active for p in pts], "f4"), lifetime, radius,
                            np.asarray(color[:3], "f4"), float(color[3]) if len(color) > 3 else 1.0,
                            white_from, white_len))

    def _render_tubes(self, m_vp_bytes, cam):
        if not self._tubes:
            return
        cam = np.asarray(cam, "f4")
        out = []
        for pos, age, life, radius, rgb, a0, w_from, w_len in self._tubes:
            n = len(pos)
            seg = np.empty_like(pos)
            seg[:-1] = pos[1:] - pos[:-1]
            seg[-1] = seg[-2]
            dist = np.concatenate([[0.0], np.cumsum(np.sqrt((seg[:-1] ** 2).sum(1)))]).astype("f4")
            t = np.clip(age / life, 0.0, 1.0)
            w = 1.0 - np.clip((dist - w_from) / max(w_len, 1e-3), 0.0, 1.0) if w_len > 0 else np.zeros(n, "f4")
            rgba = np.empty((n, 4), "f4")
            rgba[:, 0:3] = rgb[None, :] * (1.0 - w[:, None]) + w[:, None]
            rgba[:, 3] = a0 * (1.0 - t)
            view = cam[None, :] - pos
            side = np.cross(seg, view)
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

    def boost(self, key, pos, fwd, up, car_vel, team, dt, model_bytes=None):
        """Boosting car this frame: a 3D flame cone attached to the car (drawn with the car's model
        matrix, so it never turns with the camera) + nozzle glow + a few sparks."""
        fwd = np.asarray(fwd, "f4"); up = np.asarray(up, "f4")
        base = np.asarray(pos, "f4") + fwd * self.NOZZLE[0] + up * self.NOZZLE[2]
        cv = np.asarray(car_vel, "f4")
        hot, flame = TEAM_BOOST[int(team) & 1]
        if model_bytes is not None:
            self._flames.append((model_bytes, hot, flame))
        fl = float(self._rng.uniform(0.9, 1.1))
        self.add.spawn(base[None, :], cv[None, :], np.array([0.02], "f4"), np.array([50.0 * fl], "f4"), 40.0,
                       np.array([*hot, 0.65], "f4"), np.array([*flame, 0.0], "f4"))
        # smoke trail left in the world behind the car (RL's standard boost leaves a smoky wake)
        sacc = self._boost_accum.get(("smoke", key), 0.0) + 70.0 * dt
        ks = int(sacc)
        self._boost_accum[("smoke", key)] = sacc - ks
        if ks > 0:
            ks = min(ks, 6)
            jit = self._rng.normal(0.0, 1.0, (ks, 3)).astype("f4")
            sp = base[None, :] - fwd[None, :] * self._rng.uniform(40, 110, (ks, 1)) + jit * 6.0
            sv = cv[None, :] * 0.25 - fwd[None, :] * 150.0 + jit * 40.0 + np.array([0, 0, 60], "f4")
            tint = np.array([0.55, 0.55, 0.58, 0.42], "f4")
            tint[:3] = tint[:3] * 0.8 + np.asarray(flame, "f4") * 0.2
            self.alpha.spawn(sp, sv, self._rng.uniform(0.5, 0.9, ks), self._rng.uniform(22, 32, ks), 95.0,
                             tint, np.array([0.45, 0.45, 0.48, 0.0], "f4"), drag=2.0)
        rate = 120.0
        acc = self._boost_accum.get(key, 0.0) + rate * dt
        k = int(acc)
        self._boost_accum[key] = acc - k
        if k <= 0:
            return
        k = min(k, 8)
        right = np.array([fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2],
                          fwd[0] * up[1] - fwd[1] * up[0]], "f4")
        spread = self._rng.normal(0.0, 1.0, (k, 2)).astype("f4")
        p = base - fwd * self._rng.uniform(20, 90, (k, 1)) + (spread[:, :1] * right + spread[:, 1:] * up) * 4.0
        v = -fwd * self._rng.uniform(300, 600, (k, 1)) + cv + (spread[:, :1] * right + spread[:, 1:] * up) * 120.0
        self.add.spawn(p, v, self._rng.uniform(0.08, 0.18, k), self._rng.uniform(4, 6, k), 1.0,
                       np.array([*hot, 0.9], "f4"), np.array([*flame, 0.0], "f4"), drag=3.0)

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
        elif k in ("jump", "doublejump"):
            # RL jump burst: a quick warm flash of light under the chassis, ~75 ms
            up = np.asarray(ev.get("up", (0, 0, 1)), "f4")
            # small red-orange glow under the chassis, full size at once, linear fade over 200 ms
            c = pos - up * 8.0                                   # under the chassis, above the floor at take-off
            self.ring(c, up, 48.0, 48.0, 0.20, 0.0, (1.0, 0.30, 0.06, 0.9), core=(1.0, 0.70, 0.35), mode=2,
                      billboard=True)        # camera-facing: visible from the chase cam at take-off too
        elif k == "demo":
            vel = np.asarray(ev.get("vel", (0, 0, 0)), "f4") * 0.25
            pos = pos.copy(); pos[2] = max(float(pos[2]), 60.0)
            rng = self._rng
            # 1) blinding flash
            self.add.spawn(pos[None, :], vel[None, :], np.array([0.14], "f4"), np.array([650.0], "f4"),
                           1200.0, np.array([1, 0.97, 0.85, 1.0], "f4"), np.array([1, 0.6, 0.2, 0.0], "f4"))
            # 2) hot core: bright yellow -> orange puffs that STAY fire-coloured while they fade
            n = 70
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 0.8 + 0.2
            self.alpha.spawn(np.repeat(pos[None, :], n, 0) + d * rng.uniform(0, 50, (n, 1)),
                             d * rng.uniform(300, 950, (n, 1)) + vel,
                             rng.uniform(0.35, 0.65, n), rng.uniform(170, 250, n), 360.0,
                             np.array([1.0, 0.86, 0.45, 1.0], "f4"), np.array([1.0, 0.38, 0.06, 0.0], "f4"),
                             drag=4.0, grav=-220.0)
            # 3) outer fire: orange -> deep red, a little longer
            n = 60
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 0.9 + 0.1
            self.alpha.spawn(np.repeat(pos[None, :], n, 0) + d * rng.uniform(20, 90, (n, 1)),
                             d * rng.uniform(450, 1100, (n, 1)) + vel,
                             rng.uniform(0.5, 0.9, n), rng.uniform(140, 200, n), 320.0,
                             np.array([1.0, 0.55, 0.12, 0.9], "f4"), np.array([0.55, 0.10, 0.03, 0.0], "f4"),
                             drag=3.5, grav=-180.0)
            # 4) embers: long-lived bright sparks that arc and fall
            n = 80
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) * 1.2 + 0.2
            self.add.spawn(np.repeat(pos[None, :], n, 0), d * rng.uniform(700, 2200, (n, 1)) + vel,
                           rng.uniform(0.6, 1.4, n), rng.uniform(7, 13, n), 3.0,
                           np.array([1, 0.92, 0.6, 1.0], "f4"), np.array([1, 0.35, 0.05, 0.0], "f4"),
                           drag=0.9, grav=900.0)
            # 5) debris chunks: dark bits of car flung out
            n = 18
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2]) + 0.3
            self.alpha.spawn(np.repeat(pos[None, :], n, 0), d * rng.uniform(600, 1500, (n, 1)) + vel,
                             rng.uniform(0.7, 1.2, n), rng.uniform(16, 26, n), 12.0,
                             np.array([0.10, 0.10, 0.11, 1.0], "f4"), np.array([0.08, 0.08, 0.09, 0.0], "f4"),
                             drag=0.6, grav=1300.0)
            # 6) smoke: light grey, thin, rising and spreading, after the fire
            n = 28
            d = self._rand_dirs(n); d[:, 2] = np.abs(d[:, 2])
            self.alpha.spawn(np.repeat(pos[None, :], n, 0) + d * 80, d * 220 + np.array([0, 0, 260], "f4"),
                             rng.uniform(1.2, 2.0, n), 140.0, 420.0, np.array([0.50, 0.47, 0.45, 0.30], "f4"),
                             np.array([0.42, 0.42, 0.44, 0.0], "f4"), drag=1.6)
            # 7) shockwave rings (ground + vertical)
            self.ring(pos, (0, 0, 1), 40.0, 760.0, 0.35, 0.07, (1.0, 0.65, 0.25, 0.9), core=(1.0, 0.95, 0.85))
            self.ring(pos, (0, 1, 0), 40.0, 520.0, 0.28, 0.06, (1.0, 0.75, 0.35, 0.7), core=(1.0, 0.95, 0.85), billboard=True)
        elif k == "goal":
            col = TEAM_GOAL[int(ev.get("team", 0)) & 1]
            self.add.spawn(pos[None, :], np.zeros((1, 3), "f4"), np.array([0.35], "f4"), np.array([1500.0], "f4"),
                           2600.0, np.array([*col, 1.0], "f4"), np.array([*col, 0.0], "f4"))
            n = 220
            d = self._rand_dirs(n)
            self.add.spawn(np.repeat(pos[None, :], n, 0), d * self._rng.uniform(600, 2600, (n, 1)),
                           self._rng.uniform(0.7, 1.8, n), self._rng.uniform(45, 110, n), 8.0,
                           np.array([1, 1, 1, 1.0], "f4"), np.array([*col, 0.0], "f4"), drag=1.4, grav=300.0)
            self.ring(pos, (0, 1, 0), 60.0, 1600.0, 0.7, 0.07, (*col, 1.0), core=(1, 1, 1))
            self.ring(pos, (0, 0, 1), 60.0, 1300.0, 0.8, 0.05, (*col, 0.8), core=(1, 1, 1), delay=0.08)

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
        side = np.cross(seg, cam[None, :] - pos)
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
        for pos, age, life, width, col, up in self._trails:
            t = np.clip(age / life, 0.0, 1.0)
            rgba = np.empty((len(pos), 4), "f4"); rgba[:, 0:3] = col[:3]
            rgba[:, 3] = col[3] * (1.0 - t) * np.clip(age / 0.03, 0.0, 1.0)
            if up is None:
                wd = width * (1.0 - 0.6 * t)
            else:
                # upright flame wall: jagged spike heights, stable per point (hashed from its position)
                h = np.sin(pos[:, 0] * 12.9898 + pos[:, 1] * 78.233 + pos[:, 2] * 37.719) * 43758.5453
                h = h - np.floor(h)
                spike = np.where(h > 0.78, 1.0 + 1.6 * (h - 0.78) / 0.22, 0.35 + 0.5 * h)
                wd = width * spike * (1.0 - 0.75 * t)
            P.append(pos); Wd.append(wd); C.append(rgba)
            UP.append(None if up is None else np.repeat(up[None, :], len(pos), 0))
        for pos, width, rgba in self._beams:
            P.append(pos); Wd.append(np.full(len(pos), width, "f4")); C.append(rgba); UP.append(None)
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
            if u is not None:
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
        self._render_trails(m_vp_bytes, cam_pos)
        self._render_tubes(m_vp_bytes, cam_pos)
        self._render_flames(m_vp_bytes, cam_pos)
        if self.alpha.n or self.add.n or self._pad_glow is not None:
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
        if self.reset_discs:
            self._render_reset_discs(m_vp_bytes)
        ctx.fbo.depth_mask = True
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

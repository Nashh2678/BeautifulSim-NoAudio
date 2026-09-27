import os
import json
import sys

# numpy's OpenBLAS pre-commits a scratch buffer per CPU thread at import (~480 MB of Windows commit
# on this 16-thread machine) -- the vis only does tiny matrix math, and training runs near the commit
# limit. Must be set before numpy is first imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


# PyOpenGL checks glGetError after EVERY call by default (a round-trip each) -- pure overhead here.
import OpenGL
OpenGL.ERROR_CHECKING = False
import math
import random
import threading
import sys 
import argparse
import copy
import struct
import time

import fastvec  # fast pyrr Vector3 operators -- must come before any vector math
from const import *
from shaders import *
from arena_shaders import *
from socket_listener import SocketListener
from state_manager import *
import state_manager   # also need the module itself for LIVE attrs (pose_cam, hud_text, capture)
from ribbon import *
# outline_renderer (OutlineRenderer) is disabled -- not imported (it cost ~1 s at startup)
from clip_recorder import ClipRecorder
import ui
from ui import get_ui, QUIBarWidget, QRSVWindow
from config import Config, ConfigVal
import audio as rl_audio
import events as rl_events
import fx as rl_fx
import rl_shaders
import carrig
import landscape

import moderngl
import moderngl_window
import moderngl_window.loaders.scene.wavefront as wvf
from moderngl_window import resources
from moderngl_window.meta import TextureDescription

from PyQt5 import QtOpenGL, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QScreen, QColor

# (PyOpenGL's GL/GLU star imports cost ~0.6 s at startup and were only used for GL_LINES)

import numpy as np

from pyrr import Quaternion, Matrix33, Matrix44, Vector3, Vector4

import pywavefront

# Set RSV_PERF=1 to print render fps + per-stage timing (deepcopy / render / capture) to stdout.
PERF = bool(os.environ.get("RSV_PERF"))

# TODO: Move
def safe_normalize(vec: pyrr.Vector3):
    length = max(vec.length, 1e-6)
    return vec / length

# Scene render budget. The 3D scene is drawn into its own MSAA framebuffer at the window size, scaled
# down only if that exceeds RSV_MAX_RENDER_MP megapixels (default 2.1 ~= the old 1080p cap), then upscaled to
# the window (2.1 MP ~= the old 1080p cap: the vis usually runs on the 780M iGPU); HUD/overlays are drawn afterwards at full native resolution so they stay crisp.
RENDER_SCALE = float(os.environ.get("RSV_RENDER_SCALE", "1.0"))
MAX_RENDER_MP = float(os.environ.get("RSV_MAX_RENDER_MP", "2.1"))
MSAA = int(os.environ.get("RSV_MSAA", "4"))
_SKIP = set(os.environ.get("RSV_SKIP", "").split(","))   # dev: skip passes to profile GPU cost
TEAM_BODY = [(0.030, 0.20, 0.95), (1.0, 0.30, 0.02)]      # linear-space team paint (blue, orange)


def _icosphere(subdiv):
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(v, "f8") / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    ico_faces = list(faces)
    for _ in range(subdiv):
        cache = {}

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[k] = len(verts) - 1
            return cache[k]
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    v = np.array(verts, "f4")
    base = v[:12]
    centres = np.array([v[list(f)].mean(0) for f in ico_faces], "f4")
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return v, np.array(faces, "i4"), np.concatenate([base, centres], 0)


# TODO: Move game logic out of here
def _streak_points(pos):
    """The car MODEL's tips the flip streaks come off (read from the loaded mesh, car space): both spoiler tips
    (rear, outermost and highest) and both front fender corners (front, outermost and highest)."""
    P = np.asarray(pos, "f4")
    out = []
    for rear in (False, True):
        m0 = P[:, 0] < -20.0 if rear else P[:, 0] > 30.0
        for side in (1.0, -1.0):
            q = P[m0 & (np.sign(P[:, 1]) == side)]
            if len(q) == 0:
                continue
            score = np.abs(q[:, 1]) + q[:, 2] + (-0.3 if rear else 0.5) * q[:, 0]
            out.append(tuple(float(v) for v in q[int(np.argmax(score))]))
    return out


def _octane_from_obj(obj_path, tex_path):
    """RocketSimVis's own Octane.obj -> the Octane_RL.npz layout (pos, nrm, per-vertex material id for
    the car shader: 0 team paint, 1 trim, 2 tire, 3 metal, 5 lights, 7 matte black), classified from
    the colour of its texture at each vertex's UV. Wheels are baked into this body: no separate wheels."""
    from PIL import Image
    obj = pywavefront.Wavefront(obj_path, collect_faces=True, create_materials=True, parse=True)
    tex = np.asarray(Image.open(tex_path).convert("RGB"), "f4") / 255.0
    th, tw = tex.shape[:2]
    pos, nrm, mat = [], [], []
    for m in obj.materials.values():
        fmt = m.vertex_format                       # e.g. "T2F_N3F_V3F"
        stride = {"T2F_N3F_V3F": 8, "N3F_V3F": 6, "T2F_V3F": 5, "V3F": 3}[fmt]
        v = np.asarray(m.vertices, "f4").reshape(-1, stride)
        p = v[:, -3:]
        n = v[:, -6:-3] if "N3F" in fmt else np.tile([0.0, 0.0, 1.0], (len(v), 1))
        if "T2F" in fmt:
            uv = v[:, :2]
            x = np.clip((uv[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
            y = np.clip((1.0 - uv[:, 1] % 1.0) * th, 0, th - 1).astype(int)
            c = tex[y, x]
        else:
            c = np.tile([0.5, 0.6, 0.8], (len(v), 1))
        mx, mn = c.max(1), c.min(1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        k = np.full(len(v), 1.0, "f4")                                  # trim
        k[mx < 0.09] = 2.0                                              # tires
        k[(mx >= 0.09) & (mx < 0.16)] = 7.0                             # matte black
        k[(sat < 0.2) & (mx >= 0.35)] = 3.0                             # metal / rims
        k[(sat >= 0.25) & (c[:, 2] > c[:, 0])] = 0.0                    # blue = team paint
        k[(sat >= 0.25) & (c[:, 0] > c[:, 2]) & (mx > 0.5)] = 5.0       # yellow-white = lights
        pos.append(p); nrm.append(n); mat.append(k)
    out = {"pos": np.concatenate(pos), "nrm": np.concatenate(nrm), "mat": np.concatenate(mat)}
    empty = np.zeros((0, 3), "f4")
    for w in range(4):
        out["w%d_pos" % w], out["w%d_nrm" % w], out["w%d_mat" % w] = empty, empty, np.zeros(0, "f4")
    # RocketSim Octane wheel anchors (only used by the suspension rig)
    out["wheel_centers"] = np.array([[51.2, -25.0, -3.3], [51.2, 24.8, -4.0], [-33.7, -28.1, -1.8], [-33.8, 27.7, -2.6]], "f4")
    out["wheel_radius"] = np.array([12.5, 12.5, 15.0, 15.0], "f4")
    return out


class RSVRenderer:
    """All rendering + camera + effects/sound logic, independent of Qt: it draws into whatever
    framebuffer it is handed. QRSVGLWidget (bottom of this file) wraps it for the real window;
    tools/headless_test.py drives it with a standalone GL context to render PNGs + benchmark with no
    window at all."""

    def __init__(self):

        self.config = Config()

        self.spectate_count = 0
        self.spectate_idx = 0
        self.prev_interp_ratio = 0
        self.car_cam_time = 0
        self.cam_manual = None   # None = AUTO (dribble-driven car cam); True/False = user forced via Space
        # Optional GUI-controlled target selection. A new closest car must hold that lead for the
        # confirmation interval before the view changes, avoiding distracting one-frame cuts.
        self._vis_auto_switch = False
        self._auto_cam_key = None   # None = follow the GUI's Vis auto switch; True/False = forced with A
        self._vis_auto_next_poll = 0.0
        self._vis_auto_control_mtime = None
        self._vis_auto_candidate = -1
        self._vis_auto_candidate_since = 0.0
        self._car_cam_offset_smooth = None   # exponential-smoothed car-cam offset/dir (None = snap next frame)
        self._car_cam_dir_smooth = None
        self.last_render_time = time.time()
        self.fps_counter = 0
        self.last_fps = 0
        self.prev_state = None # type: GameState

        # Rolling gameplay clip recorder (last 12s of rendered frames -> mp4 on demand).
        self.clip_recorder = ClipRecorder()

        # Cached static boost-pad draw data (positions never move) + hoisted constants.
        self._zero_color = Vector4((0, 0, 0, 0)).astype('f4')
        self._pad_static = None   # list of (is_big, model_matrix_f4)
        self._pad_sig = None
        self._pad_vaos = ["BoostPad_Small_0.obj", "BoostPad_Small_1.obj",
                          "BoostPad_Big_0.obj", "BoostPad_Big_1.obj"]  # indexed 2*is_big + active

        # RSV_PERF timing accumulators
        self._perf_render = 0.0
        self._perf_capture = 0.0
        self._perf_deepcopy = 0.0
        self._perf_pads = 0.0
        self._perf_cars = 0.0
        self._perf_arena = 0.0
        self._perf_ball = 0.0
        self._perf_tail = 0.0
        self._perf_lock = 0.0
        self._perf_lines = 0.0
        self._perf_hud = 0.0
        self._perf_pai = 0.0
        self._perf_ui = 0.0
        self._perf_frames = 0
        self._perf_t0 = time.time()

        ########################################################################

        # MSAA now lives ONLY in the offscreen scene framebuffer (see _ensure_render_target); the
        # window's own framebuffer is single-sample, so the final blit, HUD and clip readback are cheap.
        self.samples = max(0, MSAA)
        self.window_mode = False       # True in the real window: the Graphics settings apply (not in clips/tests)

        self.render_target = None       # framebuffer draws currently go to (scene FBO, then screen for HUD)
        self.screen_fb = None           # the window (or headless) framebuffer handed to paint()
        self.cap_fbo = None
        self.cap_resolve_fbo = None
        self._cap_size = None
        self._quad_prog = None
        self._quad_vao = None

        # Game audio + effects (created in init_gl once a context exists; audio needs no GL).
        self.audio = rl_audio.Audio()
        self.fx = None
        self._boost_last = {}          # car index -> last time it was boosting (sound hold)
        self._spectated_now = -1
        self.gpu_name = "?"
        self._last_boosting = {}
        self._prev_speed_ss = {}

    def load_texture_2d(self, path: str) -> moderngl.Texture:
        return resources.textures.load(TextureDescription(path=path))

    def _build_digit_atlas(self):
        """0-9 rendered once with a heavy italic font (RL's boost number style) into a texture."""
        from PIL import Image, ImageDraw, ImageFont
        font = None
        for fn in ("seguibli.ttf", "seguibl.ttf", "arialbi.ttf", "segoeuib.ttf"):
            try:
                font = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", fn), 112)
                break
            except OSError:
                pass
        if font is None:
            font = ImageFont.load_default()
        cell_w, cell_h = 96, 140
        img = Image.new("L", (cell_w * 10, cell_h), 0)
        d = ImageDraw.Draw(img)
        metrics = []
        for k in range(10):
            ch = str(k)
            bb = d.textbbox((0, 0), ch, font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            x = k * cell_w + (cell_w - w) // 2 - bb[0]
            y = (cell_h - h) // 2 - bb[1]
            d.text((x, y), ch, fill=255, font=font)
            metrics.append(w / float(cell_w))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        tex = self.ctx.texture(img.size, 1, img.tobytes())
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.build_mipmaps()
        return tex, (cell_w, cell_h, metrics)

    def _label_texture(self, text):
        """A single-channel text texture (Bahnschrift SemiBold), cached: (texture, aspect, cap-height fraction)."""
        cache = self.__dict__.setdefault("_label_tex", {})
        if text in cache:
            return cache[text]
        from PIL import Image, ImageDraw, ImageFont
        fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
        font = None
        for fn in ("bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf"):
            try:
                font = ImageFont.truetype(os.path.join(fonts, fn), 96)
                try:
                    font.set_variation_by_name("SemiBold")
                except Exception:
                    pass
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        pad = 8
        bb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), text, font=font)
        cap = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), "H", font=font)
        w, h = bb[2] - bb[0] + 2 * pad, cap[3] - cap[1] + 2 * pad + 30
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((pad - bb[0], pad - cap[1]), text, fill=255, font=font)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        t = self.ctx.texture(im.size, 1, im.tobytes())
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        t.build_mipmaps()
        cache[text] = (t, w / float(h), (cap[3] - cap[1]) / float(h), pad / float(h))
        return cache[text]

    def _draw_label(self, text, x, y_top, cap_h, rgba, ortho):
        tex, aspect, cap_frac, pad_frac = self._label_texture(text)
        qh = cap_h / cap_frac
        qw = qh * aspect
        x0, y0 = x - qw * pad_frac / aspect, y_top - qh * pad_frac
        arr = np.asarray([(x0, y0, 0.0, 1.0), (x0 + qw, y0, 1.0, 1.0), (x0 + qw, y0 + qh, 1.0, 0.0),
                          (x0, y0, 0.0, 1.0), (x0 + qw, y0 + qh, 1.0, 0.0), (x0, y0 + qh, 0.0, 0.0)], "f4")
        self.text_vbo.write(arr.tobytes())
        self.prog_text["m_vp"].write(ortho.astype("f4"))
        self.prog_text["Tex"].value = 0
        self.prog_text["color"].value = tuple(rgba)
        tex.use(location=0)
        self.render_target.use()
        self.text_vao.render(moderngl.TRIANGLES, vertices=6)
        return qw * (1.0 - 2.0 * pad_frac / aspect)

    @staticmethod
    def _rect_tris(x0, y0, x1, y1, rgba):
        return [(x0, y0, 0.0, *rgba), (x1, y0, 0.0, *rgba), (x1, y1, 0.0, *rgba),
                (x0, y0, 0.0, *rgba), (x1, y1, 0.0, *rgba), (x0, y1, 0.0, *rgba)]

    def render_clip_indicator(self, width, height):
        """Top-left clip status (scales with the window like the boost gauge): a blinking red dot + "Clipping..."
        and a progress bar while the clip renders offline, then "Clip saved!" for 1 s (or "Clip failed")."""
        st = self.clip_recorder.clip_status() if hasattr(self.clip_recorder, "clip_status") else None
        if st is None:
            return
        kind, v = st
        s = height / 1080.0
        m = 26.0 * s
        y = m
        u = get_ui()
        try:
            if u is not None and u.isVisible():                  # stay clear of the stats / settings panel
                y += u.window().devicePixelRatioF() * (u.geometry().bottom() + 1)
        except Exception:
            pass
        cap = 22.0 * s
        pw, ph = 300.0 * s, (78.0 if kind == "rendering" else 58.0) * s
        a = 1.0 if kind == "rendering" else min(1.0, (1.0 if kind == "saved" else 2.5) - v) / 1.0
        a = max(0.0, min(1.0, a * 3.0))
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tris = []
        tris += self._rect_tris(m, y, m + pw, y + ph, (0.03, 0.035, 0.05, 0.72 * a))           # panel
        accent = {"rendering": (0.95, 0.22, 0.22), "saved": (0.30, 0.90, 0.45), "error": (1.0, 0.30, 0.25)}[kind]
        tris += self._rect_tris(m, y, m + 5.0 * s, y + ph, (*accent, 0.95 * a))                # accent edge
        cx, cy, r = m + 26.0 * s, y + 29.0 * s, 8.0 * s
        if kind == "rendering":
            blink = 0.55 + 0.45 * math.sin(time.time() * 6.0)
            dot = (1.0, 0.25, 0.25, blink * a)
            for k in range(20):                                                          # recording dot
                a0, a1 = 2 * math.pi * k / 20, 2 * math.pi * (k + 1) / 20
                tris += [(cx, cy, 0.0, *dot), (cx + r * math.cos(a0), cy + r * math.sin(a0), 0.0, *dot),
                         (cx + r * math.cos(a1), cy + r * math.sin(a1), 0.0, *dot)]
            bx0, bx1, by0, by1 = m + 18.0 * s, m + pw - 18.0 * s, y + ph - 22.0 * s, y + ph - 14.0 * s
            tris += self._rect_tris(bx0, by0, bx1, by1, (1.0, 1.0, 1.0, 0.15 * a))           # bar track
            tris += self._rect_tris(bx0, by0, bx0 + (bx1 - bx0) * max(0.02, v), by1, (*accent, 0.95 * a))
        arr = np.asarray(tris, "f4")
        if len(arr) * 7 * 4 <= self.hud_c_max_verts * 7 * 4:
            self._hud_draw_tris(arr)
        label = {"rendering": "Clipping...", "saved": "Clip saved!", "error": "Clip failed"}[kind]
        tx = m + (44.0 if kind == "rendering" else 20.0) * s
        tcol = (1.0, 1.0, 1.0, a) if kind == "rendering" else (*accent, a)
        self._draw_label(label, tx, cy - cap / 2.0, cap, tcol, ortho)
        if kind == "rendering":
            pct = "%d%%" % int(round(v * 100))
            tex, aspect, cap_frac, pad_frac = self._label_texture(pct)
            wpx = (cap * 0.8) / cap_frac * aspect * (1.0 - 2.0 * pad_frac / aspect)
            self._draw_label(pct, m + pw - 18.0 * s - wpx, cy - cap * 0.4, cap * 0.8, (1.0, 1.0, 1.0, 0.75 * a), ortho)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    GOAL_BANNER_S = 3.0          # on screen after a goal (longer while a goal celebration hides the ball)

    def _banner_textures(self, team):
        """'BLUE SCORED!' / 'ORANGE SCORED!' rendered once: a crisp text mask + a blurred glow mask."""
        tex = self._banner_tex.get(team)
        if tex is not None:
            return tex
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        text = ("BLUE" if team == 0 else "ORANGE") + " SCORED!"
        fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
        font = None
        for fn in ("bahnschrift.ttf", "segoeuil.ttf", "segoeui.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(os.path.join(fonts, fn), 150)
                try:
                    font.set_variation_by_name("SemiLight")      # thin strokes, like RL's goal text
                except Exception:
                    pass
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        pad = 60
        bb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), text, font=font)
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        core = Image.new("L", (w, h), 0)
        ImageDraw.Draw(core).text((pad - bb[0], pad - bb[1]), text, fill=255, font=font)
        glow = core.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(14))
        out = []
        for im in (core, glow):
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            t = self.ctx.texture(im.size, 1, im.tobytes())
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.build_mipmaps()
            out.append(t)
        tex = self._banner_tex[team] = (out[0], out[1], w / float(h), (bb[3] - bb[1]) / float(h))
        return tex

    def render_goal_banner(self, width, height, state):
        """RL's goal text: golden 'BLUE SCORED!' / 'ORANGE SCORED!' with a warm glow, pops in, holds, fades."""
        if self._goal_banner is None:
            return
        team, t0 = self._goal_banner
        age = time.time() - t0
        cel = getattr(state, "celebration_pos", None) is not None or getattr(state, "ball_hidden", False)
        end = self.GOAL_BANNER_S if not cel else max(self.GOAL_BANNER_S, age + 0.35)
        if age > end:
            self._goal_banner = None
            return
        a = min(1.0, age / 0.18) * min(1.0, max(0.0, (end - age) / 0.35))
        scale = 1.0 + 0.18 * (1.0 - min(1.0, age / 0.22)) ** 2       # quick settle from 118%
        core, glow, aspect, glyph_frac = self._banner_textures(team)
        text_h = height * 0.075 * scale                               # cap height ~7.5% of the screen
        qh = text_h / glyph_frac
        qw = qh * aspect
        cx, cy = width * 0.5, height * 0.30
        x0, x1, y0, y1 = cx - qw / 2, cx + qw / 2, cy - qh / 2, cy + qh / 2
        arr = np.asarray([(x0, y0, 0.0, 1.0), (x1, y0, 1.0, 1.0), (x1, y1, 1.0, 0.0),
                          (x0, y0, 0.0, 1.0), (x1, y1, 1.0, 0.0), (x0, y1, 0.0, 0.0)], "f4")
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self.text_vbo.write(arr.tobytes())
        self.prog_text["m_vp"].write(ortho.astype("f4"))
        self.prog_text["Tex"].value = 0
        self.render_target.use()
        self.ctx.disable(moderngl.DEPTH_TEST)       # a 2D overlay: the window's depth buffer is never cleared
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        for tex, rgba in ((glow, (1.0, 0.55, 0.12, 0.55 * a)), (core, (1.0, 0.80, 0.36, a))):
            tex.use(location=0)
            self.prog_text["color"].value = rgba
            self.text_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _draw_number(self, text, cx, cy, height, rgba):
        cell_w, cell_h, widths = self._digit_metrics
        scale = height / cell_h
        adv = [max(0.45, widths[int(c)]) * cell_w * scale * 0.92 for c in text]
        x = cx - sum(adv) / 2.0
        verts = []
        for c, a in zip(text, adv):
            k = int(c)
            u0, u1 = k / 10.0, (k + 1) / 10.0
            w = cell_w * scale
            x0 = x + a / 2.0 - w / 2.0
            y0, y1 = cy - height / 2.0, cy + height / 2.0
            verts += [(x0, y0, u0, 1.0), (x0 + w, y0, u1, 1.0), (x0 + w, y1, u1, 0.0),
                      (x0, y0, u0, 1.0), (x0 + w, y1, u1, 0.0), (x0, y1, u0, 0.0)]
            x += a
        arr = np.asarray(verts, "f4")
        self.text_vbo.write(arr.tobytes())
        self.prog_text["m_vp"].write(self._hud_ortho.astype("f4"))
        self.prog_text["color"].value = tuple(rgba)
        self._digit_tex.use(location=0)
        self.prog_text["Tex"].value = 0
        self.render_target.use()
        self.text_vao.render(moderngl.TRIANGLES, vertices=len(arr))

    def _load_pad_texture(self, path):
        """Boost-pad palette with the flat yellow disc recoloured to RL's warm orange-gold."""
        from PIL import Image
        im = np.asarray(Image.open(path).convert("RGBA")).astype("f4") / 255.0
        rgb = im[..., :3]
        sat = rgb.max(-1) - rgb.min(-1)
        m = (sat > 0.25)[..., None]
        gold = np.array([1.0, 0.62, 0.12], "f4") * (0.55 + 0.45 * rgb.max(-1, keepdims=True))
        im[..., :3] = np.where(m, gold, rgb)
        im = np.ascontiguousarray(im[::-1])                  # GL origin is bottom-left
        tex = self.ctx.texture((im.shape[1], im.shape[0]), 4, (im * 255).astype("u1").tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def init_gl(self, ctx):

        self.ctx = ctx
        moderngl_window.activate_context(None, self.ctx)

        # WHICH PHYSICAL GPU is actually drawing. On a hybrid laptop (iGPU + dGPU) an OpenGL app can
        # silently land on the integrated GPU, which looks exactly like "the renderer is slow" -- this
        # prints the ground truth instead of guessing.
        try:
            self.gpu_name = str(self.ctx.info.get("GL_RENDERER", "?")).split("/")[0]
            print("[gpu] GL_RENDERER = {}".format(self.ctx.info.get("GL_RENDERER")), flush=True)
            print("[gpu] GL_VENDOR   = {}".format(self.ctx.info.get("GL_VENDOR")), flush=True)
        except Exception as _e:
            print("[gpu] could not read GL info: {!r}".format(_e), flush=True)

        ##########################################

        print("Creating shader programs...")

        self.prog = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        self.prog_arena = self.ctx.program(
            vertex_shader=ARENA_VERT_SHADER,
            fragment_shader=ARENA_FRAG_SHADER,
            geometry_shader=ARENA_GEOM_SHADER
        )

        print("Creating outline renderer...")
        #self.outline_renderer = OutlineRenderer(self.ctx, (self.width(), self.height())) # TODO: Fix resizing bugs
        self.outline_renderer = None # Disabled due to weird shader compilation issues

        print("Linking shader varaibles...")
        self.pr_m_vp = self.prog['m_vp']
        self.pr_m_model = self.prog['m_model']
        self.pr_global_color = self.prog['globalColor']
        self.pr_camera_pos = self.prog['cameraPos']

        self.pra_m_vp = self.prog_arena['m_vp']
        self.pra_m_model = self.prog_arena['m_model']
        self.pra_ball_pos = self.prog_arena['ballPos']

        ##########################################

        self.ball_ribbon = RibbonEmitter()
        self.ball_trail = RibbonEmitter()      # RL ball trail (last-touch team colour, fast ball only)
        self._flip_until = {}                  # car -> time its flip streaks stop being emitted
        self._corner_ribs = {}                 # car -> 4 RibbonEmitters (upper corners of the car)
        self._ball_trail_on = False
        self._goal_banner = None               # (scoring team, time) -> "BLUE SCORED!" / "ORANGE SCORED!"
        self._banner_tex = {}
        self.car_ribbons = []

        print("Data path:", DATA_DIR_PATH)
        print("Loading models and textures...")

        self.vaos = {}
        # RL-style programs (rl_shaders.py). The arena no longer goes through the geometry-shader
        # wireframe program -- that pass is what drew the blue/red triangle-edge lines.
        self.prog_rl_arena = self.ctx.program(vertex_shader=rl_shaders.ARENA_VERT, fragment_shader=rl_shaders.ARENA_FRAG)
        self.prog_car = self.ctx.program(vertex_shader=rl_shaders.CAR_VERT, fragment_shader=rl_shaders.CAR_FRAG)
        self.prog_ball = self.ctx.program(vertex_shader=rl_shaders.BALL_VERT, fragment_shader=rl_shaders.BALL_FRAG)
        self.prog_sky = self.ctx.program(vertex_shader=rl_shaders.SKY_VERT, fragment_shader=rl_shaders.SKY_FRAG)
        self.sky_vao = self.ctx.vertex_array(self.prog_sky, [])
        # low-poly valley around the arena (landscape.py); the built mesh is cached next to the data
        self.prog_stadium = self.ctx.program(vertex_shader=rl_shaders.LANDSCAPE_VERT, fragment_shader=rl_shaders.LANDSCAPE_FRAG)
        st_mesh = landscape.load_or_build(os.path.join(DATA_DIR_PATH, "landscape_cache.npy"))
        self.stadium_vao = self.ctx.vertex_array(
            self.prog_stadium, [(self.ctx.buffer(st_mesh.tobytes()), "3f 2f", "in_position", "in_uv")])
        self.stadium_n = len(st_mesh)
        self.load_vao("ArenaMeshCustom.obj", self.prog_rl_arena)
        for prog in (self.prog_rl_arena,):
            prog["blueCol"].value = (0.10, 0.40, 1.00)
            prog["orangeCol"].value = (1.00, 0.42, 0.06)

        # Car body: an optional detailed mesh data/Octane_RL.npz (pos, normal, material id, 4 separate
        # wheels) -- not included here -- else RocketSimVis's low-poly Octane.obj converted to the same
        # format (wheels are part of its body).
        if os.path.exists(DATA_DIR_PATH + "Octane_RL.npz"):
            oct_npz = np.load(DATA_DIR_PATH + "Octane_RL.npz")
        else:
            oct_npz = _octane_from_obj(DATA_DIR_PATH + "Octane.obj", DATA_DIR_PATH + "T_Octane_B.png")
        oct_inter = np.concatenate([oct_npz["pos"], oct_npz["nrm"], oct_npz["mat"][:, None]], 1).astype("f4")
        self.car_vbo = self.ctx.buffer(oct_inter.tobytes())
        self.car_vao = self.ctx.vertex_array(self.prog_car, [(self.car_vbo, "3f 3f 1f", "in_position", "in_normal", "in_mat")])
        self.car_vert_count = len(oct_inter)
        self.wheel_vaos = []
        for k in range(4):
            if len(oct_npz["w%d_pos" % k]) == 0:
                continue                             # wheels baked into the body (Octane.obj fallback)
            wi = np.concatenate([oct_npz["w%d_pos" % k], oct_npz["w%d_nrm" % k], oct_npz["w%d_mat" % k][:, None]], 1).astype("f4")
            vbo = self.ctx.buffer(wi.tobytes())
            self.wheel_vaos.append(self.ctx.vertex_array(self.prog_car, [(vbo, "3f 3f 1f", "in_position", "in_normal", "in_mat")]))
        self.wheel_rig = carrig.WheelRig(oct_npz["wheel_centers"], oct_npz["wheel_radius"])
        self.car_streak_points = _streak_points(oct_npz["pos"])   # flip streaks come off these model points

        # Ball: smooth icosphere (radius = RocketSim soccar ball) + procedural panel shader.
        sv, sf_, panel_dirs = _icosphere(4)
        self.ball_vbo = self.ctx.buffer((sv * 91.25).astype("f4").tobytes())
        self.ball_ibo = self.ctx.buffer(sf_.astype("i4").tobytes())
        self.ball_vao = self.ctx.vertex_array(self.prog_ball, [(self.ball_vbo, "3f", "in_position")], self.ball_ibo)

        self.fx = rl_fx.FX(self.ctx)

        # Boost pads: own program (glowing gold top, respawn ghost, respawn flash)
        self.prog_pad = self.ctx.program(vertex_shader=rl_shaders.PAD_VERT, fragment_shader=rl_shaders.PAD_FRAG)
        self.pad_vaos_rl = {}
        for name in self._pad_vaos:
            loader = wvf.Loader(wvf.SceneDescription(path=DATA_DIR_PATH + "/" + name))
            self.pad_vaos_rl[name] = loader.load().root_nodes[0].mesh.vao.instance(self.prog_pad)
        self._pad_prev = None
        self._pad_pick_t = []
        self._pad_spawn_t = []

        # Boost meter: analytic gauge + font atlas digits
        self.prog_gauge = self.ctx.program(vertex_shader=rl_shaders.GAUGE_VERT, fragment_shader=rl_shaders.GAUGE_FRAG)
        self.gauge_vbo = self.ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self.gauge_vao = self.ctx.vertex_array(self.prog_gauge, [(self.gauge_vbo, "2f 2f", "in_pos", "in_q")])
        self.prog_text = self.ctx.program(vertex_shader=rl_shaders.TEXT_VERT, fragment_shader=rl_shaders.TEXT_FRAG)
        self.text_vbo = self.ctx.buffer(reserve=6 * 4 * 4 * 16, dynamic=True)
        self.text_vao = self.ctx.vertex_array(self.prog_text, [(self.text_vbo, "2f 2f", "in_pos", "in_uv")])
        self._digit_tex, self._digit_metrics = self._build_digit_atlas()
        self._gpu_query = self.ctx.query(time=True) if PERF else None

        # (Octane.obj / Ball.obj are no longer drawn -- replaced by Octane_RL.npz + the procedural ball --
        # so they aren't parsed at startup any more; pywavefront is slow.)

        self.load_vao("BoostPad_Small_0.obj")
        self.load_vao("BoostPad_Small_1.obj")
        self.load_vao("BoostPad_Big_0.obj")
        self.load_vao("BoostPad_Big_1.obj")

        self.ts_octane = [
            self.load_texture_2d(DATA_DIR_PATH + "T_Octane_B.png"),
            self.load_texture_2d(DATA_DIR_PATH + "T_Octane_O.png")
        ]
        self.t_ball = self.load_texture_2d(DATA_DIR_PATH + "T_Ball.png")
        self.t_boostpad = self._load_pad_texture(DATA_DIR_PATH + "T_BoostPad.png")
        self.t_boost_glow = self.load_texture_2d(DATA_DIR_PATH + "T_Boost_Glow.png")
        self.t_black = self.load_texture_2d(DATA_DIR_PATH + "T_Black.png")
        self.t_none = self.load_texture_2d(DATA_DIR_PATH + "T_None.png")

        ############################################

        # Make ribbon mesh
        self.ribbon_max_verts = 1000
        self.ribbon_verts = np.random.randn(self.ribbon_max_verts * 3) * 100
        self.ribbon_vbo = self.ctx.buffer(self.ribbon_verts.astype('f4'))
        self.ribbon_vao = self.ctx.simple_vertex_array(self.prog, self.ribbon_vbo, "in_position")
        self.vaos['ribbon'] = self.ribbon_vao

        # Make debug lines mesh
        self.lines_max_verts = RenderState.MAX_LINES * 2
        self.lines_verts = np.random.randn(self.lines_max_verts * 3) * 100
        self.lines_vbo = self.ctx.buffer(self.lines_verts.astype('f4'))
        self.lines_vao = self.ctx.simple_vertex_array(self.prog, self.lines_vbo, "in_position")
        self.vaos['render_lines'] = self.lines_vao

        # Make 2D HUD mesh (screen-space overlay, e.g. the boost gauge)
        self.hud_max_verts = 65536   # was 4096: the boost gauge now batches ~15.6k verts per frame
        self.hud_vbo = self.ctx.buffer(reserve=self.hud_max_verts * 3 * 4)  # 3 floats/vert
        self.hud_vao = self.ctx.simple_vertex_array(self.prog, self.hud_vbo, "in_position")
        self.vaos['hud'] = self.hud_vao

        ############################################

        # Fullscreen-triangle blit shader: draws the capped-resolution resolve texture (see
        # _ensure_render_target) scaled up to fill the real window. No vertex buffer needed -- the
        # 3 vertices come from gl_VertexID, the standard attributeless "big triangle" trick.
        self._quad_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                out vec2 v_uv;
                void main() {
                    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                    v_uv = pos;
                    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Tex;
                in vec2 v_uv;
                out vec4 f_color;
                // NO y-flip: cap_resolve_fbo was rendered with the SAME bottom-left-origin convention
                // ctx.screen uses (it's another GL framebuffer, not a loaded top-down image file), so
                // sampling it straight reproduces the frame right-side-up. (v_uv.y flipped here once --
                // that's what turned the picture upside down.)
                void main() { f_color = texture(Tex, v_uv); }
            ''',
        )
        self._quad_vao = self.ctx.vertex_array(self._quad_prog, [])

        # Per-vertex-COLOUR HUD program. The shared `prog` carries colour in a globalColor UNIFORM, so
        # every differently-coloured band needs its own draw (63 of them for the boost gauge, ~40 us
        # each in Python/moderngl overhead). Carrying colour per VERTEX instead lets the whole gauge --
        # every band, every alpha -- go out as ONE stitched triangle strip.
        self._hud_c_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 m_vp;
                in vec3 in_position;
                in vec4 in_color;
                out vec4 v_color;
                void main() { v_color = in_color; gl_Position = m_vp * vec4(in_position, 1.0); }
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() { f_color = v_color; }
            ''',
        )
        self.hud_c_max_verts = 65536
        self.hud_c_vbo = self.ctx.buffer(reserve=self.hud_c_max_verts * 7 * 4)   # 3 pos + 4 colour
        self.hud_c_vao = self.ctx.vertex_array(
            self._hud_c_prog, [(self.hud_c_vbo, '3f 4f', 'in_position', 'in_color')])
        self._hud_ortho = None      # set by whichever 2D pass is active (see render_boost_hud)

        ############################################

        # Auto-enable multisampling if we have multiple samples
        self.ctx.multisample = self.samples > 1

        ############################################

        print("Done.")

    def load_vao(self, model_name, program = None):
        loader = wvf.Loader(wvf.SceneDescription(path = DATA_DIR_PATH + "/" + model_name))
        model = loader.load()
        self.vaos[model_name] = model.root_nodes[0].mesh.vao.instance(self.prog if (program is None) else program)
        if not (self.outline_renderer is None):
            self.outline_renderer.load_vao(model_name, model)

    def _ensure_pad_cache(self, state):
        """Build the per-pad (is_big, static model matrix) cache once. Pads never move, so this only
        rebuilds if the pad layout actually changes (e.g. a custom map sends new locations)."""
        pads = state.boost_pad_locations
        sig = (len(pads), float(pads[0].x) if pads else 0.0, float(pads[-1].y) if pads else 0.0)
        if self._pad_static is not None and self._pad_sig == sig:
            return
        self._pad_sig = sig
        self._pad_static = []
        s = 2.5
        for p in pads:
            is_big = bool(p.z == 73)
            pos = Vector3((p.x, p.y, 0.0))
            forward = Vector3((1, 0, 0)) * s
            right = Vector3(pyrr.vector3.cross(Vector3((1, 0, 0)), Vector3((0, 0, 1)))) * s
            up = Vector3((0, 0, 1)) * s
            mat = Matrix44([
                forward[0], forward[1], forward[2], 0,
                -right[0], -right[1], -right[2], 0,
                up[0], up[1], up[2], 0,
                pos[0], pos[1], pos[2], 1,
            ]).astype('f4')
            self._pad_static.append((is_big, mat))

    def render_model(self,
                     pos, forward, up,
                     model_name, texture, scale = 1.0, global_color = None,
                     mode = moderngl.TRIANGLES, outline_color: Vector3 = None, vert_amount = None,
                     first_vert = 0):

        if pos is None:
            model_mat = Matrix44.identity()
        else:
            pos = Vector3(pos)
            forward = Vector3(forward)
            up = Vector3(up)
            right = Vector3(pyrr.vector3.cross(forward, up))

            forward *= scale
            right *= scale
            up *= scale

            model_mat = Matrix44([
                forward[0], forward[1], forward[2], 0,
                -right[0], -right[1], -right[2], 0,
                up[0], up[1], up[2], 0,
                pos[0], pos[1], pos[2], 1
            ])

        self.pr_m_model.write(model_mat.astype('f4'))
        self.pra_m_model.write(model_mat.astype('f4'))

        if global_color is None:
            global_color = Vector4((0, 0, 0, 0))
        self.pr_global_color.write(global_color.astype('f4'))

        if texture is not None:
            texture.use()
        else:
            self.t_none.use()

        self.render_target.use()
        self.vaos[model_name].render(mode, vertices=(-1 if (vert_amount is None) else vert_amount),
                                     first=first_vert)

        if outline_color is not None:
            self.outline_renderer.use_framebuf()
            self.outline_renderer.pr_m_model.write(model_mat.astype('f4'))
            self.outline_renderer.pr_color.write(Vector4((outline_color.x, outline_color.y, outline_color.z, 1)).astype('f4'))
            self.outline_renderer.vaos[model_name].render(mode)

    def render_ribbon(self, ribbon: RibbonEmitter, camera_pos, lifetime, width, start_taper_time, color):
        if len(ribbon.points) == 0:
            return

        first_point = ribbon.points[0]
        cam_to_ribbon_dir = safe_normalize(-(first_point.pos - camera_pos))
        ribbon_away_dir = safe_normalize(first_point.vel)
        ribbon_sideways_dir = ribbon_away_dir.cross(cam_to_ribbon_dir)

        # VECTORISED build (identical geometry). The old version ran a Python loop doing pyrr Vector3
        # arithmetic per point and appended two Vector3s each -- every one of those allocates, and it
        # runs for EVERY ribbon (one per car plus the ball) EVERY frame, which is the bulk of the
        # "cars" stage. Positions/times are pulled once into arrays and the offsets applied in numpy.
        pts = [pt for pt in ribbon.points if pt.connected]
        if not pts:
            return
        pos = np.asarray([tuple(pt.pos) for pt in pts], dtype='f4')          # (P,3)
        ta = np.asarray([pt.time_active for pt in pts], dtype='f4')          # (P,)
        with np.errstate(divide='ignore', invalid='ignore'):
            ws = np.where(ta < start_taper_time,
                          ta / max(start_taper_time, 1e-9),
                          1.0 - (ta / max(lifetime, 1e-9)))
        side = np.asarray(tuple(ribbon_sideways_dir), dtype='f4')
        off = side[None, :] * (float(width) * ws)[:, None]                   # (P,3)
        verts = np.empty((2 * len(pos), 3), dtype='f4')
        verts[0::2] = pos - off                                              # i=0 -> offset_scale -1
        verts[1::2] = pos + off                                              # i=1 -> offset_scale +1
        if len(verts) > self.ribbon_max_verts:
            verts = verts[:self.ribbon_max_verts]
        vertices = verts

        n = len(vertices)
        if n == 0:
            return

        # Upload + draw only the actual vertices. Previously every ribbon was padded out to
        # ribbon_max_verts (1000) with a Python loop and a 1000-vertex triangle-strip draw, almost
        # all of it degenerate — a big chunk of the per-frame "cars" cost. A partial VBO write +
        # vert_amount draw renders the identical visible trail for a fraction of the work.
        self.ribbon_vbo.write(vertices, 0)

        self.ctx.disable(moderngl.CULL_FACE)
        self.render_model(
            None, None, None,
            "ribbon", self.t_none, scale=20,
            global_color=color,
            mode=moderngl.TRIANGLE_STRIP,
            vert_amount=n
        )
        self.ctx.enable(moderngl.CULL_FACE)

    # ---- 2D HUD (screen-space overlay) ------------------------------------

    # Car speed (uu/s) at/above which the car is supersonic
    SUPERSONIC_SPEED = 2200.0
    BOOST_SOUND_HOLD = 0.35     # s: feathered boost presses closer than this = one boost (sound only)

    # 7-segment layout: which of (a, b, c, d, e, f, g) are lit per digit
    #    _a_
    #  f|   |b
    #    _g_
    #  e|   |c
    #    _d_
    _SEVEN_SEG = {
        '0': (1, 1, 1, 1, 1, 1, 0),
        '1': (0, 1, 1, 0, 0, 0, 0),
        '2': (1, 1, 0, 1, 1, 0, 1),
        '3': (1, 1, 1, 1, 0, 0, 1),
        '4': (0, 1, 1, 0, 0, 1, 1),
        '5': (1, 0, 1, 1, 0, 1, 1),
        '6': (1, 0, 1, 1, 1, 1, 1),
        '7': (1, 1, 1, 0, 0, 0, 0),
        '8': (1, 1, 1, 1, 1, 1, 1),
        '9': (1, 1, 1, 1, 0, 1, 1),
    }

    def _hud_batch_draw(self, items):
        """items = [(verts, color, mode), ...] -> ONE hud_vbo.write + one draw per item at a vertex
        offset, preserving draw order (alpha blending depends on it).

        Why this exists: _hud_draw writes the SHARED hud_vbo and immediately draws it, so calling it
        N times per frame forces N GPU pipeline stalls (each write must wait for the previous draw to
        release the buffer). The boost gauge's radial fade issues 120 such calls per frame, which
        measured at 10-16 ms -- the entire frame budget -- while the 34 boost-pad draws cost 0.11 ms
        total. Draw calls are cheap; write-then-draw on a live buffer is not. Batching removes 119 of
        the 120 writes and leaves the visuals byte-identical (same geometry, colours and order)."""
        items = [(v, c, m) for (v, c, m) in items if len(v) > 0]
        if not items:
            return
        blocks, offsets, off = [], [], 0
        for verts, _c, _m in items:
            arr = verts if isinstance(verts, np.ndarray) else np.asarray(verts, dtype='f4')
            if arr.dtype != np.float32:
                arr = arr.astype('f4')
            offsets.append((off, len(arr)))
            blocks.append(arr)
            off += len(arr)
        if off > self.hud_max_verts:                      # never overrun the buffer; drop the tail
            keep = 0
            while keep < len(offsets) and offsets[keep][0] + offsets[keep][1] <= self.hud_max_verts:
                keep += 1
            if keep == 0:
                return
            items, offsets, blocks = items[:keep], offsets[:keep], blocks[:keep]
    def _hud_stitch(self, items):
        """[(verts, rgba, TRIANGLE_STRIP), ...] -> one interleaved (N,7) pos+colour array, the strips
        joined by DEGENERATE (zero-area) triangles. Zero-area triangles emit no fragments and strip
        order is preserved, so the blended result is identical to drawing each band separately.
        Cull/depth are already off for 2D, so the winding flips stitching causes are harmless."""
        items = [(v, c, m) for (v, c, m) in items if len(v) > 0]
        if not items:
            return None
        total = sum(len(v) for v, _c, _m in items) + 2 * (len(items) - 1)
        if total > self.hud_c_max_verts:
            return None
        inter = np.empty((total, 7), dtype='f4')
        w = 0
        for i, (verts, color, _m) in enumerate(items):
            arr = verts if isinstance(verts, np.ndarray) else np.asarray(verts, dtype='f4')
            if arr.dtype != np.float32:
                arr = arr.astype('f4')
            rgba = np.asarray(color, dtype='f4')
            if i > 0:                                   # degenerate bridge: prev last, then new first
                inter[w] = inter[w - 1]
                inter[w + 1, 0:3] = arr[0]
                inter[w + 1, 3:7] = rgba
                w += 2
            n = len(arr)
            inter[w:w + n, 0:3] = arr
            inter[w:w + n, 3:7] = rgba
            w += n
        return inter[:w]

    def _hud_draw_strip(self, inter):
        """Draw a pre-stitched interleaved strip from _hud_stitch (one write + one draw)."""
        if inter is None or len(inter) == 0 or self._hud_ortho is None:
            return
        self.hud_c_vbo.write(inter.tobytes(), 0)
        self._hud_c_prog['m_vp'].write(self._hud_ortho.astype('f4'))
        self.render_target.use()
        self.hud_c_vao.render(moderngl.TRIANGLE_STRIP, vertices=len(inter))

    def _hud_batch_draw_unused(self, items):
        # ONE-DRAW path: every item is a TRIANGLE_STRIP and we know the ortho matrix -> concatenate
        # them into a single strip joined by DEGENERATE (zero-area) triangles, carrying colour per
        # vertex. Zero-area triangles emit no fragments, and strip order is preserved, so the blended
        # result is identical to drawing each band separately. Cull/depth are already disabled by the
        # 2D caller, so the winding flips degenerate stitching causes are harmless.
        if self._hud_ortho is not None and all(m == moderngl.TRIANGLE_STRIP for _v, _c, m in items)                 and off + 2 * len(blocks) <= self.hud_c_max_verts:
            pos_parts, col_parts = [], []
            for i, (arr, (verts, color, _m)) in enumerate(zip(blocks, items)):
                col = np.empty((len(arr), 4), dtype='f4')
                col[:] = np.asarray(color, dtype='f4')
                if i > 0:
                    pos_parts.append(pos_parts[-1][-1:])        # repeat previous last vertex
                    col_parts.append(col_parts[-1][-1:])
                    pos_parts.append(arr[:1])                   # and the new first vertex
                    col_parts.append(col[:1])
                pos_parts.append(arr)
                col_parts.append(col)
            pos = np.concatenate(pos_parts, axis=0)
            col = np.concatenate(col_parts, axis=0)
            inter = np.empty((len(pos), 7), dtype='f4')
            inter[:, 0:3] = pos
            inter[:, 3:7] = col
            self.hud_c_vbo.write(inter.tobytes(), 0)
            self._hud_c_prog['m_vp'].write(self._hud_ortho.astype('f4'))
            self.render_target.use()
            self.hud_c_vao.render(moderngl.TRIANGLE_STRIP, vertices=len(inter))
            return

        self.hud_vbo.write(np.concatenate(blocks, axis=0).tobytes(), 0)

        # LEAN per-band draw. Going through render_model cost ~7 ms for 120 bands because every call
        # rebuilt Matrix44.identity().astype('f4'), wrote BOTH model-matrix uniforms, and rebound the
        # texture + framebuffer -- all identical for every band in the batch. Hoist that setup out and
        # per band only the colour uniform changes. Same state, same draws, same result.
        ident = Matrix44.identity().astype('f4')
        self.pr_m_model.write(ident)
        self.pra_m_model.write(ident)
        self.t_none.use()
        self.render_target.use()
        vao = self.vaos["hud"]
        for (verts, color, mode), (start, count) in zip(items, offsets):
            self.pr_global_color.write(Vector4(color).astype('f4'))
            vao.render(mode, vertices=count, first=start)

    _IDENT_F4 = np.eye(4, dtype="f4").tobytes()

    def _hud_draw(self, verts, color, mode):
        """One HUD draw with a uniform colour. Lean path (no render_model / pyrr per call), and each
        draw's vertices go to a fresh region of hud_vbo (offset reset once per frame in paint()), so
        a write never has to wait for the GPU to finish reading the previous draw's vertices."""
        n = len(verts)
        if n == 0:
            return
        arr = verts if isinstance(verts, np.ndarray) else np.asarray(verts, dtype='f4')
        if arr.dtype != np.float32:
            arr = arr.astype('f4')
        off = getattr(self, "_hud_off", 0)
        if off + n > self.hud_max_verts:
            off = 0
        self.hud_vbo.write(arr.tobytes(), off * 12)
        self._hud_off = off + n
        self.pr_m_model.write(self._IDENT_F4)
        self.pr_global_color.write(np.asarray(color, dtype='f4').tobytes())
        self.t_none.use()
        self.render_target.use()
        self.vaos["hud"].render(mode, vertices=n, first=off)

    def _hud_draw_tris(self, inter):
        """Per-vertex-colour triangles (interleaved (N,7) pos+rgba) in one draw."""
        if inter is None or len(inter) == 0 or self._hud_ortho is None:
            return
        self.hud_c_vbo.write(inter.tobytes(), 0)
        self._hud_c_prog['m_vp'].write(self._hud_ortho.astype('f4'))
        self.render_target.use()
        self.hud_c_vao.render(moderngl.TRIANGLES, vertices=len(inter))

    @staticmethod
    def _disc_verts(cx, cy, r, segments=48):
        # Triangle fan: centre + perimeter (screen coords, y points down)
        verts = [(cx, cy, 0.0)]
        for k in range(segments + 1):
            a = 2.0 * math.pi * k / segments
            verts.append((cx + r * math.sin(a), cy - r * math.cos(a), 0.0))
        return verts

    @staticmethod
    def _ring_verts(cx, cy, r_in, r_out, frac, segments=48):
        # Triangle strip annulus sweeping `frac` of a full turn, starting at top
        frac = max(0.0, min(1.0, frac))
        if frac <= 0.0:
            return []
        n = max(1, int(round(segments * frac)))
        verts = []
        for k in range(n + 1):
            a = 2.0 * math.pi * frac * (k / n)
            sa, ca = math.sin(a), math.cos(a)
            verts.append((cx + r_out * sa, cy - r_out * ca, 0.0))
            verts.append((cx + r_in * sa, cy - r_in * ca, 0.0))
        return verts

    @staticmethod
    def _arc_ring_verts(cx, cy, r_in, r_out, start_deg, span_deg, frac, segments=64):
        """Triangle strip annulus over an ARBITRARY arc (not a full circle) -- the actual Rocket League
        boost-meter shape: a ring with a GAP at the bottom, not a closed loop. Angle convention: 0deg =
        straight up (12 o'clock), increasing CLOCKWISE (90=3 o'clock/right, 180=6 o'clock/bottom,
        270=9 o'clock/left) -- matches how a viewer reads clock positions on screen (y-down space).
        Sweeps `frac` (0..1) of `span_deg` starting at `start_deg`, so frac=1 draws the FULL arc (the
        boost track) and frac=<boost/100> draws just the filled portion."""
        frac = max(0.0, min(1.0, frac))
        if frac <= 0.0 or span_deg <= 0.0:
            return []
        sweep = span_deg * frac
        n = max(1, int(round(segments * frac)))
        verts = []
        for k in range(n + 1):
            a = math.radians(start_deg + sweep * (k / n))
            sa, ca = math.sin(a), math.cos(a)
            verts.append((cx + r_out * sa, cy - r_out * ca, 0.0))
            verts.append((cx + r_in * sa, cy - r_in * ca, 0.0))
        return verts

    @staticmethod
    def _rect_verts(x0, y0, x1, y1):
        # Two triangles for an axis-aligned rectangle
        return [
            (x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0),
            (x0, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),
        ]

    # Minimal 5x7 dot-matrix font used ONLY by the GAIL p_ai HUD (render_pai_hud), which needs
    # letters/punctuation that the 7-segment digit font above can't draw. Covers exactly the
    # characters that string can contain: "P_AI <val|N/A>  <label> (HARD)". Add glyphs here if the
    # sender ever emits a wider vocabulary. Lookup is upper-cased, so lowercase input still resolves.
    _FONT_5X7 = {
        '0': ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        '1': ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        '2': ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        '3': ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
        '4': ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        '5': ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
        '6': ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
        '7': ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        '8': ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
        '9': ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
        'A': ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        'D': ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
        'H': ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        'R': ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        'N': ("10001", "11001", "10101", "10101", "10011", "10001", "10001"),
        'P': ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        'I': ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
        'E': ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        'L': ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        'O': ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        'T': ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
        '+': ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
        '_': ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
        '.': ("00000", "00000", "00000", "00000", "00000", "00000", "00100"),
        '-': ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
        '/': ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
        '(': ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
        ')': ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
        ' ': ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    }

    @classmethod
    def _char_pixel_verts(cls, ch, x, y, w, h):
        """Filled-pixel 5x7 dot-matrix glyph, top-left at (x, y), cell size (w, h). Unknown
        characters (not in _FONT_5X7) draw nothing rather than raising, so a stray/unsupported
        character in a sender-supplied string just leaves a blank cell instead of crashing render()."""
        pattern = cls._FONT_5X7.get(ch.upper())
        if not pattern:
            return []
        cols, rows = 5, 7
        cell_w = w / cols
        cell_h = h / rows
        pad = min(cell_w, cell_h) * 0.14  # small gap between pixels for a legible "LED" look
        verts = []
        for r, row in enumerate(pattern):
            for c, bit in enumerate(row):
                if bit == '1':   # patterns are '1'/'0' bit-strings (was '#' -> never matched -> blank text)
                    x0 = x + c * cell_w + pad
                    x1 = x + (c + 1) * cell_w - pad
                    y0 = y + r * cell_h + pad
                    y1 = y + (r + 1) * cell_h - pad
                    verts += cls._rect_verts(x0, y0, x1, y1)
        return verts

    @staticmethod
    def digit_advance(ch, w):
        # '1' is just a single bar, so it gets a narrower cell -> proper centring
        return w * 0.45 if ch == '1' else w

    @classmethod
    def _digit_verts(cls, ch, x, y, w, h, t):
        """Filled-quad 7-segment digit, top-left at (x, y), size (w, h), thickness t."""
        # '1' has no left column, so draw its bar centred in the cell instead of
        # hugging the right edge (otherwise numbers like "100" look off-centre).
        if ch == '1':
            bx = x + (w - t) / 2.0
            return cls._rect_verts(bx, y, bx + t, y + h)

        segs = cls._SEVEN_SEG.get(ch)
        if segs is None:
            return []
        a, b, c, d, e, f, g = segs
        hh = h / 2.0
        rects = []  # (x0, y0, x1, y1)

        if a: rects.append((x, y, x + w, y + t))                       # top
        if g: rects.append((x, y + hh - t / 2.0, x + w, y + hh + t / 2.0))  # middle
        if d: rects.append((x, y + h - t, x + w, y + h))               # bottom
        if f: rects.append((x, y, x + t, y + hh))                      # top-left
        if b: rects.append((x + w - t, y, x + w, y + hh))              # top-right
        if e: rects.append((x, y + hh, x + t, y + h))                  # bottom-left
        if c: rects.append((x + w - t, y + hh, x + w, y + h))          # bottom-right

        verts = []
        for (x0, y0, x1, y1) in rects:
            verts += cls._rect_verts(x0, y0, x1, y1)
        return verts

    def render_boost_hud(self, width, height, boost_amount, team_num=0):
        """RL-style boost meter, bottom-right: dark disc, segmented arc (gap at the bottom-right) filled
        with a team gradient + glow, and the boost number in a heavy italic font. The ring is ONE
        analytic-shader quad (smooth at any size) and the number is a font-atlas text draw."""
        boost = int(round(max(0.0, min(100.0, boost_amount))))
        R = max(52.0, min(width, height) * 0.095)
        margin = R * 0.45 + min(width, height) * 0.02
        cx, cy = width - R - margin, height - R - margin
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        if int(team_num) == 1:
            c1, c2, track = (1.0, 0.72, 0.18), (1.0, 0.36, 0.04), (0.22, 0.12, 0.04)
        else:
            c1, c2, track = (0.45, 0.85, 1.0), (0.12, 0.42, 1.0), (0.05, 0.10, 0.22)
        e = R * 1.12
        q = 1.12
        quad = np.asarray([(cx - e, cy - e, -q, -q), (cx + e, cy - e, q, -q), (cx + e, cy + e, q, q),
                           (cx - e, cy - e, -q, -q), (cx + e, cy + e, q, q), (cx - e, cy + e, -q, q)], "f4")
        self.gauge_vbo.write(quad.tobytes())
        g = self.prog_gauge
        g["m_vp"].write(ortho.astype("f4"))
        g["fill"].value = boost / 100.0
        g["fillCol"].value = c1
        g["fillCol2"].value = c2
        g["trackCol"].value = track
        g["startDeg"].value = 197.0
        g["spanDeg"].value = 250.0
        self.render_target.use()
        self.gauge_vao.render(moderngl.TRIANGLES, vertices=6)
        self._draw_number(str(boost), cx, cy + R * 0.02, R * 0.78, (1.0, 1.0, 1.0, 1.0))
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def render_scoreboard_hud(self, width, height, scoreboard):
        """Top-center match scoreboard: "<blue>   <M:SS>   <orange>" -- clock BETWEEN the scores,
        replaced by "OT" once regulation time is up. Handles multi-digit scores (widths are measured,
        not assumed). The vis runs a real match clock via EnvSetConfig::fullGame, re-rolled per
        episode (randomizeScoreboardOnReset) since only the displayed arena is ever stepped.

        FLAG-GATED like render_pai_hud: `scoreboard` comes from socket_listener's parse of the
        optional packet field "scoreboard" ({"score_blue", "score_orange", "time_left",
        "is_overtime"}) and defaults to None when the sender doesn't include it -- so senders that
        never set it (state-set editor, clip players) draw nothing.

        Reuses render_boost_hud's 7-segment digit font for the numerals and the 5x7 dot-matrix font
        only for the "OT" letters, which the 7-segment font can't draw.
        """
        if not scoreboard:
            return

        score_blue = int(scoreboard.get("score_blue", 0) or 0)
        score_orange = int(scoreboard.get("score_orange", 0) or 0)
        time_left = float(scoreboard.get("time_left", 0.0) or 0.0)
        is_overtime = bool(scoreboard.get("is_overtime", False))

        mins, secs = divmod(max(0, int(round(time_left))), 60)
        clock_str = "{:d}:{:02d}".format(mins, secs)

        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self.pr_m_vp.write(ortho.astype('f4'))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # Layout: [blue score]   [clock]   [orange score] -- clock BETWEEN the two scores, one
        # centered row, no panels. Every size derives from `ch` so the whole thing scales with the
        # window and stays correct through a live resize.
        ch = float(np.clip(height * 0.052, 34.0, 104.0))    # clock digit height (the dominant element)
        cw = ch * 0.58
        ct = ch * 0.125                                     # 7-segment stroke thickness (lighter)
        cgap = cw * 0.36                                    # more air between digits
        dh = ch * 0.86                                      # score digits, slightly smaller than the clock
        dw = dh * 0.58
        dt = dh * 0.135
        dgap = dw * 0.36
        side_gap = ch * 0.85                                # space between a score and the clock

        COLON_ADV = 0.42                                    # colon cell is narrower than a digit

        def measure(text, w, gap):
            """(total_width, advances) -- MUST mirror the advances used when drawing below, or the
            centering and the backing panel drift from what actually appears."""
            adv = [w * COLON_ADV if c == ':' else self.digit_advance(c, w) for c in text]
            return sum(adv) + gap * max(0, len(text) - 1), adv

        blue_str, orange_str = str(score_blue), str(score_orange)   # 2+ digits handled by measure()
        blue_w, blue_adv = measure(blue_str, dw, dgap)
        orange_w, orange_adv = measure(orange_str, dw, dgap)
        if is_overtime:
            mid_adv = [cw] * 2
            mid_w = 2 * cw + cgap
        else:
            mid_w, mid_adv = measure(clock_str, cw, cgap)

        total_w = blue_w + side_gap + mid_w + side_gap + orange_w
        top_y = max(14.0, height * 0.022)
        x0 = width / 2.0 - total_w / 2.0

        # Backing panel sized to the CLOCK row (the tallest element), with the scores centred in it.
        pad_x, pad_y = ch * 0.55, ch * 0.26
        self._hud_draw(
            self._rect_verts(x0 - pad_x, top_y - pad_y, x0 + total_w + pad_x, top_y + ch + pad_y),
            (0.0, 0.0, 0.0, 0.62), moderngl.TRIANGLES
        )

        blue_rgb, orange_rgb = (0.30, 0.65, 1.0), (1.0, 0.60, 0.10)
        score_y = top_y + (ch - dh) / 2.0                   # vertically centre scores against the clock

        x = x0
        verts = []
        for c, adv in zip(blue_str, blue_adv):
            verts += self._digit_verts(c, x, score_y, adv, dh, dt)
            x += adv + dgap
        self._hud_draw(verts, (blue_rgb[0], blue_rgb[1], blue_rgb[2], 1.0), moderngl.TRIANGLES)

        x = x0 + blue_w + side_gap
        clock_color = (1.0, 0.85, 0.2, 1.0) if is_overtime else (0.96, 0.96, 0.96, 1.0)
        verts = []
        if is_overtime:
            for c in "OT":
                verts += self._char_pixel_verts(c, x, top_y, cw, ch)
                x += cw + cgap
        else:
            for c, adv in zip(clock_str, mid_adv):
                if c == ':':
                    # Two square pips, inset from the digit top/bottom so the colon reads as part of
                    # the same 7-segment face instead of a stray pair of blocks.
                    s = ct * 1.45
                    px = x + (adv - s) / 2.0
                    verts += self._rect_verts(px, top_y + ch * 0.26, px + s, top_y + ch * 0.26 + s)
                    verts += self._rect_verts(px, top_y + ch * 0.60, px + s, top_y + ch * 0.60 + s)
                else:
                    verts += self._digit_verts(c, x, top_y, adv, ch, ct)
                x += adv + cgap
        self._hud_draw(verts, clock_color, moderngl.TRIANGLES)

        x = x0 + blue_w + side_gap + mid_w + side_gap
        verts = []
        for c, adv in zip(orange_str, orange_adv):
            verts += self._digit_verts(c, x, score_y, adv, dh, dt)
            x += adv + dgap
        self._hud_draw(verts, (orange_rgb[0], orange_rgb[1], orange_rgb[2], 1.0), moderngl.TRIANGLES)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def render_supersonic_streaks(self, width, height, intensity, t):
        """A FIXED number of long, thin white DARTS that live a DESYNCED animated lifecycle: each
        fades IN at a random perimeter spot, grows to full size, then fades OUT and RESPAWNS
        somewhere else — never overlapping a live neighbour. Per-dart random lifespan (0.3-0.8s) so
        they are NOT in sync. Every dart aims at the screen CENTRE (w/2, h/2) with a very narrow
        ~3 deg apex (long slivers). Sizes derive from window dims -> scales with the window.

        Called EVERY frame with a target `intensity` (0 when not supersonic). A smoothed global
        activation eases toward that target, so the whole ring fades IN on supersonic entry and OUT
        on exit instead of being hard-cut. State persists on self across frames."""
        # ---- smoothed global activation (eases the ring IN/OUT on supersonic entry/exit) ----
        target = float(np.clip(intensity, 0.0, 1.0))
        now = float(t)
        dt = max(0.0, min(0.1, now - getattr(self, "_ss_last_t", now)))
        self._ss_last_t = now
        act = getattr(self, "_ss_activation", 0.0)
        if dt > 0.0:
            act += (target - act) * (1.0 - math.exp(-dt / 0.18))   # 0.18s ease time-constant
        self._ss_activation = act
        if act <= 0.01 and target <= 0.0:
            return
        eff = act                                        # eased effective intensity

        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.pr_m_vp.write(ortho.astype('f4'))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        min_dim = min(width, height)
        cx, cy = width * 0.5, height * 0.5
        N = 24                                          # fixed dart count on screen
        dart_len = min_dim * 0.20                        # long
        min_gap = max(dart_len * 0.55, min_dim * 0.085)  # min perimeter distance between live darts

        # Perimeter parametrisation: s in [0, P) walks the border top->right->bottom->left.
        P = 2.0 * (width + height)

        def s_to_xy(s):
            s = s % P
            if s < width:
                return (s, 0.0)                          # top edge
            s -= width
            if s < height:
                return (width, s)                        # right edge
            s -= height
            if s < width:
                return (width - s, height)               # bottom edge
            s -= width
            return (0.0, height - s)                      # left edge

        def s_dist(a, b):
            d = abs(a - b) % P
            return min(d, P - d)

        def _new_period():
            return random.uniform(0.3, 0.8)              # per-dart random lifespan -> desynced

        def _new_half_angle():
            return math.tan(math.radians(random.uniform(0.75, 1.0)))   # per-dart HALF-angle 0.75-1.0deg
            #   (apex TOTAL 1.5-2.0deg) -> darts read as slightly different thicknesses, not uniform.

        def _new_opacity():
            return random.uniform(0.5, 0.9)              # per-dart random opacity, re-rolled on respawn

        # Lazy/persistent init. Re-seed if the count or the window perimeter changed (resize).
        # dart record: [s_pos, birth, period, half_w_ratio, opacity_mult]
        st = getattr(self, "_ss_darts", None)
        if st is None or len(st) != N or getattr(self, "_ss_P", -1.0) != round(P):
            st = []
            for i in range(N):
                per = _new_period()
                st.append([(P * i / N + random.uniform(0.0, P / N * 0.5)) % P,   # spread out
                           now - random.uniform(0.0, per),                       # desync each phase
                           per, _new_half_angle(), _new_opacity()])
            self._ss_darts = st
            self._ss_P = round(P)

        # Per-dart draw calls (this pipeline only supports one uniform color per _hud_draw call, so a
        # shared opacity across all darts isn't possible without per-vertex color -- 24 tiny draws/frame
        # is negligible cost).
        darts_out = []
        for d in st:
            s_pos, birth, period, half_w_ratio, op_mult = d
            p = (now - birth) / period
            if p >= 1.0:
                # Respawn: pick a fresh spot at least min_gap (perimeter distance) from every other
                # dart, and fresh randomised lifespan/angle/opacity so they stay desynced & varied.
                others = [o[0] for o in st if o is not d]
                new_s = None
                for _ in range(24):
                    cand = random.uniform(0.0, P)
                    if all(s_dist(cand, os) > min_gap for os in others):
                        new_s = cand
                        break
                d[0] = new_s if new_s is not None else random.uniform(0.0, P)
                d[1] = now
                d[2] = _new_period()
                d[3] = half_w_ratio = _new_half_angle()
                d[4] = op_mult = _new_opacity()
                s_pos, birth, period = d[0], d[1], d[2]
                p = 0.0

            # Per-dart size envelope: ONE continuous smooth in->out motion (grow to full at mid-life,
            # shrink back). sin(pi*p) is 0 at birth, 1 at p=0.5, 0 at death. Multiplied by the eased
            # global activation `eff` so the whole ring also fades smoothly on supersonic entry/exit.
            env = math.sin(math.pi * max(0.0, min(1.0, p)))
            scale = env * eff
            if scale <= 0.02:
                continue

            px, py = s_to_xy(s_pos)
            dx, dy = cx - px, cy - py
            dd = math.hypot(dx, dy)
            if dd < 1e-6:
                continue
            ux, uy = dx / dd, dy / dd                     # unit aim direction -> screen centre
            L = dart_len * scale
            half_w = L * half_w_ratio
            ax, ay = px + ux * L, py + uy * L             # apex reaches in toward centre
            perp_x, perp_y = -uy, ux                       # base perpendicular to the aim direction
            dart_verts = [
                (px + perp_x * half_w, py + perp_y * half_w, 0.0),
                (px - perp_x * half_w, py - perp_y * half_w, 0.0),
                (ax, ay, 0.0),
            ]
            a_ = op_mult * eff
            for vx_, vy_, vz_ in dart_verts:
                darts_out.append((vx_, vy_, vz_, 1.0, 1.0, 1.0, a_))
        self._hud_ortho = ortho
        self._hud_draw_tris(np.asarray(darts_out, dtype="f4") if darts_out else None)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def render_pai_hud(self, width, height, gail_hud):
        """GAIL-training-only discriminator readout: LARGE RED text, top-right, e.g.
        "P_AI 0.05  D2" (appends " (HARD)" when flagged, shows "N/A" when p_ai is null).

        FLAG-GATED: `gail_hud` comes straight from socket_listener's parse of the optional
        packet field "gail_hud" ({"p_ai": float|null, "label": str, "hard": bool}) and defaults to
        None when the sender doesn't include it. None/empty here means normal (non-GAIL) play mode
        -> this function draws nothing at all. The mere presence of the field is the flag; there is
        no separate on/off switch to keep in sync.

        Mirrors render_boost_hud's approach (ortho projection + the same hud_vbo/_hud_draw quad
        pipeline), but needs letters the 7-segment digit font can't draw, so it uses the 5x7
        dot-matrix font (_FONT_5X7 / _char_pixel_verts) defined above instead.

        Resize-safe: every size/position below is recomputed from the CURRENT width/height on
        every call (no cached/absolute pixel values), so it stays correctly scaled and anchored
        top-right through a live window resize.
        """
        if not gail_hud:
            return

        p_ai = gail_hud.get("p_ai")
        label = str(gail_hud.get("label") or "D")
        hard = bool(gail_hud.get("hard", False))
        elo = gail_hud.get("elo")

        pai_str = "N/A" if p_ai is None else "{:.2f}".format(float(p_ai))
        line1 = "P_AI {}  {}".format(pai_str, label)
        if hard:
            line1 += " (HARD)"
        lines = [line1]
        if elo is not None:                                # ELO under the p_ai text, SAME size
            lines.append("ELO {:+d}".format(int(elo)))

        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.pr_m_vp.write(ortho.astype('f4'))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # 30% smaller than before (was height*0.05, clamp [30,130]) per user; still the biggest HUD text.
        ch_h = float(np.clip(height * 0.035, 21.0, 91.0))
        ch_w = ch_h * 0.62
        gap = ch_w * 0.32
        advance = ch_w + gap
        line_gap = ch_h * 0.35                              # vertical gap between the stacked lines
        total_w = max(advance * len(t) - gap for t in lines)
        block_h = ch_h * len(lines) + line_gap * (len(lines) - 1)

        margin = max(20.0, min(width, height) * 0.025)
        right_x = width - margin
        top_y = margin

        # Dim backing panel behind BOTH lines so red text stays legible over a bright sky.
        pad_x, pad_y = ch_w * 0.5, ch_h * 0.22
        self._hud_draw(
            self._rect_verts(right_x - total_w - pad_x, top_y - pad_y, right_x + pad_x, top_y + block_h + pad_y),
            (0.0, 0.0, 0.0, 0.55), moderngl.TRIANGLES
        )

        # The 5x7 dot-matrix glyphs are rebuilt as individual quads in Python, ~2 ms/frame for two
        # lines. The text only changes when p_ai/Elo change (once per training iteration at most), so
        # cache the geometry and rebuild only when the strings or the layout actually differ.
        ck = (tuple(lines), round(right_x, 2), round(top_y, 2), round(ch_w, 3), round(ch_h, 3),
              round(advance, 3), round(gap, 3), round(line_gap, 3))
        if getattr(self, "_pai_cache_key", None) != ck:
            verts = []
            y = top_y
            for t in lines:
                lw = advance * len(t) - gap                 # right-align each line under the other
                x = right_x - lw
                for ch in t:
                    verts += self._char_pixel_verts(ch, x, y, ch_w, ch_h)
                    x += advance
                y += ch_h + line_gap
            self._pai_cache_key = ck
            self._pai_cache_verts = np.asarray(verts, dtype='f4') if verts else np.zeros((0, 3), 'f4')
        self._hud_draw(self._pai_cache_verts, (1.0, 0.08, 0.05, 1.0), moderngl.TRIANGLES)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    @staticmethod
    def _pitch_dir(d, angle_deg):
        """Rotate direction d up (+) / down (-) by angle_deg around the horizontal axis."""
        d = safe_normalize(Vector3(d))
        up = Vector3((0.0, 0.0, 1.0)) - d * float(d[2])
        if up.length < 1e-4:
            return d
        up = safe_normalize(up)
        a = math.radians(angle_deg)
        return safe_normalize(d * math.cos(a) + up * math.sin(a))

    def calc_camera_state(self, state, interp_ratio, delta_time):
        self._update_auto_spectate(state, interp_ratio)
        # Pose-editor override: sit behind the selected object (billiard/chase POV) while posing.
        pc = getattr(state_manager, "pose_cam", None)
        if pc is not None:
            aspect = max(getattr(self, "_aspect", 16 / 9), 1e-3)     # config FOV is horizontal
            return Vector3(pc[0]), Vector3(pc[1]), math.degrees(
                2.0 * math.atan(math.tan(math.radians(self.config.camera_fov.val) / 2.0) / aspect))

        pos = Vector3((-4000, 0, 1000))
        ball_pos = state.ball_state.get_pos(interp_ratio)

        cam_dir = safe_normalize(ball_pos - pos)

        is_spectating_car = self.spectate_idx > -1 and len(state.car_states) > self.spectate_idx
        if is_spectating_car:
            car_pos = state.car_states[self.spectate_idx].phys.get_pos(interp_ratio)
            car_vel = state.car_states[self.spectate_idx].phys.get_vel(interp_ratio)
            car_forward = state.car_states[self.spectate_idx].phys.get_forward(interp_ratio)

            # Ball cam -- Rocket League's model: the camera sits at the car + a YAW-ONLY offset of
            # (-Distance, 0, +Height), yawed to face the ball, looking at the ball, then pitched by the
            # Angle setting. Stiffness < 1 lets the camera pull back at speed (RL behaviour).
            if True:
                speed = car_vel.length
                stiff = self.config.camera_stiffness.val
                dist = self.config.camera_distance.val * (1.0 + (1.0 - stiff) * 0.30 * min(1.0, speed / 2300.0))
                height = self.config.camera_height.val
                to_ball = ball_pos - car_pos
                yaw_dir = Vector3((to_ball[0], to_ball[1], 0.0))
                if yaw_dir.length < 1.0:
                    yaw_dir = Vector3((car_forward[0], car_forward[1], 0.0))
                yaw_dir = safe_normalize(yaw_dir)
                ball_cam_pos = car_pos - yaw_dir * dist + Vector3((0.0, 0.0, height))
                ball_cam_dir = self._pitch_dir(safe_normalize(ball_pos - ball_cam_pos), self.config.camera_angle.val)

            # Calculate car cam dir: genuinely AIRBORNE (>=200uu above the floor AND not touching any
            # surface) -> follow MOMENTUM (velocity direction, full 3D) so an aerial dribble stays
            # stable even as the car's facing spins around underneath the ball. Near the ground OR
            # touching a WALL/ceiling (on_ground is a general surface-contact flag here, not floor-only
            # -- a car driving up a wall is NOT "in the air") -> follow the car's FACING, XY-only
            # (ignore pitch) so it reads as a normal level chase cam. car_pos.z is height above the
            # floor (floor = z=0).
            if True:
                on_surface = bool(state.car_states[self.spectate_idx].on_ground)
                height_above_floor = car_pos.z
                if on_surface or height_above_floor < 200.0:
                    cd = car_forward * Vector3((1, 1, 0))
                    car_cam_dir_raw = safe_normalize(cd) if cd.length > 0.1 else Vector3((1, 1, 0))
                else:
                    if car_vel.length > 50:
                        car_cam_dir_raw = safe_normalize(car_vel)
                    else:
                        cd = car_forward * Vector3((1, 1, 0))
                        car_cam_dir_raw = safe_normalize(cd) if cd.length > 0.1 else Vector3((1, 1, 0))
                # SAME base distance as ball cam (camera_distance.val), and -- like ball cam's own
                # "make sure we are actually of the correct distance" step -- RENORMALIZED back to that
                # exact distance AFTER the z-override below. Without this, however much of the unit
                # direction's length happened to be vertical (car_cam_dir_raw.z, discarded by the
                # z-override) was simply LOST, so the resulting 3D offset length varied by mode: ground/
                # wall's direction is always purely horizontal (nothing lost, offset stayed correct-
                # looking "by luck"), while air's direction (raw 3D velocity) often has a big vertical
                # component, silently shrinking the horizontal spread and giving air a DIFFERENT net
                # distance than ground/wall for the exact same config value. Renormalizing guarantees
                # all three modes sit the identical `dist` away from the car, matching ball cam exactly.
                # RL car cam: same yaw-only (-Distance, +Height) offset as ball cam (no renormalising
                # height into distance -- that is what made equal settings feel unlike the game),
                # looking along the heading pitched by Angle.
                dist_cc = dist
                h_dir = safe_normalize(Vector3((car_cam_dir_raw[0], car_cam_dir_raw[1], 0.0)))
                car_cam_offset_raw = -h_dir * dist_cc + Vector3((0.0, 0.0, self.config.camera_height.val))
                car_cam_dir_raw = self._pitch_dir(car_cam_dir_raw, self.config.camera_angle.val)

                # SMOOTHED car-cam movement: exponential low-pass on the CAR-RELATIVE OFFSET (not the
                # absolute world-space camera position!). Smoothing the absolute position made the
                # camera chase a moving target -- for a fast-moving car (much faster in the air than on
                # ground/wall) the steady-state lag grows with speed (lag ~= car_speed * TAU), which is
                # exactly the "2x too far, especially in the air" bug. Smoothing the OFFSET instead
                # still irons out jerky direction changes but recombines with the CURRENT car_pos every
                # frame, so there is zero translational lag regardless of how fast the car moves. Reset
                # (snap, no lag) whenever we start/re-start spectating a car so it doesn't drift in from
                # a stale previous car's state.
                # XY (horizontal) smoothing unchanged; Z (vertical) gets EXTRA smoothing (TAU doubled)
                # per user -- "smooth it even more, only on the z axis" -- so horizontal tracking stays
                # just as responsive while vertical bob/height changes ease in more gently.
                TAU_XY = 0.24
                TAU_Z = 0.48
                lerp_xy = 1.0 - math.exp(-delta_time / TAU_XY)
                lerp_z = 1.0 - math.exp(-delta_time / TAU_Z)
                prev_offset = getattr(self, "_car_cam_offset_smooth", None)
                prev_cam_dir = getattr(self, "_car_cam_dir_smooth", None)
                if prev_offset is None:
                    car_cam_offset = car_cam_offset_raw
                    car_cam_dir = car_cam_dir_raw
                else:
                    d_off = car_cam_offset_raw - prev_offset
                    car_cam_offset = (prev_offset + d_off * Vector3((1, 1, 0)) * lerp_xy
                                      + d_off * Vector3((0, 0, 1)) * lerp_z)
                    car_cam_dir = safe_normalize(prev_cam_dir + (car_cam_dir_raw - prev_cam_dir) * lerp_xy)
                self._car_cam_offset_smooth = car_cam_offset
                self._car_cam_dir_smooth = car_cam_dir
                car_cam_pos = car_pos + car_cam_offset

            # Determine if dribbling (raw, per-frame)
            car_ball_delta = ball_pos - car_pos
            dribbling = False
            if car_vel.length > 500:
                if 90 < car_ball_delta.z < 200:
                    if (car_ball_delta * Vector3((1, 1, 0))).length < 135:
                        if ball_pos.z < 300:
                            dribbling = True

            # Duration timers: how long `dribbling` has been continuously TRUE, and continuously FALSE
            # (each resets the instant the other becomes true) -- single-frame blips on EITHER side
            # shouldn't flip the cam.
            self.dribble_duration = (getattr(self, "dribble_duration", 0.0) + delta_time) if dribbling else 0.0
            self.no_dribble_duration = 0.0 if dribbling else (getattr(self, "no_dribble_duration", 0.0) + delta_time)
            confirmed_now = self.dribble_duration > 0.3
            confirmed_ended = self.no_dribble_duration > 0.3

            # Closest OPPONENT gate: >500uu from the spectated car, checked ONLY at the MOMENT the
            # dribble is first confirmed (the instant confirmed_now flips false->true this frame) -- an
            # uncontested dribble START is safe to follow tight on the car for its whole duration, even
            # if an opponent closes in later. No opponents on the field -> gate passes vacuously.
            was_confirmed = getattr(self, "_dribble_confirmed_prev", False)
            just_confirmed = confirmed_now and not was_confirmed
            self._dribble_confirmed_prev = confirmed_now
            engaged = getattr(self, "car_cam_engaged", False)
            if not engaged and just_confirmed:
                my_team = state.car_states[self.spectate_idx].team_num
                opp_dists = [(car_pos - cs.phys.get_pos(interp_ratio)).length
                             for i, cs in enumerate(state.car_states)
                             if i != self.spectate_idx and cs.team_num != my_team and not cs.is_demoed]
                opponent_clear_at_start = (min(opp_dists) > 500.0) if opp_dists else True
                if opponent_clear_at_start:
                    engaged = True
            elif engaged and confirmed_ended:
                engaged = False
            self.car_cam_engaged = engaged

            # cam_manual: None = AUTO (sticky `engaged` state above); True/False = user forced ball/car
            # cam via Space (see keyPressEvent) -- a manual override bypasses the dribble/opponent
            # gating entirely. Either way the desired state is a single boolean target that
            # self.car_cam_time (0..1) ramps toward LINEARLY at a constant rate so a full switch always
            # takes exactly TRANSITION_SEC, in EITHER direction -- ball-cam<->car-cam is never an
            # instant cut.
            cam_manual = getattr(self, "cam_manual", None)
            want_car_cam = engaged if cam_manual is None else (not cam_manual)
            TRANSITION_SEC = 0.6 / max(0.5, self.config.camera_transition_speed.val)
            target = 1.0 if want_car_cam else 0.0
            step = delta_time / TRANSITION_SEC
            if self.car_cam_time < target:
                self.car_cam_time = min(target, self.car_cam_time + step)
            else:
                self.car_cam_time = max(target, self.car_cam_time - step)
            # Slightly smoothed transition (S-curve / smoothstep, per user's sketch): self.car_cam_time
            # is still the RAW linear 0..1 progress (so the switch still takes exactly TRANSITION_SEC
            # total), but the actual blend weight eases in/out -- slow-start, faster through the middle,
            # slow-finish -- instead of a constant-speed linear blend.
            t = self.car_cam_time
            car_cam_ratio = t * t * (3.0 - 2.0 * t)          # classic cubic smoothstep

            pos = ball_cam_pos*(1-car_cam_ratio) + car_cam_pos*car_cam_ratio
            cam_dir = safe_normalize(ball_cam_dir*(1-car_cam_ratio) + car_cam_dir*car_cam_ratio)
        else:
            self.car_cam_time = 0
            self.dribble_duration = 0.0
            self.no_dribble_duration = 0.0
            self._dribble_confirmed_prev = False
            self.car_cam_engaged = False
            self._car_cam_offset_smooth = None
            self._car_cam_dir_smooth = None

        # RL's FOV setting is HORIZONTAL; the projection wants vertical.
        aspect = max(getattr(self, "_aspect", 16 / 9), 1e-3)
        if is_spectating_car:
            vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(self.config.camera_fov.val) / 2.0) / aspect))
        else:
            vfov = self.config.camera_bird_fov.val
        # Goal celebration: look at where the ball went in (eased in/out over ~0.4 s).
        cel = getattr(state, "celebration_pos", None)
        if cel is not None:
            self._cel_last = cel
        w0 = getattr(self, "_cel_w", 0.0)
        self._cel_w = min(1.0, w0 + delta_time / 0.4) if cel is not None else max(0.0, w0 - delta_time / 0.4)
        if self._cel_w > 0.0 and getattr(self, "_cel_last", None) is not None:
            to_goal = safe_normalize(Vector3(self._cel_last) - pos)
            wgt = self._cel_w * self._cel_w * (3.0 - 2.0 * self._cel_w)
            cam_dir = safe_normalize(cam_dir * (1.0 - wgt) + to_goal * wgt)
        return pos, pos + cam_dir, vfov

    def auto_cam_enabled(self):
        return self._vis_auto_switch if self._auto_cam_key is None else self._auto_cam_key

    def _update_auto_spectate(self, state, interp_ratio):
        """Follow a stably closest active car when auto camera is on (A key, or the GUI's Vis auto switch)."""
        now = time.monotonic()
        if now >= self._vis_auto_next_poll:
            self._vis_auto_next_poll = now + 0.25
            path = os.environ.get("RSV_CONTROL_PATH", "")
            try:
                mtime = os.path.getmtime(path)
                if mtime != self._vis_auto_control_mtime:
                    with open(path, "r", encoding="utf-8") as f:
                        ctrl = json.load(f)
                    self._vis_auto_switch = bool(ctrl.get("vis_auto_switch", False))
                    self._vis_auto_control_mtime = mtime
            except (OSError, ValueError, TypeError):
                # A control.json replacement can be observed between unlink and rename. Keep the
                # last known setting and retry on the next short poll rather than affecting rendering.
                pass

        if not self.auto_cam_enabled():
            self._vis_auto_candidate = -1
            return

        # Kickoff (ball parked at centre field): follow a RANDOM car until the ball is first touched,
        # instead of whoever happens to be closest.
        bp = state.ball_state.next_pos
        at_kickoff = (abs(float(bp[0])) < 5 and abs(float(bp[1])) < 5 and float(bp[2]) < 100
                      and state.ball_state.next_vel.length < 1.0)
        if at_kickoff:
            if not getattr(self, "_kickoff_pick_done", False):
                active_idx = [i for i, cs in enumerate(state.car_states) if not cs.is_demoed]
                if active_idx:
                    self.spectate_idx = random.choice(active_idx)
                    self._car_cam_offset_smooth = None
                    self._car_cam_dir_smooth = None
                    self._kickoff_pick_done = True
            self._vis_auto_candidate = -1
            return
        self._kickoff_pick_done = False

        active = [(i, cs) for i, cs in enumerate(state.car_states) if not cs.is_demoed]
        if not active:
            return
        ball_pos = state.ball_state.get_pos(interp_ratio)
        closest_idx = min(active, key=lambda item: (item[1].phys.get_pos(interp_ratio) - ball_pos).length)[0]
        if closest_idx == self.spectate_idx:
            self._vis_auto_candidate = -1
            return

        if closest_idx != self._vis_auto_candidate:
            self._vis_auto_candidate = closest_idx
            self._vis_auto_candidate_since = now
            return

        # The new car has been closest continuously for 0.75 seconds: switch and snap the relative
        # camera smoothing state to that car. This is intentionally independent of the Space/P camera
        # controls: turning the GUI option off immediately restores fully manual target selection.
        if now - self._vis_auto_candidate_since >= 0.75:
            self.spectate_idx = closest_idx
            self._vis_auto_candidate = -1
            self._car_cam_offset_smooth = None
            self._car_cam_dir_smooth = None

    def _ensure_render_target(self, width, height):
        """-> (render_w, render_h). The 3D scene ALWAYS renders into our own MSAA framebuffer (the
        window's framebuffer is single-sample): at the window size times RSV_RENDER_SCALE, reduced only
        if that exceeds RSV_MAX_RENDER_MP megapixels. paint() resolves it and blits it (bilinear) to the
        window, then draws the HUD straight onto the window at full native resolution."""
        scale, cap, samples = RENDER_SCALE, MAX_RENDER_MP, self.samples
        if self.window_mode:               # Edit Settings > Graphics
            res = self.config.gfx_resolution
            scale = {"150": 1.5, "200": 2.0}.get(res, 1.0)
            cap = MAX_RENDER_MP if res == "auto" else 1e9
            samples = min(int(self.config.gfx_aa), int(self.ctx.info.get("GL_MAX_SAMPLES", 8)))
        mp = width * height * scale * scale / 1e6
        if mp > cap:
            scale *= math.sqrt(cap / mp)
        render_w, render_h = max(1, round(width * scale)), max(1, round(height * scale))
        if self._cap_size != (render_w, render_h, samples):
            if self.cap_fbo is not None:
                self.cap_fbo.release(); self.cap_resolve_fbo.release()
            s = samples if samples > 1 else 0
            self.ctx.multisample = s > 1
            self.cap_fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer((render_w, render_h), samples=s)],
                depth_attachment=self.ctx.depth_renderbuffer((render_w, render_h), samples=s),
            )
            tex = self.ctx.texture((render_w, render_h), 3)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.cap_resolve_fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._cap_size = (render_w, render_h, samples)
        self.render_target = self.cap_fbo
        return render_w, render_h

    def paint(self, width, height, screen_fb, clip_capture=True):
        """Render one frame into screen_fb (the window's framebuffer, or an offscreen one headless)."""
        self.screen_fb = screen_fb
        render_w, render_h = self._ensure_render_target(width, height)
        self.render_target.use()
        self.ctx.viewport = (0, 0, render_w, render_h)

        cur_time = time.time()
        delta_time = cur_time - self.last_render_time

        _r0 = time.perf_counter()
        if self._gpu_query is not None:
            self._gpu_query.__enter__()
        state, interp_ratio, spectated = self.render(cur_time, delta_time, render_w, render_h)

        # MSAA resolve (same size), then scale into the window with the fullscreen-triangle blit.
        self.ctx.copy_framebuffer(self.cap_resolve_fbo, self.cap_fbo)
        screen_fb.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.cap_resolve_fbo.color_attachments[0].use(location=0)
        self._quad_prog['Tex'].value = 0
        self._quad_vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.enable(moderngl.BLEND)

        # HUD / overlays at full native resolution, straight onto the window framebuffer.
        self.render_target = screen_fb
        self._hud_off = 0
        _h0 = time.perf_counter() if PERF else 0.0
        self.render_hud(state, interp_ratio, spectated, cur_time, width, height)
        if PERF:
            self._perf_hud += time.perf_counter() - _h0
        if self._gpu_query is not None:
            self._gpu_query.__exit__(None, None, None)
            self._perf_gpu = getattr(self, "_perf_gpu", 0.0) + self._gpu_query.elapsed / 1e6
        _r1 = time.perf_counter()

        # Feed the finished frame (scene + HUD) into the rolling clip buffer, and honour a clip request
        # from the training GUI (the 'C' hotkey triggers the same save directly).
        if clip_capture:
            state_manager.record_cam(cur_time, int(self.spectate_idx), self.cam_manual)
            self.clip_recorder.capture(self.ctx, width, height, screen_fb)
            self.clip_recorder.poll_signal()
        _r2 = time.perf_counter()

        self.fps_counter += 1

        if int(cur_time) > int(self.last_render_time):
            self.last_fps = self.fps_counter
            self.fps_counter = 0

        if PERF:
            self._perf_render += (_r1 - _r0)
            self._perf_capture += (_r2 - _r1)
            self._perf_frames += 1
            span = cur_time - self._perf_t0
            if span >= 2.0:
                n = max(1, self._perf_frames)
                print("[perf] fps={:.0f} cpu_render={:.2f} gpu={:.2f} lock={:.2f} deepcopy={:.2f} pads={:.2f} ball={:.2f} cars={:.2f} arena={:.2f} fx={:.2f} hud={:.2f} capture={:.2f} (ms) scene={}x{}".format(
                    self._perf_frames / span,
                    self._perf_render / n * 1000.0,
                    getattr(self, "_perf_gpu", 0.0) / n,
                    self._perf_lock / n * 1000.0,
                    self._perf_deepcopy / n * 1000.0,
                    self._perf_pads / n * 1000.0,
                    self._perf_ball / n * 1000.0,
                    self._perf_cars / n * 1000.0,
                    self._perf_arena / n * 1000.0,
                    self._perf_tail / n * 1000.0,
                    self._perf_hud / n * 1000.0,
                    self._perf_capture / n * 1000.0, render_w, render_h), flush=True)
                self._perf_render = self._perf_capture = self._perf_deepcopy = 0.0
                self._perf_lock = self._perf_gpu = 0.0
                self._perf_pads = self._perf_cars = self._perf_arena = 0.0
                self._perf_ball = self._perf_tail = 0.0
                self._perf_lines = self._perf_hud = self._perf_pai = self._perf_ui = 0.0
                self._perf_frames = 0
                self._perf_t0 = cur_time

        self.last_render_time = cur_time

    @staticmethod
    def _model_matrix(pos, forward, up, scale=1.0):
        fx_, fy, fz = float(forward[0]), float(forward[1]), float(forward[2])
        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
        # right = forward x up ; the matrix's 2nd column is -right (= left), as in render_model
        rx, ry, rz = fy * uz - fz * uy, fz * ux - fx_ * uz, fx_ * uy - fy * ux
        return np.array([
            fx_ * scale, fy * scale, fz * scale, 0,
            -rx * scale, -ry * scale, -rz * scale, 0,
            ux * scale, uy * scale, uz * scale, 0,
            float(pos[0]), float(pos[1]), float(pos[2]), 1,
        ], dtype="f4")

    def render(self, total_time, delta_time, width, height):
        """Draw the 3D scene into self.render_target. Returns (state, interp_ratio, spectated) for the
        HUD pass that paint() runs afterwards at native resolution."""
        # NOTE the split: _perf_lock is time spent WAITING for the socket-listener thread to release
        # the state mutex (plus rotate_with_ang_vel), _perf_deepcopy is the copy itself.
        _lk0 = time.perf_counter() if PERF else 0.0
        with global_state_mutex:
            state_manager.apply_due_packets(time.time())
            if not global_state_manager.state.ball_state.has_rot:
                global_state_manager.state.ball_state.rotate_with_ang_vel(delta_time)

            _dc0 = time.perf_counter() if PERF else 0.0
            state = copy.deepcopy(global_state_manager.state)
            if PERF:
                self._perf_deepcopy += time.perf_counter() - _dc0
                self._perf_lock += _dc0 - _lk0
        self.prev_state = state
        self.spectate_count = len(state.car_states)

        # Streaming tools can lock the camera to a specific car via the "spectate_idx" packet field
        # (e.g. view_credit.py aims the POV at the highest-credit player). Opt-in: only when sent.
        _fsi = getattr(state_manager, "forced_spectate_idx", None)
        if _fsi is not None and 0 <= int(_fsi) < self.spectate_count:
            if int(_fsi) != self.spectate_idx:
                self._car_cam_offset_smooth = None
                self._car_cam_dir_smooth = None
            self.spectate_idx = int(_fsi)

        # two supersonic trail ribbons per car (rear wheels)
        while len(self.car_ribbons) != 2 * len(state.car_states):
            if len(self.car_ribbons) < 2 * len(state.car_states):
                self.car_ribbons.append(RibbonEmitter())
            else:
                self.car_ribbons.pop()

        cur_time = time.time()
        interp_interval = max(state.recv_interval, 1e-6)
        interp_ratio = min(max((cur_time - state.recv_time) / interp_interval, 0), 1)

        self.ctx.clear(0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.cull_face = "back"
        self.ctx.front_face = "cw"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._aspect = width / max(1, height)
        camera_pos, camera_target_pos, camera_fov = self.calc_camera_state(state, interp_ratio, delta_time)
        proj = Matrix44.perspective_projection(camera_fov, -width/height, 1.0, 50 * 1000.0)
        lookat = Matrix44(fastvec.look_at(camera_pos, camera_target_pos, (0.0, 0.0, 1.0)))
        vp = (proj * lookat).astype('f4')
        vp_bytes = vp.tobytes()
        cam_bytes = Vector3(camera_pos).astype('f4').tobytes()

        self.pr_camera_pos.write(cam_bytes)
        self.pr_m_vp.write(vp_bytes)
        self.pra_m_vp.write(vp_bytes)
        for prog in (self.prog_rl_arena, self.prog_car, self.prog_ball):
            prog["m_vp"].write(vp_bytes)
            prog["camPos"].write(cam_bytes)

        # Listener for 3D sound: "right" = the world direction that appears on the RIGHT of the screen
        # (the projection mirrors x, see the negative aspect above).
        fwd_v = safe_normalize(Vector3(camera_target_pos) - Vector3(camera_pos))
        scr_right = safe_normalize(Vector3(pyrr.vector3.cross(Vector3((0.0, 0.0, 1.0)), fwd_v)))
        self.audio.set_listener(camera_pos, scr_right)

        spectated = self.spectate_idx if 0 <= self.spectate_idx < len(state.car_states) else -1
        self._spectated_now = spectated

        tnow = float(time.time() % 3600.0)

        _p0 = time.perf_counter() if PERF else 0.0
        if not (state.boost_pad_states is None) and state.gamemode != "heatseeker": # Render boost pads
            self._ensure_pad_cache(state)
            states = state.boost_pad_states
            cache = self._pad_static
            n_p = min(len(states), len(cache))
            locs = state.boost_pad_locations
            now_p = time.time()
            if self._pad_prev is None or len(self._pad_prev) != n_p:
                self._pad_prev = [bool(x) for x in states[:n_p]]
                self._pad_pick_t = [None] * n_p
                self._pad_spawn_t = [None] * n_p
            for i in range(n_p):
                act = bool(states[i])
                if self._pad_prev[i] and not act:              # just picked up
                    self._pad_pick_t[i] = now_p
                    self.fx.pad_pickup((float(locs[i][0]), float(locs[i][1]), 0.0), bool(cache[i][0]))
                elif act and not self._pad_prev[i]:            # just respawned
                    self._pad_spawn_t[i] = now_p
                self._pad_prev[i] = act
            pp = self.prog_pad
            pp["m_vp"].write(vp_bytes)
            pp["camPos"].write(cam_bytes)
            self.t_boostpad.use(location=0)
            pp["Texture"].value = 0
            pp["ghost"].value = 0.0
            self.render_target.use()
            ghosts = []
            for i in range(n_p):
                is_big, mat = cache[i]
                pp["m_model"].write(mat)
                if states[i]:
                    st_ = self._pad_spawn_t[i]
                    pp["flash"].value = max(0.0, 1.0 - (now_p - st_) / 0.35) if st_ is not None else 0.0
                    pp["pulse"].value = math.sin(now_p * 3.0 + i * 1.7)
                    self.pad_vaos_rl[self._pad_vaos[2 * is_big + 1]].render(moderngl.TRIANGLES)
                else:
                    pp["flash"].value = 0.0
                    self.pad_vaos_rl[self._pad_vaos[2 * is_big]].render(moderngl.TRIANGLES)
                    pt = self._pad_pick_t[i]
                    if pt is not None:
                        dur = 10.0 if is_big else 4.0
                        since = now_p - pt
                        if since > 0.4:
                            ghosts.append((mat, is_big, min(1.0, (since - 0.4) / (dur - 0.4))))
            if ghosts:                                          # recharge holograms (additive fill-up)
                self.ctx.fbo.depth_mask = False
                self.ctx.blend_func = moderngl.ONE, moderngl.ONE
                for mat, is_big, g in ghosts:
                    pp["m_model"].write(mat)
                    pp["ozRange"].value = (3.3, 42.4) if is_big else (3.1, 7.0)
                    pp["ghost"].value = max(0.02, g)
                    self.pad_vaos_rl[self._pad_vaos[2 * is_big + 1]].render(moderngl.TRIANGLES)
                pp["ghost"].value = 0.0
                self.ctx.fbo.depth_mask = True
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if PERF:
            self._perf_pads += time.perf_counter() - _p0

        _b0 = time.perf_counter() if PERF else 0.0
        ball_phys = state.ball_state
        ball_pos = ball_phys.get_pos(interp_ratio)
        self.ctx.disable(moderngl.CULL_FACE)
        self.prog_ball["m_model"].write(self._model_matrix(ball_pos, ball_phys.get_forward(interp_ratio),
                                                           ball_phys.get_up(interp_ratio)).tobytes())
        bx_, by_, bz_ = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        in_goal = abs(bx_) < 893.0 and bz_ < 642.775 and abs(by_) > 5124.25 - 91.25
        if in_goal != getattr(self, "_ball_in_goal", None):
            self._ball_in_goal = in_goal
            self.prog_ball["inGoal"].value = 1.0 if in_goal else 0.0
        self.render_target.use()
        if not getattr(state, "ball_hidden", False):          # out of play during a goal celebration
            self.ball_vao.render(moderngl.TRIANGLES)
        self._update_ball_trail(state, ball_phys, ball_pos, interp_ratio, delta_time)
        if state.gamemode == "heatseeker":
            ball_speed = ball_phys.get_vel(interp_ratio).length
            self.ball_ribbon.update(ball_speed > 600, 0, ball_pos, Vector3((100, 0, 0)), 0.8, delta_time)
            if ball_phys.is_teleporting():
                self.ball_ribbon.points.clear()
            self.render_ribbon(self.ball_ribbon, camera_pos, 0.8, width=50, start_taper_time=0.08,
                               color=Vector4((1, 1, 1, 0.75)))
        if PERF:
            self._perf_ball += time.perf_counter() - _b0

        _c0 = time.perf_counter() if PERF else 0.0
        casters = [] if getattr(state, "ball_hidden", False) else [(ball_pos[0], ball_pos[1], ball_pos[2], 98.0)]
        caster_fwd = [(0.0, 0.0)]
        boosting = {}
        now = time.time()
        for i, car_state in enumerate(state.car_states):
            if car_state.is_demoed:
                self.car_ribbons[2 * i].points.clear(); self.car_ribbons[2 * i + 1].points.clear()
                continue
            car_pos = car_state.phys.get_pos(interp_ratio)
            car_forward = car_state.phys.get_forward(interp_ratio)
            car_up = car_state.phys.get_up(interp_ratio)
            car_vel = car_state.phys.get_vel(interp_ratio)
            team = int(car_state.team_num) & 1

            self.fx.car_poses[i] = (car_pos, car_forward, car_up)      # flip-reset disc follows the car
            self._update_flip_streaks(i, car_pos, car_forward, car_up, delta_time, car_state.phys.is_teleporting(),
                                      bool(car_state.on_ground))
            car_model = self._model_matrix(car_pos, car_forward, car_up).tobytes()
            self.prog_car["m_model"].write(car_model)
            self.prog_car["bodyCol"].value = TEAM_BODY[team]
            self.prog_car["wheelGlow"].value = 0.0
            self.prog_car["glowCol"].value = (0.55, 0.9, 1.0)
            self.prog_car["brakeLight"].value = 0.0
            self.render_target.use()
            self.car_vao.render(moderngl.TRIANGLES)
            # wheels: suspension travel + spin + front steering (carrig.py)
            for wv, wm in zip(self.wheel_vaos, self.wheel_rig.matrices(
                    i, car_pos, car_forward, car_up, car_vel, bool(car_state.on_ground), delta_time)):
                self.prog_car["m_model"].write(wm)
                wv.render(moderngl.TRIANGLES)

            if len(casters) < 9:
                f2 = Vector3((car_forward[0], car_forward[1], 0.0))
                f2 = f2 / max(f2.length, 1e-4)
                casters.append((car_pos[0], car_pos[1], car_pos[2], 72.0))
                caster_fwd.append((f2[0], f2[1]))

            if car_state.is_boosting:
                self.fx.boost(i, tuple(car_pos), tuple(car_forward), tuple(car_up), tuple(car_vel), team, delta_time,
                              car_model)
                self._boost_last[i] = cur_time
            # SOUND hold: bots feather boost (0.1 s presses, ~0.1 s gaps); treat presses < 0.35 s apart
            # as one boost so the Alpha ignition + tail don't retrigger twice a second ("bubbling").
            # The flame above stays exact.
            if cur_time - self._boost_last.get(i, -1e9) <= self.BOOST_SOUND_HOLD:
                boosting[i] = (tuple(car_pos), i == spectated, float(car_vel.length))

            # supersonic trails from the rear wheels (RL "classic" trail) -- batched in fx.render
            # RL only draws the supersonic trail while the wheels are on a surface
            supersonic = car_vel.length >= self.SUPERSONIC_SPEED and bool(car_state.on_ground)
            if supersonic or self.car_ribbons[2 * i].points or self.car_ribbons[2 * i + 1].points:
                right = fastvec.cross(car_forward, car_up)
                for k, side in enumerate((-1.0, 1.0)):
                    rib = self.car_ribbons[2 * i + k]
                    emit = car_pos - car_forward * 36.0 + right * (side * 29.0) - car_up * 13.0
                    rib.update(supersonic, 0, emit, Vector3((0.0, 0.0, 0.0)), 0.30, delta_time)
                    if car_state.phys.is_teleporting():
                        rib.points.clear()
                    if len(rib.points) > 1:
                        # team-coloured, like RL: a wide soft glow + a bright narrow core so it reads clearly
                        # ONE team colour: an upright jagged flame wall + a faint flat streak on the surface
                        tc = (0.25, 0.55, 1.0) if int(team) == 0 else (1.0, 0.50, 0.10)
                        self.fx.add_trail(rib, 0.30, 5.0, (*tc, 0.45))
                        self.fx.add_trail(rib, 0.30, 26.0, (*tc, 0.95), up=tuple(car_up))
        self.ctx.enable(moderngl.CULL_FACE)
        self.audio.update_boost(boosting)
        eng = None
        if spectated >= 0 and not state.car_states[spectated].is_demoed:
            sc_ = state.car_states[spectated]
            v_ = sc_.phys.get_vel(interp_ratio)
            f_ = sc_.phys.get_forward(interp_ratio)
            ctl = sc_.controls if getattr(sc_, "has_controls", False) else None
            eng = {"pos": tuple(sc_.phys.get_pos(interp_ratio)), "dt": delta_time,
                   "v_fwd": float(v_[0] * f_[0] + v_[1] * f_[1] + v_[2] * f_[2]), "on_ground": bool(sc_.on_ground),
                   "throttle": None if ctl is None else float(ctl.throttle),
                   "steer": None if ctl is None else float(ctl.steer), "boosting": spectated in boosting,
                   "speed": float(v_.length)}
        self.audio.update_engine(eng)
        if PERF:
            self._perf_cars += time.perf_counter() - _c0

        ###########################################

        _a0 = time.perf_counter() if PERF else 0.0
        cst = np.zeros((9, 4), "f4"); cfw = np.zeros((9, 2), "f4")
        cst[:len(casters)] = casters; cfw[:len(caster_fwd)] = caster_fwd
        self.prog_rl_arena["casters"].write(cst.tobytes())
        self.prog_rl_arena["casterFwd"].write(cfw.tobytes())
        self.prog_rl_arena["nCasters"].value = len(casters)
        self.render_target.use()
        self.prog_rl_arena["detailBias"].value = 2.0 if self.config.gfx_detail == "sharp" else 1.0
        self.prog_rl_arena["passMode"].value = 0
        if "arena" not in _SKIP:
            self.vaos['ArenaMeshCustom.obj'].render(moderngl.TRIANGLES)
        # Draw order is for overdraw: everything opaque first, so the depth test rejects the hidden
        # parts of the stadium and the sky only shades the pixels nothing else covered.
        self.ctx.disable(moderngl.CULL_FACE)
        self.prog_stadium["m_vp"].write(vp_bytes)
        self.prog_stadium["camPos"].write(cam_bytes)
        if "stadium" not in _SKIP:
            self.stadium_vao.render(moderngl.TRIANGLES, vertices=self.stadium_n)
        self.ctx.fbo.depth_mask = False
        self.prog_sky["invVP"].write(np.linalg.inv(vp.reshape(4, 4).astype("f8")).astype("f4").tobytes())
        self.prog_sky["time"].value = tnow
        if "sky" not in _SKIP:
            # sky at depth 1.0 with <=: only where nothing was drawn. (It used to sit at 0.9999 with <,
            # which overwrote everything farther than ~10 000 uu -- the "stadium vanishes far away" bug.)
            self.ctx.depth_func = "<="
            self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
            self.ctx.depth_func = "<"
        self.ctx.fbo.depth_mask = True
        self.ctx.enable(moderngl.CULL_FACE)
        # translucent glass walls + ceiling (premultiplied alpha, no depth write) so the stadium,
        # skyline and anything behind a wall show through like in game
        self.prog_rl_arena["passMode"].value = 1
        self.ctx.fbo.depth_mask = False
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        if "walls" not in _SKIP:
            self.vaos['ArenaMeshCustom.obj'].render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.fbo.depth_mask = True
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if PERF:
            self._perf_arena += time.perf_counter() - _a0
        _t0 = time.perf_counter() if PERF else 0.0

        ###########################################

        if len(state.render_state.lines) > 0:
            vertices_flat = np.array(state.render_state.lines).flatten()
            num_verts = vertices_flat.shape[0] // 3
            self.lines_vbo.write(vertices_flat.astype('f4'))
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.render_model(
                None, None, None,
                "render_lines", self.t_boost_glow, 1, Vector4((1, 1, 1, 1)),
                mode=moderngl.LINES,
                vert_amount=num_verts
            )
            self.ctx.enable(moderngl.DEPTH_TEST)

        # ---- game events -> sounds + effects, fired when interpolation reaches their packet ----
        self._drain_events(state, spectated)
        self.fx.update(delta_time)
        self.render_target.use()
        px_scale = height / (2.0 * math.tan(math.radians(camera_fov) / 2.0))
        cam_f = safe_normalize(Vector3(camera_target_pos) - Vector3(camera_pos))
        cam_r = safe_normalize(Vector3(pyrr.vector3.cross(cam_f, Vector3((0.0, 0.0, 1.0)))))
        self.fx.cam_right = np.asarray(cam_r, "f4")
        self.fx.cam_up = np.asarray(pyrr.vector3.cross(cam_r, cam_f), "f4")
        self.fx.render(vp_bytes, px_scale, camera_pos)

        if PERF:
            self._perf_tail += time.perf_counter() - _t0

        self.prev_interp_ratio = interp_ratio
        return state, interp_ratio, spectated

    # ---- events -------------------------------------------------------------------------------- #

    def _drain_events(self, state, spectated):
        q = rl_events.event_queue
        now = time.time()
        delay = min(max(state.recv_interval, 0.0), 0.12)
        while q and q[0]["t"] + delay <= now:
            ev = q.popleft()
            if now - ev["t"] > 0.6:
                continue                              # stale (window was inactive): drop silently
            try:
                self._handle_event(ev, spectated)
            except Exception as e:
                print("[events] handler error: {!r}".format(e))

    # Impact sounds (like RL): too-soft contacts are skipped, and the same sound repeating within its window only
    # plays again if clearly harder than the last one -- so a ball rubbing a post, a car grinding a wall or two
    # cars pushing each other don't machine-gun the sound. key -> (min strength, window s).
    SOUND_GATES = {"ball_hit": (130.0, 0.15), "post": (350.0, 0.50), "bounce": (300.0, 0.20),
                   "bump": (150.0, 0.40), "body": (300.0, 0.40), "land": (140.0, 0.25)}

    def _sound_gate(self, kind, key, strength, harder=1.6):
        min_s, window = self.SOUND_GATES[kind]
        now = time.time()
        gates = self.__dict__.setdefault("_snd_gates", {})
        last_t, last_s = gates.get(key, (-1e9, 0.0))
        if strength < min_s or (now - last_t < window and strength < last_s * harder):
            return False
        gates[key] = (now, strength)
        return True

    # RL ball trail: a ribbon following the ball centre in the colour of the team that touched it last, only
    # while the ball moves faster than 75 kph; a round tube, each point fading out over 600 ms, white for the
    # first 50 uu past the ball's surface. Once the ball slows under 75 kph the trail keeps being emitted for
    # 600 ms with a strength going 1 -> 0 (a reverse fade), so it tapers off instead of stopping dead.
    BALL_TRAIL_SPEED = 75.0 / 0.036                           # 75 kph in uu/s (1 uu = 1 cm)
    BALL_TRAIL_TAIL = 0.60           # after the ball drops under 75 kph it keeps emitting, fainter and fainter
    BALL_TRAIL_LIFE = 0.60
    # 70% of the previous saturation ((0.30, 0.45, 1.0) / (1.0, 0.52, 0.12)): 70% of the way from grey, same luminance
    TEAM_TRAIL = ((0.347, 0.452, 0.837), (0.878, 0.542, 0.262))

    # Flip streaks: very faint, very thin white lines left by the car's four upper corners while it flips.
    FLIP_STREAK_EMIT = 0.60          # a dodge's rotation lasts ~0.6 s
    FLIP_STREAK_LIFE = 0.40          # each point fully faded after 400 ms

    def _update_flip_streaks(self, i, car_pos, car_forward, car_up, delta_time, teleported, on_surface):
        ribs = self._corner_ribs.get(i)
        # airborne flips only: nothing while the wheels touch the floor / a wall (wavedash, wall dash)
        flipping = time.time() < self._flip_until.get(i, 0.0) and not on_surface
        if ribs is None:
            if not flipping:
                return
            ribs = self._corner_ribs[i] = [RibbonEmitter() for _ in self.car_streak_points]
        left = fastvec.cross(car_up, car_forward)
        for rib, (cx, cy, cz) in zip(ribs, self.car_streak_points):
            p = car_pos + car_forward * cx + left * cy + car_up * cz
            rib.update(flipping, 0, Vector3(p), Vector3((0.0, 0.0, 0.0)), self.FLIP_STREAK_LIFE, delta_time)
            if teleported:
                rib.points.clear()
            if len(rib.points) > 1:
                self.fx.add_trail(rib, self.FLIP_STREAK_LIFE, 0.9, (1.0, 1.0, 1.0, 0.22))
        if not flipping and not any(r.points for r in ribs):
            del self._corner_ribs[i]

    def _update_ball_trail(self, state, ball_phys, ball_pos, interp_ratio, delta_time):
        team = rl_events.g_detector.last_touch_team
        hidden = getattr(state, "ball_hidden", False)
        usable = not hidden and team is not None and state.gamemode != "heatseeker"
        now = time.time()
        if usable and ball_phys.get_vel(interp_ratio).length > self.BALL_TRAIL_SPEED:
            self._trail_fast_t = now
        since = now - getattr(self, "_trail_fast_t", -1e9)
        k = max(0.0, 1.0 - since / self.BALL_TRAIL_TAIL) if usable else 0.0
        on = k > 0.0
        if on and not self._ball_trail_on and self.ball_trail.points:
            self.ball_trail.points.clear()                     # restart: never bridge across the gap
        self._ball_trail_on = on
        self.ball_trail.update(on, 0, Vector3(ball_pos), Vector3((0.0, 0.0, 0.0)), self.BALL_TRAIL_LIFE, delta_time)
        if on and self.ball_trail.points:
            self.ball_trail.points[0].k = k                    # newest point: emission strength
        if hidden or ball_phys.is_teleporting():
            self.ball_trail.points.clear()
        if len(self.ball_trail.points) > 1 and team is not None:
            self._trail_team = int(team) & 1
        if len(self.ball_trail.points) > 1:
            self.fx.add_tube(self.ball_trail, self.BALL_TRAIL_LIFE, 14.0,
                             (*self.TEAM_TRAIL[getattr(self, "_trail_team", 0)], 0.55),
                             white_from=91.25, white_len=50.0)

    def _handle_event(self, ev, spectated):
        a = self.audio
        k = ev["kind"]
        pos = ev["pos"]
        car = ev.get("car", -1)
        local = car == spectated and car >= 0
        sfx = "_local" if local else "_other"
        stall = bool(ev.get("stall"))              # a stall: flip input, no impulse and no rotation
        if k == "dodge" and car >= 0 and not stall:
            self._flip_until[car] = time.time() + self.FLIP_STREAK_EMIT
        if k in ("jump", "doublejump", "dodge"):
            a.play(k + sfx, pos, 0.9, local)
            if not stall:
                self.fx.on_event(ev, spectated)            # the jump glow (jumps, double jumps, flips)
        elif k == "flipreset":
            a.play("flipreset" + sfx, pos, 1.0 if local else 0.85, local)
            self.fx.on_event(ev, spectated)
        elif k == "land":
            if not self._sound_gate("land", ("land", car), ev.get("strength", 0.0)):
                return
            g = min(1.0, max(0.25, (ev.get("strength", 0.0) - 140.0) / 900.0))
            a.play("land" + sfx, pos, g, local)
        elif k == "ball_hit":
            # RL doesn't re-trigger the hit sound on every tiny dribble contact: soft touches are
            # skipped, and hits closer than 0.15 s play only if clearly harder than the last one.
            if not self._sound_gate("ball_hit", "ball_hit", ev.get("strength", 0.0)):
                return
            d = a.distance(pos)
            band = "close" if d < 1500 else ("mid" if d < 4000 else "far")
            g = min(1.0, 0.45 + ev.get("strength", 0.0) / 2500.0)
            a.play("ball_hit_" + ("close" if local else band), pos, g, local)
            cp = ev.get("contact")
            if cp is not None and not ev.get("underside", True):     # body touch (not the wheels): sparks
                self.fx.sparks(cp, np.subtract(cp, pos), ev.get("strength", 0.0))
        elif k == "ball_bounce":
            d = a.distance(pos)
            band = "close" if d < 1500 else ("mid" if d < 4000 else "far")
            s_ = ev.get("strength", 0.0)
            surf = ev.get("surface")
            kind = "floor" if surf == "floor" else ("post" if surf == "post" else "wall")
            if kind == "post":
                if not self._sound_gate("post", "post", s_):
                    return
                a.play("ball_post_{}".format(band), pos, min(1.0, max(0.2, s_ / 1500.0)))
                # the metallic ring on its own: in the premix it is buried under the impact thud
                a.play("ball_post_ping", pos, min(1.0, max(0.5, s_ / 1200.0)))
            else:
                if not self._sound_gate("bounce", ("bounce", kind), s_):
                    return
                a.play("ball_bounce_{}_{}".format(kind, band), pos, min(1.0, max(0.15, s_ / 1800.0)))
        elif k == "bump":
            s = ev.get("strength", 0.0)
            if not self._sound_gate("bump", ("bump",) + tuple(sorted((car, ev.get("other", -2)))), s):
                return
            stage = 0 if s < 500 else (1 if s < 900 else (2 if s < 1400 else 3))
            involved = spectated in (car, ev.get("other", -2))
            a.play("bump_{}".format(stage), pos, min(1.0, max(0.35, s / 1400.0)), involved)
            if ev.get("spark") and ev.get("contact") is not None:
                self.fx.sparks(ev["contact"], ev.get("normal", (0, 0, 1)), s)
        elif k == "demo":
            a.play("demo", pos, 1.0)
            a.play("demo_small" + sfx, pos, 0.8, local)          # the demolished car's own layer
            if spectated >= 0 and ev.get("by", -1) == spectated:
                a.play("demolish_stinger", None, 0.8, True)     # RL's "Demolition" stat jingle for YOUR demo
            self.fx.on_event(ev, spectated)
        elif k == "body":
            # car body (roof / side / nose) into the floor, a wall or the ceiling
            if not self._sound_gate("body", ("body", car), ev.get("strength", 0.0)):
                return
            g = min(1.0, max(0.25, (ev.get("strength", 0.0) - 200.0) / 1200.0))
            a.play("body" + sfx, pos, g, local)
            self.fx.sparks(pos, ev.get("normal", (0, 0, 1)), ev.get("strength", 0.0))
        elif k == "pad":
            if local:                                  # RL only plays pickups for your own car
                a.play("pad_pickup", pos, 0.9, True)
        elif k == "goal":
            self._goal_banner = (int(ev.get("team", 0)) & 1, time.time())
            a.play("goal_explosion_default", pos, 1.0, True)
            a.play("goal_explosion", pos, 0.8, True)             # the goal event layer on top
            a.play("goal_horn", None, 0.8, True)
            self.fx.on_event(ev, spectated)
        elif k == "supersonic":
            if local:
                a.play("supersonic", pos, 0.7, True)

    # ---- HUD pass (native resolution, on the window framebuffer) ------------------------------- #

    def render_hud(self, state, interp_ratio, spectated, total_time, width, height):
        ss_target = 0.0
        if spectated >= 0:
            spectated_car = state.car_states[spectated]
            if not spectated_car.is_demoed:
                speed = spectated_car.phys.get_vel(interp_ratio).length
                if speed >= self.SUPERSONIC_SPEED:
                    ss_target = float(np.clip((speed - self.SUPERSONIC_SPEED) / 200.0, 0.0, 1.0)) * 0.5 + 0.5
                self.render_boost_hud(width, height, spectated_car.boost_amount, spectated_car.team_num)
        self.render_supersonic_streaks(width, height, ss_target, total_time)
        self.render_pai_hud(width, height, getattr(state_manager, "gail_hud", None))
        self.render_scoreboard_hud(width, height, getattr(state_manager, "scoreboard", None))
        self.render_goal_banner(width, height, state)
        self.render_clip_indicator(width, height)

        v = self.config.volume.val / 100.0          # settings-panel volume slider drives the mixer
        for name in self.config.SOUND_MIX:          # per-category sliders
            self.audio.cat[name[4:]] = getattr(self.config, name).val / 100.0
        if abs(v - self.audio.volume) > 1e-4:
            self.audio.volume = v

        ui_text = ""
        ui_text += "Render FPS: {}".format(self.last_fps) + "\n"
        ui_text += "GPU: {}".format(self.gpu_name) + "\n"
        ui_text += "Connected: {}".format(state.recv_time > 0) + "\n"
        if state.recv_interval > 0:
            ui_text += "Network rate: {:.2f}fps".format(1 / state.recv_interval) + "\n"
        ui_text += "Ball speed: {:.2f}kph".format(state.ball_state.prev_vel.length * (9 / 250)) + "\n"
        self._poll_feed_answer()
        tm = getattr(self, "_toast_msg", None)
        if tm is not None and time.time() - tm[1] < 3.0 and tm[0]:
            ui_text += ">> " + tm[0] + "\n"
        ui_text += "Camera: {}  [A] auto  [P] closest  [Space] ball cam".format(
            "auto" if self.auto_cam_enabled() else "manual") + "\n"
        if self.audio.ok:
            ui_text += "Sound: {} ({:.0f}%)  [M] mute  [H] hide panel".format(
                "muted" if self.audio.muted else "on", self.audio.volume * 100) + "\n"
        try:
            from pose_editor import g_pose_editor
            ui_text += g_pose_editor.hud_text()
            ui_text += getattr(state_manager, "hud_text", "")   # PLAY driver overlay (reward/record)
        except Exception:
            pass
        u = get_ui()
        if u is not None:
            u.set_text(ui_text)
        self.last_ui_text = ui_text

    def render_flipreset_flash(self, width, height):
        """RL v2.66 flip-reset indicator, screen part: blue light blades sweep in from the four corners
        for ~0.45 s when the SPECTATED car gets a reset (the 3D shockwave is in fx.py)."""
        age = time.time() - self.fx.screen_flash
        if age > 0.45:
            return
        k = 1.0 - age / 0.45
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        m = min(width, height)
        L = m * (0.30 + 0.25 * (1.0 - k))                # blades grow inward while fading
        tris = []
        for cx, cy, sx, sy in ((0, 0, 1, 1), (width, 0, -1, 1), (0, height, 1, -1), (width, height, -1, -1)):
            for off, wid in ((-0.30, 0.020), (0.0, 0.035), (0.30, 0.020)):
                a = math.atan2(sy, sx) + off * 0.5
                dx, dy = math.cos(a), math.sin(a)
                px, py = -dy, dx
                w = m * wid
                ex, ey = cx + dx * L, cy + dy * L
                tris += [(cx + px * w, cy + py * w, 0.0, 0.35, 0.75, 1.0, 0.85 * k),
                         (cx - px * w, cy - py * w, 0.0, 0.35, 0.75, 1.0, 0.85 * k),
                         (ex, ey, 0.0, 0.8, 0.95, 1.0, 0.0)]
        arr = np.asarray(tris, "f4")
        self.hud_c_vbo.write(arr.tobytes(), 0)
        self._hud_c_prog['m_vp'].write(ortho.astype('f4'))
        self.render_target.use()
        self.hud_c_vao.render(moderngl.TRIANGLES, vertices=len(arr))
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    # ---- input (called by the Qt widget) ------------------------------------------------------- #

    def handle_mouse_press(self, event):
        """-> True if the view changed (the widget repaints immediately)."""
        # While a play/pose tool is capturing input, RocketSimVis's own camera controls stand down.
        if state_manager.is_input_captured():
            return False
        if event.button() == Qt.LeftButton:
            if self.spectate_count == 0:
                return False
            self.spectate_idx += 1
            if self.spectate_idx >= self.spectate_count:
                self.spectate_idx = -1
            self._car_cam_offset_smooth = None
            self._car_cam_dir_smooth = None
            return True
        return False

    FEED_PORT = 9275     # the sender's side channel (focus heartbeats + these requests), see networking-format.md

    def _send_feed_request(self, msg, label):
        """Visualizer key -> the game sender: b"view:<n>" (team size) or b"det:<0|1>" (bot actions). The sender
        decides (e.g. only team sizes its bot's observation supports) and answers with "vis_msg" in a packet."""
        try:
            if getattr(self, "_feed_sock", None) is None:
                import socket as _s
                self._feed_sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            self._feed_sock.sendto(msg, ("127.0.0.1", self.FEED_PORT))
            self._feed_wait = time.time()
            self._toast("Requested " + label + "...")
        except OSError as e:
            self._toast("Could not reach the sender: {}".format(e))

    def _toast(self, msg):
        self._toast_msg = (msg, time.time())

    def _poll_feed_answer(self):
        vm = getattr(state_manager, "vis_msg", None)
        w = getattr(self, "_feed_wait", None)
        if vm is not None and w is not None and vm[1] >= w:
            self._feed_wait = None
            self._toast(vm[0])
        elif w is not None and time.time() - w > 1.5:
            self._feed_wait = None
            self._toast("No answer: this sender doesn't handle visualizer keys")

    def handle_key_press(self, event):
        # Ball-cam toggle (Space) — like Rocket League. Ignore auto-repeat so a hold = one toggle.
        if event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                cur = getattr(self, "cam_manual", None)
                self.cam_manual = False if cur is None else (not cur)
            return

        # Auto camera: follow whichever player has been closest to the ball (kickoff: a random one)
        if event.key() == Qt.Key_A:
            if not event.isAutoRepeat():
                self._auto_cam_key = not self.auto_cam_enabled()
                self._vis_auto_candidate = -1
                self._kickoff_pick_done = False
            return

        # Team size of the spectated arena: '&' / 'e-acute' / '"' (AZERTY number row) or 1 / 2 / 3.
        # Bot actions: S = stochastic, D = deterministic (not while a pose editor / play tool owns ZQSD).
        # Sent to the game sender (UDP side channel), which decides and answers (see networking-format.md).
        size = {Qt.Key_Ampersand: 1, Qt.Key_1: 1, Qt.Key_Eacute: 2, Qt.Key_2: 2,
                Qt.Key_QuoteDbl: 3, Qt.Key_3: 3}.get(event.key())
        if size is not None and not event.isAutoRepeat():
            self._send_feed_request(b"view:%d" % size, "%dv%d" % (size, size))
            return
        if event.key() in (Qt.Key_S, Qt.Key_D) and not event.isAutoRepeat() and not (
                getattr(state_manager, "edit_mode", False) or state_manager.is_pose_edit_allowed()
                or state_manager.is_input_captured()):
            det = event.key() == Qt.Key_D
            self._send_feed_request(b"det:1" if det else b"det:0", "deterministic" if det else "stochastic")
            return

        # Save a clip of the last 12 gameplay seconds to mp4
        if event.key() == Qt.Key_C:
            self.clip_recorder.save_clip()
            return

        # Sound: M = mute toggle, [ / ] = volume down / up (persisted)
        if event.key() == Qt.Key_M and not event.isAutoRepeat():
            self.audio.toggle_mute()
            return
        if event.key() in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            d = 10.0 if event.key() == Qt.Key_BracketRight else -10.0
            self.config.volume.val = min(100.0, max(0.0, self.config.volume.val + d))
            self.config.save()
            return

        # Switch to player closest to ball
        if event.key() == Qt.Key_P:
            if not (self.prev_state is None):
                closest_idx = -1
                closest_dist = 100_000
                for i in range(len(self.prev_state.car_states)):
                    player = self.prev_state.car_states[i] # type: CarState
                    dist_to_ball = (player.phys.next_pos - self.prev_state.ball_state.next_pos).length
                    if dist_to_ball < closest_dist:
                        closest_idx = i
                        closest_dist = dist_to_ball

                self.spectate_idx = closest_idx


def _is_foreground(window):
    """True only if the vis really is the OS foreground window. Qt's isActiveWindow() can already be True for a
    window Windows refused to bring forward (launched from the GUI: focus-stealing protection), which made the
    vis render + play sound + un-pause the feed behind the GUI at startup."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        u32 = ctypes.windll.user32
        u32.GetForegroundWindow.restype = ctypes.c_void_p
        u32.GetAncestor.restype = ctypes.c_void_p
        u32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        fg = u32.GetForegroundWindow()
        if not fg:
            return False
        hwnd = int(window.winId())
        return fg == hwnd or u32.GetAncestor(fg, 3) == hwnd       # 3 = GA_ROOTOWNER (popups, dialogs)
    except Exception:
        return True


def set_swap_interval(n):
    """VSync on/off at runtime on the CURRENT GL context (WGL_EXT_swap_control). -> success."""
    try:
        import ctypes
        gl = ctypes.WinDLL("opengl32")
        gl.wglGetProcAddress.restype = ctypes.c_void_p
        gl.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        addr = gl.wglGetProcAddress(b"wglSwapIntervalEXT")
        if not addr:
            return False
        return bool(ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(addr)(int(n)))
    except Exception:
        return False


class QRSVGLWidget(QtOpenGL.QGLWidget):
    """The window's GL surface: owns the Qt/OpenGL context and forwards everything to RSVRenderer."""

    def __init__(self, screen: QScreen):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setDoubleBuffer(True)
        # RSV_VSYNC=0 to bypass vsync (see git history: on a muxless-Optimus laptop an OpenGL app's own
        # vsync can stack with the dGPU->iGPU copy). Default 1 = tear-free.
        from config import Config as _Cfg
        self._vsync = int(_Cfg().gfx_vsync)          # Edit Settings > Graphics > VSync (live, see paintGL)
        fmt.setSwapInterval(self._vsync)
        # NO sample buffers on the window: MSAA is done in the renderer's own scene framebuffer.
        super(QRSVGLWidget, self).__init__(fmt, None)
        self.setMouseTracking(True)
        self.renderer = RSVRenderer()
        self.renderer.window_mode = True
        self.config = self.renderer.config

    def initializeGL(self):
        self.renderer.init_gl(moderngl.create_context())

    def paintGL(self):
        want = int(self.config.gfx_vsync)
        if want != self._vsync:
            set_swap_interval(want)
            self._vsync = want
        self.renderer.paint(self.width(), self.height(), self.renderer.ctx.screen)

    def mousePressEvent(self, event):
        if self.renderer.handle_mouse_press(event):
            # Force an immediate repaint after re-aiming the camera so the click never leaves a stale frame.
            self.update()

    def keyPressEvent(self, event):
        # H toggles the whole top-left panel (stats + Edit Settings) while focused.
        if event.key() == Qt.Key_H and not event.isAutoRepeat():
            win = self.window()
            if hasattr(win, "toggle_panel"):
                win.toggle_panel()
            return
        self.renderer.handle_key_press(event)


g_socket_listener = None
def run_socket_thread(port):
    global g_socket_listener
    g_socket_listener = SocketListener()
    g_socket_listener.run(port)

def main():
    # Log any unhandled exception to a file (PyQt aborts silently on slot exceptions; pythonw has
    # no console). This makes RocketSimVis crashes diagnosable.
    import traceback as _tb
    _crash_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_crash.txt")

    def _excepthook(t, v, tb):
        try:
            with open(_crash_log, "a") as f:
                f.write("".join(_tb.format_exception(t, v, tb)) + "\n")
        except Exception:
            pass
        sys.__excepthook__(t, v, tb)
    sys.excepthook = _excepthook

    # RSV_PORT lets a second instance (tests, a side-by-side comparison) run without stealing the
    # training vis's socket.
    port = int(os.environ.get("RSV_PORT", "9273"))

    print("Starting RocketSimVis...")

    print("Starting socket thread...")
    socket_thread = threading.Thread(target=run_socket_thread, args=(int(port),))
    socket_thread.start()

    print("Starting visualizer window...")

    app = QtWidgets.QApplication([])
    ui.update_scaling_factor(app)

    gl_widget = QRSVGLWidget(app.primaryScreen())
    window = QRSVWindow(gl_widget)
    window.showNormal()
    window._norm_geo = window.frameGeometry()   # seed the anti-auto-minimize geometry

    from PyQt5.QtCore import QTimer
    import socket as _socket
    _focus_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _was_active = [False]

    _side_next = [0.0]
    _fps_applied = [None]

    def _drive_render():
        # Edit Settings > Graphics > Frame rate cap (live)
        f = gl_widget.config.gfx_fps
        if f != _fps_applied[0]:
            _fps_applied[0] = f
            render_timer.setInterval(0 if f == -1 else (frame_ms if f == 0 else max(1, round(1000.0 / f))))
        # Only repaint the GL widget while the window is the ACTIVE window. Repainting an OpenGL
        # context for a backgrounded window crashes on some drivers — that was the vis "closing" on
        # click-away. While unfocused the window simply holds its last frame (it does NOT close).
        active = False
        try:
            active = window.isActiveWindow() and not window.isMinimized() and _is_foreground(window)
            # Suspend BEFORE repainting stops: Qt still repaints once on focus loss, which used to
            # restart the boost loop with no later frame to stop it.
            gl_widget.renderer.audio.set_active(active)
            if active:
                window.gl_widget.update()
        except Exception:
            pass
        _was_active[0] = active
        now_ = time.monotonic()
        if now_ < _side_next[0]:
            return                       # clip poll + focus heartbeat at ~30 Hz, not every frame
        _side_next[0] = now_ + 0.03
        # The GUI's "Clip 12s" request must work while the GUI (not the vis) has focus: packets are
        # recorded by the socket thread regardless, so poll the trigger file here, not only in paint.
        try:
            gl_widget.renderer.clip_recorder.poll_signal()
        except Exception:
            pass
        # Report focus to a PLAY driver (if any) so it can HOLD playback while we're hidden.
        try:
            _focus_sock.sendto(b"1" if active else b"0", ("127.0.0.1", 9275))
        except Exception:
            pass

    render_timer = QTimer()
    render_timer.setTimerType(Qt.PreciseTimer)
    render_timer.timeout.connect(_drive_render)
    # Frame pacing: 60 fps by default -- the vis shares the GPU with training, and 60 keeps the SPS cost
    # small (see tools/gpu_contention_bench.py). RSV_FPS raises/lowers it (capped at the monitor's
    # refresh rate); RSV_FRAME_MS still overrides the interval directly.
    try:
        # fastest attached screen (the primary may be a 60 Hz external while the vis sits on the 165 Hz panel)
        refresh = max(float(sc.refreshRate()) for sc in app.screens()) or 60.0
    except Exception:
        refresh = 60.0
    # Default: uncapped = the monitor's refresh (vsync paces the rest). RSV_FPS=<n> caps it lower.
    fps_env = float(os.environ.get("RSV_FPS", "0") or 0)
    fps = refresh if fps_env <= 0 else min(refresh, fps_env)
    frame_ms = int(os.environ.get("RSV_FRAME_MS", max(1, int(1000.0 / max(1.0, fps)))))
    print("[timing] monitor {:.0f} Hz -> frame interval {} ms".format(refresh, frame_ms), flush=True)
    render_timer.start(frame_ms)

    pose_timer = None
    try:
        from pose_editor import g_pose_editor
        pose_timer = QTimer()
        pose_timer.timeout.connect(g_pose_editor.poll_keys)
        pose_timer.start(8)   # ~120Hz so posing is smooth and test-mode input isn't undersampled
    except Exception:
        _tb_ = __import__("traceback").format_exc()
        try:
            open(_crash_log, "a").write("pose_editor timer setup failed:\n" + _tb_ + "\n")
        except Exception:
            pass

    app.exec_()

    print("Shutting down...")
    g_socket_listener.stop_async()
    exit()

if __name__ == "__main__":
    main()

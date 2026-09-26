r"""headless_test.py -- render RocketSimVis with NO window: scripted 2v2 scenario -> PNG screenshots,
per-frame CPU/GPU timing, event log, and (optionally) the mixed audio written to a file.

  python tools/headless_test.py --out <dir> [--size 1920x1080] [--seconds 10] [--audio]

Uses a standalone OpenGL context (moderngl.create_standalone_context) and drives RSVRenderer.paint()
into an offscreen framebuffer, feeding packets through the same GameState.read_from_json +
EventDetector path the UDP listener uses. --audio routes pygame's mixer to SDL's "disk" driver
(<out>/audio.raw, s16 stereo 48 kHz) so sound can be checked without playing anything out loud.
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--fps-cap", type=float, default=0.0, help="0 = render as fast as possible")
    ap.add_argument("--no-finish", action="store_true", help="time CPU submission only (no GPU wait)")
    return ap.parse_args()


args = parse_args()
os.makedirs(args.out, exist_ok=True)
os.environ["RSV_CLIP_DISABLE"] = "1"
if args.audio:
    os.environ["SDL_AUDIODRIVER"] = "disk"
    os.environ["SDL_DISKAUDIOFILE"] = os.path.join(args.out, "audio.raw")
else:
    os.environ.setdefault("RSV_SOUND", "0")
sys.path.insert(0, SRC)

import numpy as np
import moderngl
from PIL import Image

import main as rsv
import state_manager
import events

W, H = (int(x) for x in args.size.lower().split("x"))
DT = 1.0 / 30.0

# --------------------------------------------------------------------------------------------- #
# Scripted scenario (not physics -- just enough motion + flags to trigger every event)
# --------------------------------------------------------------------------------------------- #


class Car:
    def __init__(self, team, pos, vel):
        self.team = team
        self.pos = np.array(pos, "f8"); self.vel = np.array(vel, "f8")
        self.ang = np.zeros(3)
        self.on_ground = True; self.has_flip = True; self.demoed = False
        self.boosting = False; self.touched = False; self.boost = 60.0
        self.fwd = np.array([0, 1, 0], "f8"); self.up = np.array([0, 0, 1], "f8")
        self.pitch_spin = 0.0

    def step(self, dt):
        if self.demoed:
            return
        if not self.on_ground:
            self.vel[2] -= 650 * dt
        if self.boosting:
            h = self.fwd
            self.vel += h * 991.0 * dt
            self.boost = max(0.0, self.boost - 33.3 * dt)
        self.pos += self.vel * dt
        if self.pos[2] < 17.0:
            self.pos[2] = 17.0
            if self.vel[2] < 0:
                self.vel[2] = 0
        hv = np.array([self.vel[0], self.vel[1], 0.0])
        if np.linalg.norm(hv) > 50 and self.on_ground:
            self.fwd = hv / np.linalg.norm(hv)
        if self.pitch_spin:
            a = self.pitch_spin * dt
            right = np.cross(self.fwd, self.up)
            f = self.fwd * math.cos(a) + self.up * math.sin(a)
            self.up = np.cross(right, f); self.fwd = f
            self.up /= np.linalg.norm(self.up); self.fwd /= np.linalg.norm(self.fwd)

    def json(self):
        return {"team_num": self.team, "phys": {"pos": list(self.pos), "vel": list(self.vel), "ang_vel": list(self.ang),
                                                "forward": list(self.fwd), "up": list(self.up)},
                "boost_amount": self.boost, "is_boosting": self.boosting, "on_ground": self.on_ground,
                "has_flip": self.has_flip, "ball_touched": self.touched, "is_demoed": self.demoed}


class Scenario:
    def __init__(self):
        self.t = 0.0
        self.cars = [Car(0, (-900, -2600, 17), (300, 1300, 0)), Car(0, (1200, -1200, 17), (0, 900, 0)),
                     Car(1, (2000, 1500, 17), (-1200, 0, 0)), Car(1, (400, 1500, 17), (1200, 0, 0))]
        self.cars[0].boosting = True
        self.ball = np.array([0.0, -500.0, 93.0]); self.bvel = np.array([0.0, 0.0, 0.0])
        self.pads = [True] * 34
        self.shots = {}
        self.spectate = {0.0: 0, 3.0: 1, 5.0: 0, 8.2: -1}
        self.fired = set()

    def at(self, key, when):
        if self.t >= when and key not in self.fired:
            self.fired.add(key)
            return True
        return False

    def step(self):
        c0, c1, c2, c3 = self.cars
        for c in self.cars:
            c.touched = False
        t = self.t
        if self.at("c0_jump", 1.5):
            c0.on_ground = False; c0.vel += c0.up * 320; c0.boosting = False
        if self.at("c0_dodge", 1.8):
            c0.has_flip = False; c0.vel += c0.fwd * 500; c0.ang = np.array([6.0, 0, 0]); c0.pitch_spin = 11.0
        if self.at("c0_hit", 2.05):
            c0.touched = True; self.bvel = np.array([250.0, 1900.0, 650.0])
        if self.at("c0_spin_stop", 2.35):
            c0.pitch_spin = 0.0; c0.fwd = np.array([0, 1.0, 0]); c0.up = np.array([0, 0, 1.0]); c0.ang[:] = 0
        if not c0.on_ground and c0.pos[2] <= 17.5 and c0.vel[2] <= 0 and t > 2.0:
            c0.on_ground = True; c0.has_flip = True
        # car1: aerial to the ball + flip reset
        if self.at("c1_air", 2.4):
            c1.on_ground = False; c1.has_flip = False; c1.boosting = True
            c1.pos = np.array([150.0, 1150.0, 700.0]); c1.vel = np.array([0.0, 500.0, 700.0])
            c1.fwd = np.array([0.0, 0.6, 0.8]); c1.fwd /= np.linalg.norm(c1.fwd)
            c1.up = np.array([0.0, 0.8, -0.6]); c1.up /= np.linalg.norm(c1.up)
            self.ball = np.array([150.0, 1600.0, 1000.0]); self.bvel = np.array([0.0, 300.0, 350.0])
        if 2.4 < t < 4.2 and not c1.on_ground:
            c1.vel[2] += 650 * DT                               # "hold" altitude while aerialing
            c1.fwd = self.ball - c1.pos; c1.fwd /= np.linalg.norm(c1.fwd)
            c1.up = np.cross(np.cross(c1.fwd, np.array([0, 0, -1.0])), c1.fwd); c1.up /= np.linalg.norm(c1.up)
            to = self.ball - c1.pos
            if np.linalg.norm(to) > 180:
                c1.vel = to / np.linalg.norm(to) * 900 + self.bvel
            else:
                c1.vel = self.bvel.copy()
        if self.at("c1_reset", 3.6):
            # wheels on the ball (like RocketSim): car's floor pan facing the ball centre, 95 uu away
            to = self.ball - c1.pos; to /= np.linalg.norm(to)
            c1.up = -to
            side = np.cross(c1.up, np.array([1.0, 0.0, 0.0])); side /= np.linalg.norm(side)
            c1.fwd = np.cross(side, c1.up); c1.fwd /= np.linalg.norm(c1.fwd)
            c1.pos = self.ball - to * 95.0
            c1.has_flip = True; c1.touched = True; c1.on_ground = True
        if self.at("c1_reset_off", 3.7):
            c1.on_ground = False
            c1.pos = self.ball - (self.ball - c1.pos) / np.linalg.norm(self.ball - c1.pos) * 150
        # bump between car2 and car3
        if self.at("bump_setup", 4.2):
            c2.pos = np.array([1250.0, 1500.0, 17.0]); c3.pos = np.array([550.0, 1500.0, 17.0])
            c2.vel = np.array([-1400.0, 0.0, 0.0]); c3.vel = np.array([1400.0, 0.0, 0.0])
        if self.at("bump", 4.45):                          # collide: exchange + scatter velocities
            c2.vel = np.array([900.0, 400.0, 250.0]); c3.vel = np.array([-900.0, -300.0, 200.0])
        # car0 demos car3
        if self.at("demo_setup", 5.2):
            c0.pos = np.array([-400.0, 1000.0, 17.0]); c0.vel = np.array([0.0, 2250.0, 0.0]); c0.boosting = True
            c3.pos = np.array([-400.0, 1850.0, 17.0]); c3.vel = np.array([0.0, 0.0, 0.0])
        if self.at("demo", 5.55):
            c3.demoed = True
        if self.at("demo_end", 6.0):
            c0.boosting = False; c0.vel = np.array([0.0, 900.0, 0.0])
        # pad pickup by car0 (pad 20 = (0, 1024))
        if self.at("pad_setup", 6.2):
            c0.pos = np.array([0.0, 700.0, 17.0]); c0.vel = np.array([0.0, 1200.0, 0.0])
        if self.at("pad", 6.5):
            self.pads[20] = False
        # wall bounce + goal
        if self.at("wall_setup", 6.8):
            self.ball = np.array([3500.0, 2000.0, 300.0]); self.bvel = np.array([2200.0, 400.0, 0.0])
        if self.at("goal_setup", 7.8):
            self.ball = np.array([100.0, 4400.0, 250.0]); self.bvel = np.array([0.0, 2400.0, 50.0])
        # ball physics
        self.bvel[2] -= 650 * DT
        self.ball += self.bvel * DT
        if self.ball[2] < 92.75 and self.bvel[2] < 0:
            self.ball[2] = 92.75; self.bvel[2] = -self.bvel[2] * 0.6
        if abs(self.ball[0]) > 4000 and np.sign(self.bvel[0]) == np.sign(self.ball[0]):
            self.bvel[0] = -self.bvel[0] * 0.6
        if abs(self.ball[1]) > 5800:
            self.bvel[:] = 0
        for c in self.cars:
            c.step(DT)
        self.t += DT

    def packet(self):
        return {"gamemode": "soccar",
                "ball_phys": {"pos": list(self.ball), "vel": list(self.bvel), "ang_vel": [1.0, 2.0, 0.5]},
                "cars": [c.json() for c in self.cars], "boost_pad_states": list(self.pads)}


# --------------------------------------------------------------------------------------------- #

ctx = moderngl.create_standalone_context(require=330)
r = rsv.RSVRenderer()
r.audio.wait_loaded()                   # sound loads on a background thread
r.init_gl(ctx)
screen = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 4)], depth_attachment=ctx.depth_renderbuffer((W, H)))
print("GPU:", r.gpu_name, "size", W, H, flush=True)

sc = Scenario()
PLAYLOG = []
_orig_play = r.audio.play
def _logged_play(name, pos=None, gain=1.0, local=False):
    l, rr = r.audio._gains(pos, gain, local) if r.audio.ok else (0, 0)
    PLAYLOG.append((round(sc.t, 2), name, round(l, 2), round(rr, 2)))
    return _orig_play(name, pos, gain, local)
r.audio.play = _logged_play
SHOTS = {1.0: "01_drive_boost", 1.3: "01b_drive_boost", 2.1: "02_dodge_hit", 3.72: "03_flipreset",
         3.9: "03b_flipreset_later", 4.7: "04_bump", 5.62: "05_demo_a", 5.72: "05_demo_b", 5.85: "05_demo_c",
         6.2: "05_demo_d", 8.35: "06_goal", 9.3: "07_birdseye"}
shots_left = dict(SHOTS)
event_log = []

cpu_ms, gpu_ms = [], []
q = ctx.query(time=True)
t0 = time.time()
next_packet = t0
frames = 0
while True:
    now = time.time()
    el = now - t0
    if el > args.seconds:
        break
    while now >= next_packet:
        sc.step()
        for k in sorted(sc.spectate):
            if sc.t >= k:
                r.spectate_idx = sc.spectate[k]
        j = sc.packet()
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            st.read_from_json(j)
            st.recv_interval = now - st.recv_time if st.recv_time > 0 else DT
            st.recv_time = now
            pads = st.boost_pad_locations
        n0 = len(events.event_queue)
        events.g_detector.process(j, pads, now)
        for ev in list(events.event_queue)[n0:]:
            event_log.append((round(sc.t, 2), ev["kind"], ev.get("car"), round(ev.get("strength", 0.0))))
        next_packet += DT
    c0 = time.perf_counter()
    if r._gpu_query is None:
        with q:
            r.paint(W, H, screen, clip_capture=False)
    else:
        r.paint(W, H, screen, clip_capture=False)   # RSV_PERF: the renderer times the GPU itself
    if not args.no_finish:
        ctx.finish()
    cpu_ms.append((time.perf_counter() - c0) * 1000.0)
    gpu_ms.append(q.elapsed / 1e6 if r._gpu_query is None else r._gpu_query.elapsed / 1e6)
    frames += 1
    for ts in sorted(shots_left):
        if sc.t >= ts:
            name = shots_left.pop(ts)
            if not args.no_shots:
                img = Image.frombytes("RGBA", (W, H), screen.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
                img.convert("RGB").save(os.path.join(args.out, name + ".png"))
    if args.fps_cap > 0:
        target = t0 + frames / args.fps_cap
        while time.time() < target:
            pass

cpu = np.array(cpu_ms[10:]); gpu = np.array(gpu_ms[10:])
print("frames", frames, "avg fps", round(frames / args.seconds, 1))
print("cpu ms/frame  mean {:.2f}  p50 {:.2f}  p95 {:.2f}  max {:.2f}".format(cpu.mean(), np.median(cpu), np.percentile(cpu, 95), cpu.max()))
print("gpu ms/frame  mean {:.2f}  p50 {:.2f}  p95 {:.2f}".format(gpu.mean(), np.median(gpu), np.percentile(gpu, 95)))
print("events:")
for e in event_log:
    print("  ", e)
print("sounds played (sim t, name, L, R):")
for e in PLAYLOG:
    print("  ", e)
if args.audio:
    r.audio.stop_all()
    import pygame
    pygame.mixer.quit()

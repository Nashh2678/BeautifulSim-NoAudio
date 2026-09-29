r"""readme_shots.py -- render the README screenshots (docs/*.jpg) headless, at High settings.

  python tools/readme_shots.py [--out docs] [--size 1920x1080] [--only boost,goal]

Each shot is a short scripted scene fed frame by frame like a live stream (so trails, boost smoke and effects build
up naturally), then one frame is saved as a JPEG.
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["RSV_CLIP_DISABLE"] = "1"
os.environ.setdefault("RSV_SOUND", "0")
os.environ.setdefault("RSV_MSAA", "8")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "docs"))
ap.add_argument("--size", default="1920x1080")
ap.add_argument("--only", default="")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import moderngl
from PIL import Image
import main as rsv
import state_manager
import events as rl_events

W, H = (int(x) for x in args.size.split("x"))
ctx = moderngl.create_standalone_context(require=330)
r = rsv.RSVRenderer()
r.init_gl(ctx)
r.config.apply_preset("high")
screen = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 4)], depth_attachment=ctx.depth_renderbuffer((W, H)))


def car(team, pos, fwd=(0, 1, 0), up=(0, 0, 1), vel=(0, 0, 0), boosting=False, boost=60):
    return {"team_num": team, "phys": {"pos": list(pos), "vel": list(vel), "ang_vel": [0, 0, 0],
                                       "forward": list(fwd), "up": list(up)},
            "boost_amount": boost, "is_boosting": boosting, "on_ground": pos[2] < 30, "is_demoed": False}


def frame():
    r.paint(W, H, screen, clip_capture=False)
    ctx.finish()
    return Image.frombytes("RGBA", (W, H), screen.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")


def shot(name, mapname, scene, at, events=(), spectate=-1, touch_team=0):
    """scene(t) -> (ball_pos, ball_vel, cars, cam) with cam = (eye, target) or None (spectate camera)."""
    if args.only and name not in args.only.split(","):
        return
    r.set_map(mapname, save=False)
    r.spectate_idx = spectate
    evs = sorted(events, key=lambda e: e[0])
    t0 = time.time()
    im = None
    while True:
        t = time.time() - t0
        bp, bv, cars, cam = scene(t)
        j = {"gamemode": "soccar", "ball_phys": {"pos": list(bp), "vel": list(bv), "ang_vel": [0, 0, 0]},
             "cars": cars, "boost_pad_states": [True] * 34}
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            st.read_from_json(j); st.read_from_json(j)
            st.recv_time = time.time(); st.recv_interval = 1 / 120
        rl_events.g_detector.last_touch_team = touch_team
        state_manager.pose_cam = cam
        while evs and t >= evs[0][0]:
            ev = dict(evs.pop(0)[1]); ev["t"] = time.time()
            r._handle_event(ev, spectate)
        im = frame()
        if t >= at:
            break
    im.save(os.path.join(args.out, name + ".jpg"), quality=92)
    print("saved", name, flush=True)


# 1. Boosting towards the ball (Parc de Paris), the real ball cam behind the car
def boost_scene(t):
    y = -3200.0 + 1500.0 * t
    c = car(0, (-300.0 + 60.0 * t, y, 17.0), fwd=(0.04, 1, 0), vel=(60.0, 1500.0, 0.0), boosting=True, boost=64)
    mate = car(0, (-1300.0, y - 900.0, 17.0), vel=(0, 1400, 0))
    opp = car(1, (500.0, 1200.0, 17.0), fwd=(0, -1, 0), vel=(0, -900, 0))
    cp = c["phys"]["pos"]
    cam = ((cp[0] - 330.0, cp[1] - 520.0, 170.0), (cp[0] + 90.0, cp[1] + 380.0, 60.0))
    return (100.0, 900.0, 93.0), (0, 0, 0), [c, mate, opp], cam


shot("boost", "paris", boost_scene, 1.1, spectate=0)


# 2. Flip reset (Evening Valley): the car's wheels on the underside of the ball, seen from the side
_s = 1.0 / math.sqrt(2.0)


def reset_scene(t):
    ball = (0.0, 0.0, 800.0 + 300.0 * t)
    c = car(0, (0.0, -120.0, 740.0 + 300.0 * t), fwd=(0, 0.6, 0.8), up=(0, 0.8, -0.6), vel=(0, 200, 300))
    return ball, (0, 200, 300), [c], ((-360.0, -300.0, 700.0 + 300.0 * t), (0.0, -40.0, 800.0 + 300.0 * t))


shot("flipreset", "valley", reset_scene, 0.63,
     events=[(0.60, {"kind": "flipreset", "pos": (0.0, -80.0, 920.0), "car": 0, "team": 0, "up": (0, 0.8, -0.6),
                     "car_pos": (0.0, -120.0, 920.0), "ball_pos": (0.0, 0.0, 980.0), "normal": (0.0, -0.8, -0.6)})])


# 3. Demolition (Forbidden Temple)
def demo_scene(t):
    hit = car(0, (-160.0, -60.0, 17.0), fwd=(_s, _s, 0), vel=(1500, 1500, 0), boosting=True)
    return (600.0, 2200.0, 93.0), (0, 0, 0), [hit], ((-760.0, -700.0, 260.0), (0.0, 20.0, 110.0))


shot("demo", "temple", demo_scene, 0.5,
     events=[(0.38, {"kind": "demo", "pos": (0.0, 60.0, 17.0), "car": 1, "team": 1, "vel": (0, 800, 0)})])


# 4. Overview (Forbidden Temple): a shot flying across the pitch with its trail, cars chasing
def overview_scene(t):
    a = t * 1.3
    bp = (-2600.0 + 2800.0 * t, -1200.0 + 1400.0 * t, 250.0 + 700.0 * math.sin(a))
    bv = (2800.0, 1400.0, 910.0 * math.cos(a))
    cars = [car(0, (-2900.0 + 1900.0 * t, -1600.0 + 900.0 * t, 17.0), fwd=(0.9, 0.44, 0), vel=(1900, 900, 0),
                boosting=True),
            car(1, (1800.0 - 800.0 * t, 900.0 - 300.0 * t, 17.0), fwd=(-0.94, -0.35, 0), vel=(-800, -300, 0)),
            car(1, (2600.0, 3200.0, 17.0), fwd=(-0.6, -0.8, 0)),
            car(0, (-3400.0, -3800.0, 17.0), fwd=(0.5, 0.86, 0))]
    return bp, bv, cars, ((-2100.0, 2000.0, 650.0), (-450.0, -250.0, 450.0))


shot("overview", "temple", overview_scene, 1.15, touch_team=0)


# 5. Goal (Parc de Paris): the explosion bursting out of the orange goal, the scorer in front
def goal_scene(t):
    sc = car(0, (-350.0, 3600.0, 17.0), fwd=(0.2, 1, 0), vel=(0, 600, 0))
    return (0.0, 5300.0, 250.0), (0, 0, 0), [sc], ((-900.0, 2300.0, 500.0), (0.0, 5000.0, 300.0))


shot("goal", "paris", goal_scene, 0.72,
     events=[(0.30, {"kind": "goal", "pos": (0.0, 5300.0, 250.0), "team": 0})])

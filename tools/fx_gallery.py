r"""fx_gallery.py -- headless close-ups of each effect at fixed ages (for tuning visuals).

  python tools/fx_gallery.py --out <dir>

Static scene, camera pinned with state_manager.pose_cam, events injected straight into the fx /
sound handler, frames captured at chosen ages after each event -> one contact sheet per effect.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
os.environ["RSV_CLIP_DISABLE"] = "1"
os.environ.setdefault("RSV_SOUND", "0")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--size", default="1280x720")
ap.add_argument("--only", default="", help="comma list of scenes")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import numpy as np
import moderngl
from PIL import Image
import main as rsv
import state_manager

W, H = (int(x) for x in args.size.split("x"))
ctx = moderngl.create_standalone_context(require=330)
r = rsv.RSVRenderer()
r.audio.wait_loaded()
r.init_gl(ctx)
screen = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 4)], depth_attachment=ctx.depth_renderbuffer((W, H)))


def car(team, pos, fwd=(0, 1, 0), up=(0, 0, 1), boosting=False, boost=50):
    return {"team_num": team, "phys": {"pos": list(pos), "vel": [0, 0, 0], "ang_vel": [0, 0, 0],
                                       "forward": list(fwd), "up": list(up)},
            "boost_amount": boost, "is_boosting": boosting, "on_ground": pos[2] < 20, "is_demoed": False}


def set_scene(ball, cars):
    j = {"gamemode": "soccar", "ball_phys": {"pos": list(ball), "vel": [0, 0, 0], "ang_vel": [0, 0, 0]},
         "cars": cars, "boost_pad_states": [True] * 34}
    with state_manager.global_state_mutex:
        st = state_manager.global_state_manager.state
        for _ in range(2):                       # twice -> prev == next (no interpolation motion)
            st.read_from_json(j)
        st.recv_time = time.time(); st.recv_interval = 1 / 30


def frame():
    r.paint(W, H, screen, clip_capture=False)
    ctx.finish()
    return Image.frombytes("RGBA", (W, H), screen.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")


def run(name, eye, target, ball, cars, ev, ages, spectate=0):
    if args.only and name not in args.only.split(","):
        return
    set_scene(ball, cars)
    state_manager.pose_cam = None if eye is None else (eye, target)
    r.spectate_idx = spectate
    for _ in range(5):
        frame()
    t0 = time.time()
    if ev is not None:
        ev = dict(ev); ev["t"] = t0
        if ev.get('kind') == 'fxsparks':
            r.fx.sparks(ev['pos'], ev['normal'], ev['strength'])
        else:
            r._handle_event(ev, spectate)
    shots = []
    for a in ages:
        while time.time() - t0 < a:
            frame()
        shots.append(frame())
    sheet = Image.new("RGB", (W * 2, H * ((len(shots) + 1) // 2)))
    for i, im in enumerate(shots):
        sheet.paste(im, ((i % 2) * W, (i // 2) * H))
    sheet.save(os.path.join(args.out, name + ".png"))
    print("saved", name)


# flip reset: car under the ball, wheels touching it, camera from the side
ball = (0.0, 0.0, 800.0)
c = car(0, (0.0, -120.0, 740.0), fwd=(0, 0.6, 0.8), up=(0, 0.8, -0.6))
run("flipreset", eye=(-380.0, -330.0, 820.0), target=(0.0, 0.0, 780.0), ball=ball, cars=[c],
    ev={"kind": "flipreset", "pos": (0.0, -80.0, 740.0), "car": 0, "team": 0, "up": (0, 0.8, -0.6),
        "car_pos": (0.0, -120.0, 740.0), "ball_pos": ball, "normal": (0.0, -0.8, -0.6)}, ages=[0.005, 0.02, 0.035, 0.048])
# jump burst under a car
run("jump", eye=(-160.0, -130.0, 30.0), target=(0.0, 0.0, 30.0), ball=(0.0, 1500.0, 93.0),
    cars=[car(0, (0.0, 0.0, 22.0))], ev={"kind": "jump", "pos": (0.0, 0.0, 22.0), "car": 0, "team": 0, "up": (0, 0, 1)},
    ages=[0.01, 0.07, 0.13, 0.19], spectate=-1)
# demo on the ground
run("demo", eye=(-900.0, -900.0, 350.0), target=(0.0, 0.0, 80.0), ball=(0.0, 1500.0, 93.0),
    cars=[car(0, (-600.0, -600.0, 17.0)), car(1, (0.0, 0.0, 17.0))],
    ev={"kind": "demo", "pos": (0.0, 0.0, 17.0), "car": 1, "team": 1, "vel": (0, 800, 0)},
    ages=[0.03, 0.12, 0.3, 0.7], spectate=0)
# goal
run("goal", eye=(0.0, 3500.0, 600.0), target=(0.0, 5300.0, 250.0), ball=(0.0, 5300.0, 250.0), cars=[],
    ev={"kind": "goal", "pos": (0.0, 5300.0, 250.0), "team": 0}, ages=[0.05, 0.2, 0.45, 0.9], spectate=-1)
# a hit on the ball (sparks)
run("hit", eye=(-300.0, -220.0, 150.0), target=(0.0, 0.0, 100.0), ball=(0.0, 0.0, 93.0), cars=[car(0, (-95.0, 0.0, 17.0), fwd=(1, 0, 0))],
    ev={"kind": "fxsparks", "pos": (-74.0, -55.0, 93.0), "normal": (-0.8, -0.6, 0.1), "strength": 900.0}, ages=[0.02, 0.05, 0.09, 0.14], spectate=-1)
# boost flame close-up
run("boost", eye=(-260.0, -250.0, 90.0), target=(0.0, 0.0, 40.0), ball=(0.0, 1500.0, 93.0),
    cars=[car(1, (0.0, 0.0, 17.0), boosting=True)], ev=None, ages=[0.1, 0.25, 0.4, 0.55])

# boost meter empty / half (chase cam on the car)
state_manager.pose_cam = None
for bst in (0, 50):
    run("meter%d" % bst, eye=None, target=None, ball=(0.0, 1500.0, 93.0),
        cars=[car(0, (0.0, 0.0, 17.0), boost=bst)], ev=None, ages=[0.3], spectate=0)
# ball close-up (skin check)
run("ball", eye=(-230.0, -80.0, 240.0), target=(0.0, 0.0, 200.0), ball=(0.0, 0.0, 200.0), cars=[],
    ev=None, ages=[0.05, 0.1], spectate=-1)
# ball from 4 orientations
if not args.only or "ball_spin" in args.only.split(","):
    shots = []
    for fwd, up in (((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (1, 0, 0)), ((0, 0, 1), (0, 1, 0)), ((0.7, 0.7, 0), (0, 0, 1))):
        j = {"gamemode": "soccar", "ball_phys": {"pos": [0, 0, 200], "vel": [0, 0, 0], "ang_vel": [0, 0, 0],
             "forward": list(fwd), "up": list(up)}, "cars": [], "boost_pad_states": [True] * 34}
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            st.read_from_json(j); st.read_from_json(j); st.recv_time = time.time(); st.recv_interval = 1 / 30
        state_manager.pose_cam = ((-230.0, -80.0, 240.0), (0.0, 0.0, 200.0)); r.spectate_idx = -1
        for _ in range(3): frame()
        shots.append(frame().crop((W // 2 - H // 3, H // 2 - H // 3, W // 2 + H // 3, H // 2 + H // 3)))
    sheet = Image.new("RGB", (shots[0].width * 4, shots[0].height))
    for i, im in enumerate(shots): sheet.paste(im, (i * shots[0].width, 0))
    sheet.save(os.path.join(args.out, "ball_spin.png")); print("saved ball_spin")
# ball above the crossbar / beside the post, past the line (must NOT turn black)
run("ball_nogoal", eye=(-420.0, 4700.0, 900.0), target=(0.0, 5200.0, 760.0), ball=(0.0, 5200.0, 760.0), cars=[],
    ev=None, ages=[0.05, 0.1], spectate=-1)
# ball half across the goal line (black part + white seam)
run("ball_line", eye=(-420.0, 4700.0, 220.0), target=(0.0, 5124.0, 120.0), ball=(0.0, 5124.0, 120.0), cars=[],
    ev=None, ages=[0.05, 0.1, 0.15, 0.2], spectate=-1)


# boost pads: pickup -> dark -> white ghost fading in -> respawn flash
def pads_seq():
    cam = ((-900.0, -700.0, 380.0), (-1792.0, -4184.0 + 3000, 0.0))
    state_manager.pose_cam = ((-2750.0, -3700.0, 260.0), (-3072.0, -4096.0, 60.0))
    r.spectate_idx = -1
    def pads(active):
        j = {"gamemode": "soccar", "ball_phys": {"pos": [0, 0, 93], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]},
             "cars": [], "boost_pad_states": [active if i in (1, 3) else True for i in range(34)]}
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            st.read_from_json(j); st.recv_time = time.time(); st.recv_interval = 1 / 30
    pads(True)
    for _ in range(5): frame()
    pads(False)
    t0 = time.time(); shots = []
    for a in [0.1, 0.6, 3.5, 7.0, 9.8]:
        while time.time() - t0 < a: frame()
        shots.append(frame())
    pads(True)
    t1 = time.time()
    while time.time() - t1 < 0.08: frame()
    shots.append(frame())
    sheet = Image.new("RGB", (W * 2, H * 3))
    for i, im in enumerate(shots):
        sheet.paste(im, ((i % 2) * W, (i // 2) * H))
    sheet.save(os.path.join(args.out, "pads.png")); print("saved pads")


if not args.only or "pads" in args.only.split(","):
    pads_seq()

r"""readme_shots.py -- stills from a recorded game, rendered headless at the High preset (the README screenshots).

  python tools/readme_shots.py <recording> --map paris [--every 0.5 | --at 3.0,27.0] [--out docs] [--size 1920x1080]

<recording> is either a clip replay (clips/.<stamp>.json.gz: run the vis with RSV_CLIP_KEEP_REPLAY=1 and press C, the
file is kept next to the mp4) or a raw feed log (.jsonl.gz, one [time, packet-json] per line). The game is replayed
frame by frame on a virtual clock (like the clip renderer), with the recording's camera choices when it has them,
else the auto camera (follows whoever is closest to the ball). --every saves a frame every N seconds (to pick moments
from), --at saves the frames at those times (seconds from the start). Files: <out>/<map>_<time>.jpg.
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("--map", default="valley", help="valley, temple, paris or space")
ap.add_argument("--every", type=float, default=0.0)
ap.add_argument("--at", default="")
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "docs"))
ap.add_argument("--size", default="1920x1080")
args = ap.parse_args()
if not args.every and not args.at:
    ap.error("give --every <s> or --at <t1,t2,...>")

os.environ.setdefault("RSV_MSAA", "8")
os.environ["RSV_CLIP_SIZE"] = args.size
os.environ["RSV_SOUND"] = "0"
sys.path.insert(0, SRC)
import offline_render as orr                     # installs the virtual clock before the renderer loads
from PIL import Image
import state_manager

with gzip.open(args.recording, "rt", encoding="utf-8") as f:
    head = f.read(1)
if head == "{":                                  # clip replay
    src = orr.load_replay(args.recording)
else:                                            # raw feed log
    with gzip.open(args.recording, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    src = {"packets": [(float(t), json.loads(d)) for t, d in rows], "timeline": [], "pov": None, "t_start": None}
os.makedirs(args.out, exist_ok=True)
R = orr.OfflineRenderer()
r = R.r
r.config.apply_preset("high")
r.set_map(args.map, save=False)
sched = list(zip(state_manager.playout_times([p[0] for p in src["packets"]]), [p[1] for p in src["packets"]]))
R.reset_scene()
if not src["timeline"]:
    r.spectate_idx = 0
    r._auto_cam_key = True
at = [float(x) for x in args.at.split(",") if x.strip()]
fps = 60.0
t0, t_end = sched[0][0], sched[-1][0]
orr.VCLOCK.t = t0
R._feed(*sched[0])
pi, ti, f, nxt = 1, 0, 0, args.every
r.last_render_time = t0 - 1 / fps
while True:
    T = t0 + f / fps
    if T > t_end:
        break
    orr.VCLOCK.t = T
    while pi < len(sched) and sched[pi][0] <= T:
        R._feed(*sched[pi])
        pi += 1
    ti = R._camera(src, T, ti)
    r.paint(R.W, R.H, R.fbo, clip_capture=False)
    rel = T - t0
    if (args.every and rel >= nxt) or any(abs(rel - a) < 0.5 / fps for a in at):
        R.ctx.finish()
        im = Image.frombytes("RGB", (R.W, R.H), R.fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
        im.save(os.path.join(args.out, "%s_%06.2f.jpg" % (args.map, rel)), quality=93)
        if args.every:
            nxt += args.every
    f += 1
print("done:", args.map, f, "frames")

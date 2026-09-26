r"""gpu_contention_bench.py -- how much does the vis slow down GPU training?

Runs a training-like CUDA workload (the policy MLP: 379 -> 384x3 -> 132, forward + backward + Adam,
big minibatches) and measures iterations/s while the vis renders headless in a subprocess at a given
fps / resolution. Conditions are INTERLEAVED over several rounds so drift (thermals, other apps)
cancels out. Reports each condition's throughput relative to "no vis".

  python tools/gpu_contention_bench.py [--rounds 4] [--secs 8] [--size 3200x2000] [--fps 60,120]

Note: headless rendering skips the window present/compositor copy, so the real window costs a bit
more than measured here; treat results as a close lower bound.
"""
import argparse
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--size", default="3200x2000")
ap.add_argument("--fps", default="60,120")
ap.add_argument("--batch", type=int, default=32768)
ap.add_argument("--vis", default="", help="extra vis variants: name=ENV1:VAL1;ENV2:VAL2@fps, comma separated")
args = ap.parse_args()

dev = "cuda"
net = torch.nn.Sequential(torch.nn.Linear(379, 384), torch.nn.LeakyReLU(), torch.nn.Linear(384, 384),
                          torch.nn.LeakyReLU(), torch.nn.Linear(384, 384), torch.nn.LeakyReLU(),
                          torch.nn.Linear(384, 132)).to(dev)
opt = torch.optim.Adam(net.parameters(), 3e-4)
x = torch.randn(args.batch, 379, device=dev)
y = torch.randint(0, 132, (args.batch,), device=dev)


def train_for(secs):
    torch.cuda.synchronize()
    t0 = time.perf_counter(); n = 0
    while True:
        for _ in range(10):
            loss = torch.nn.functional.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n += 1
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        if el >= secs:
            return n / el


def start_vis(fps, extra_env=None):
    out = os.path.join(os.environ.get("TEMP", "."), "rsv_bench_out")
    env = dict(os.environ, RSV_SOUND="0")
    env.update(extra_env or {})
    return subprocess.Popen([sys.executable, os.path.join(HERE, "headless_test.py"), "--out", out, "--no-shots",
                             "--seconds", str(args.secs + 6), "--size", args.size, "--fps-cap", str(fps)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


train_for(3.0)                                    # warm-up
conds = ["none"] + ["vis@%s" % f for f in args.fps.split(",") if f]
variants = {}
for spec in [v for v in args.vis.split(",") if v]:
    name, rest = spec.split("=", 1)
    envs, fps = rest.rsplit("@", 1)
    variants[name] = (float(fps), dict(kv.split(":", 1) for kv in envs.split(";") if kv))
    conds.append(name)
res = {c: [] for c in conds}
for r in range(args.rounds):
    for c in conds:
        p = None
        if c in variants:
            p = start_vis(variants[c][0], variants[c][1])
            time.sleep(4.0)
        elif c != "none":
            p = start_vis(float(c.split("@")[1]))
            time.sleep(4.0)                       # let it finish loading and reach steady state
        res[c].append(train_for(args.secs))
        if p is not None:
            p.kill(); p.wait()
        print("round %d %-8s %.1f it/s" % (r, c, res[c][-1]), flush=True)

base = sum(res["none"]) / len(res["none"])
print("\nresult (mean over rounds, relative to no vis):")
for c in conds:
    m = sum(res[c]) / len(res[c])
    print("  %-8s %.1f it/s   %+.1f%%" % (c, m, 100.0 * (m / base - 1.0)))

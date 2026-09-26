"""engine_synth.py -- Rocket League's Octane engine sound, rebuilt from its Wwise bank.

SFX_Motor_OctaneMK2 (dumped with wwiser, 2026-09) is not "a loop pitched by speed". It is:

  two blend containers (both always playing), each with two tracks crossfaded by a 0..1 LOAD
  parameter (on-throttle track above 0.5, off-throttle track below), and inside each track 2-3 raw
  loops crossfaded by engine RPM (0..10000), every loop with its OWN RPM->pitch curve (cents);
  on top, a throttle parameter (-1..1) pulls the whole engine down ~15 dB when coasting and a steer
  parameter lowers the pitch by up to 188 cents.

This module reproduces that graph sample-accurately (per-source phase accumulators, per-sample pitch
and gain ramps, so RPM changes glide instead of stepping) and drives it with a small RPM model:
on the ground the RPM follows wheel speed, throttle revs it (also at standstill -- kickoff revving --
and in the air, where RL's wheels spin up with throttle), releasing lets it fall back.

Used live by audio.py (streamed to a pygame channel in small blocks) and offline by offline_render.py.
Curve values are copied from the bank; volume curves use Wwise's "dB-scaled" storage, where the
stored value v means linear gain 1 + v (v = -1 is silence).
"""
import math
import os
import wave

import numpy as np

SR = 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "sounds", "engine_src")

# ---- curves from the bank ------------------------------------------------------------------ #
P_MAIN = ((0.0, 728.4, 3316.6, 10000.0), (-4800.0, -2205.0, -283.0, 909.0))    # cents vs RPM
P_IDLE = ((0.0, 1000.0, 5013.8, 10000.0), (0.0, 0.0, 1938.0, 1922.0))
P_B = ((0.0, 728.4, 3316.6, 10000.0), (-2000.0, -2000.0, 575.0, 1767.0))
X_LO = ((0.0, 0.0005, 3000.0), (1.0, 1.0, 0.0))                                # blend vs RPM
X_MID = ((0.0005, 3000.0, 6000.0), (0.0, 1.0, 0.0))
X_HI = ((3000.0, 6000.0, 10000.0), (0.0, 1.0, 1.0))
Y_LO = ((0.0, 0.0005, 4500.0), (1.0, 1.0, 0.0))
Y_HI = ((0.0005, 4500.0, 10000.0), (0.0, 1.0, 1.0))
# (source wem id, makeup dB, pitch curve, blend curve, track: 1 = on-throttle, 0 = off-throttle)
LAYERS = (
    (1072222933, -9.0, P_IDLE, X_LO, 1), (971492431, 0.0, P_MAIN, X_MID, 1), (924662072, 0.0, P_MAIN, X_HI, 1),
    (1072222933, -9.0, P_IDLE, X_LO, 0), (405608813, 0.0, P_MAIN, X_MID, 0), (95566843, 0.0, P_MAIN, X_HI, 0),
    (943180388, 2.0, P_B, Y_LO, 1), (632240338, 0.0, P_MAIN, Y_HI, 1),
    (790896015, 2.0, P_B, Y_LO, 0), (286033530, 0.0, P_MAIN, Y_HI, 0),
)
# LOAD (0..1) track crossfade, "dB-scaled" storage -> gain = 1 + v. The bank's on-track overshoots to
# +17 dB just above 0.5 (a throttle-blip surge); with bot throttle flicker the load kept crossing it and
# the engine "bubbled", so the crossfade is monotonic here (no overshoot).
LOAD_ON = ((0.0, 0.485, 0.51, 1.0), (-0.9999, -0.9999, 0.0, 0.0))
LOAD_OFF = ((0.0, 0.49, 0.515, 1.0), (0.0, -1.0, -0.9999, -0.9999))
THROTTLE_VOL = ((-1.0, 0.1, 0.31424, 1.0), (-0.9553, -0.8221, 0.00395, 0.0))    # -1..1 -> gain = 1 + v
STEER_PITCH = ((-1.0, 0.0, 1.0), (-188.0, 0.0, -188.0))                         # cents


def _curve(c, x):
    return float(np.interp(x, c[0], c[1]))


def _load_wav(path):
    with wave.open(path, "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        a = np.frombuffer(w.readframes(n), "<i2").astype("f4").reshape(-1, ch) * (1.0 / 32768.0)
    if ch == 1:
        a = np.repeat(a, 2, 1)
    return np.ascontiguousarray(a[:, :2])


class EngineModel:
    """Game inputs -> Wwise parameters (rpm, load, throttle, steer), with RTPC-style smoothing."""
    IDLE = 900.0

    THROTTLE_HOLD = 0.25          # s: bots flicker throttle every step; hold "on" through short gaps

    def __init__(self):
        self.rpm = self.IDLE
        self.load = 0.0
        self.thr = 0.0
        self.steer = 0.0
        self._prev_vf = None
        self._on_hold = 0.0

    def update(self, dt, on_ground, v_fwd, throttle=None, steer=None, boosting=False, speed=None):
        dt = min(max(dt, 0.0), 0.1)
        if speed is None:
            speed = abs(v_fwd)
        if throttle is None:
            # Inferred when the sender doesn't stream controls: boost = full throttle; on the ground a
            # forward acceleration well above rolling drag means the throttle is held.
            thr = 0.0
            if boosting:
                thr = 1.0
            elif on_ground and self._prev_vf is not None and dt > 1e-4:
                acc = (v_fwd - self._prev_vf) / dt * (1.0 if v_fwd >= 0 else -1.0)
                thr = 1.0 if acc > 250.0 else (-1.0 if acc < -1200.0 else 0.0)
            throttle = thr * (1.0 if v_fwd >= -50.0 else -1.0)
        self._prev_vf = v_fwd
        if boosting:
            throttle = 1.0
        # throttle relative to the direction of travel: accelerating (either way) = +, braking = -
        rel = throttle * (1.0 if v_fwd >= 0 else -1.0) if (on_ground and abs(v_fwd) > 150.0) else abs(throttle)
        if rel > 0.2:
            self._on_hold = self.THROTTLE_HOLD
        else:
            self._on_hold = max(0.0, self._on_hold - dt)
        on = rel > 0.2 or self._on_hold > 0.0
        if on and rel <= 0.2:
            rel = 1.0                                          # held "on" through a flicker gap
        if on_ground:
            target = self.IDLE + 7600.0 * min(1.0, abs(v_fwd) / 2300.0)
            if on:
                target = max(target, self.IDLE + 2600.0 * abs(throttle if abs(throttle) > 0.2 else 1.0))
            tau = 0.12 if target > self.rpm else 0.35
        else:
            # In the air the engine follows the car's SPEED (like the game) -- no sudden drop when the
            # throttle is released mid-air -- and stays at least as loud as that speed implies.
            target = self.IDLE + 7600.0 * min(1.0, speed / 2300.0)
            rel = max(rel, min(1.0, speed / 2300.0) * 1.4)
            tau = 0.5
        self.rpm += (target - self.rpm) * (1.0 - math.exp(-dt / tau))
        step = dt / 0.25                                                # LOAD slew: 0 <-> 1 in 0.25 s
        self.load = min(self.load + step, 1.0) if on else max(self.load - step, 0.0)
        self.thr += (rel - self.thr) * (1.0 - math.exp(-dt / 0.08))
        s = 0.0 if steer is None else max(-1.0, min(1.0, float(steer)))
        self.steer += (s - self.steer) * (1.0 - math.exp(-dt / 0.1))
        return self.params()

    def params(self):
        return {"rpm": self.rpm, "load": self.load, "thr": self.thr, "steer": self.steer}


class EngineSynth:
    """Continuous-phase renderer of the layer graph. render(n, params, gl, gr) -> (n, 2) float32,
    ramping every parameter linearly from the previous call's values (no zipper noise)."""

    def __init__(self, sources=None):
        if sources is None:
            sources = {}
            for wid in {l[0] for l in LAYERS}:
                p = os.path.join(SRC_DIR, "{}.wav".format(wid))
                if os.path.isfile(p):
                    sources[wid] = _load_wav(p)
        self.sources = sources
        self.ok = all(l[0] in sources for l in LAYERS)
        self.phase = [0.0] * len(LAYERS)
        self._last = None

    def _layer_values(self, p):
        """-> per layer (rate, gain) for one parameter set."""
        steer_c = _curve(STEER_PITCH, p["steer"])
        top = max(0.0, 1.0 + _curve(THROTTLE_VOL, p["thr"]))
        g_on = min(2.0, max(0.0, 1.0 + _curve(LOAD_ON, p["load"])))
        g_off = max(0.0, 1.0 + _curve(LOAD_OFF, p["load"]))
        out = []
        for wid, mk, pc, bc, track in LAYERS:
            cents = _curve(pc, p["rpm"]) + steer_c
            g = _curve(bc, p["rpm"]) * (g_on if track else g_off) * (10.0 ** (mk / 20.0)) * top
            out.append((2.0 ** (cents / 1200.0), g))
        return out

    def render(self, n, params, gl=1.0, gr=1.0):
        out = np.zeros((n, 2), "f4")
        if not self.ok or n <= 0:
            return out
        cur = self._layer_values(params)
        last = self._last or [(r, g, gl, gr) for r, g in cur]
        ramp = (np.arange(1, n + 1, dtype="f8") / n)
        el = np.float32(last[0][2]) + (np.float32(gl) - np.float32(last[0][2])) * ramp.astype("f4")
        er = np.float32(last[0][3]) + (np.float32(gr) - np.float32(last[0][3])) * ramp.astype("f4")
        for i, (wid, _mk, _pc, _bc, _tr) in enumerate(LAYERS):
            r0, g0 = last[i][0], last[i][1]
            r1, g1 = cur[i]
            src = self.sources[wid]
            L = len(src)
            rates = r0 + (r1 - r0) * ramp
            pos = self.phase[i] + np.cumsum(rates)
            self.phase[i] = float(pos[-1] % L)
            if g0 <= 1e-4 and g1 <= 1e-4:
                continue                                   # silent: phase still advances
            i0 = np.floor(pos).astype(np.int64)
            fr = (pos - i0).astype("f4")[:, None]
            a = src[i0 % L]
            b = src[(i0 + 1) % L]
            s = a + (b - a) * fr
            g = (g0 + (g1 - g0) * ramp).astype("f4")[:, None]
            out += s * g
        out[:, 0] *= el
        out[:, 1] *= er
        self._last = [(r, g, gl, gr) for r, g in cur]
        return out

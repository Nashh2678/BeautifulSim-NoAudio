"""Game audio, spatialised against the camera. This repo ships NO sound files: drop your own into
data/sounds/ with the file names listed in data/sounds/manifest.json (see data/sounds/README.md). Any
subset works -- an event with no installed file just stays silent, and with no files at all the mixer
never starts (every method is a silent no-op).

Design:
  * pygame.mixer only (no pygame window). Everything is a one-shot on a mixer channel except the boost
    loop (one looping channel per boosting car) and the spectated car's engine + boost loop, which are
    synthesised (engine_synth.py, from the loops in data/sounds/engine_src/) and streamed to one
    reserved channel in short blocks by a feeder thread.
  * Boost: start layer, loop entering 300 ms later (pitched up with car speed), on release the loop
    fades 100 ms / the start layer 300 ms, plus a tail.
  * Spatialisation is done by us: constant-power stereo pan from the camera's right vector + a
    distance roll-off. The spectated car uses the "_local" variants, everything else "_other".
  * Called only from the render (GUI) thread. If the mixer can't open (no audio device, no sounds)
    every method is a silent no-op, so the visualizer never breaks because of audio.

Env: RSV_SOUND=0 disables audio entirely, RSV_VOLUME=<0..1> sets the initial master volume.
Keys (main.py): M = mute toggle, [ / ] = volume down/up. Mute + volume persist in rsv_settings.json.
"""
import json
import threading
import math
import os
import random
import time

from const import DATA_DIR_PATH

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_settings.json")
SOUND_DIR = os.path.join(DATA_DIR_PATH, "sounds")


def _load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_settings(d):
    try:
        cur = _load_settings()
        cur.update(d)
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f)
        os.replace(tmp, SETTINGS_PATH)
    except OSError:
        pass


class Audio:
    NUM_CHANNELS = 64
    LOOP_CHANNELS = 8           # boost loops (one per car)
    STREAM_CH = LOOP_CHANNELS   # reserved: synthesized engine + local boost loop
    ENGINE_GAIN = 0.30
    BOOST_LOOP_DELAY = 0.30     # Alpha: the loop starts 300 ms after the start layer
    BOOST_PITCH_STEPS = (0.0, 150.0, 300.0, 450.0, 611.0)   # cents; other cars pick one at loop start
    REF_DIST = 2600.0        # uu: distance at which a 3D sound is at ~half gain
    MIN_GAIN = 0.18          # far sounds never fully vanish (RL's arena is small and reverberant)

    # sound name prefix -> mixer category (Settings > Audio sliders); anything else = master only
    CATEGORIES = (("ball_", "ball"), ("bump", "demo"), ("demo", "demo"), ("boost", "boost"),
                  ("pad_pickup", "boost"), ("engine", "engine"), ("supersonic", "engine"),
                  ("flipreset", "reset"), ("jump", "flips"), ("doublejump", "flips"), ("dodge", "flips"),
                  ("land", "flips"))

    def __init__(self):
        self.ok = False
        self.cat = {"ball": 1.0, "demo": 1.0, "boost": 1.0, "engine": 1.0, "flips": 1.0, "reset": 1.0}
        self._cat_cache = {}
        self.sounds = {}                  # name -> [pygame.mixer.Sound]
        self._last_pick = {}
        self._boost = {}                  # car key -> dict(ch, fading)
        settings = _load_settings()
        self.muted = bool(settings.get("muted", False))
        self.volume = float(os.environ.get("RSV_VOLUME", settings.get("volume", 0.7)))
        self.listener = None              # (pos, right) tuples of floats
        self.active = True                # False while the window is unfocused: everything silent
        self._stream = None               # EngineStream (spectated car's engine + boost loop)
        self.engine_rates = []
        self._loop_free = []
        self._engine_first_ch = 0
        if os.environ.get("RSV_SOUND", "1") == "0":
            print("[audio] disabled by RSV_SOUND=0")
            return
        # pygame import + decoding ~230 oggs took most of the vis startup: load on a background
        # thread; until it finishes every call is a silent no-op (self.ok stays False).
        self._loader = threading.Thread(target=self._load, name="rsv-audio-load", daemon=True)
        self._loader.start()

    def _load(self):
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            self._pg = pygame
            mpath = os.path.join(SOUND_DIR, "manifest.json")
            if not os.path.isfile(mpath):
                print("[audio] no {} (no sounds installed): running silent".format(mpath))
                return
            manifest = json.load(open(mpath, "r"))
            sr = int(manifest.get("sample_rate", 48000))
            pygame.mixer.pre_init(sr, -16, 2, 512)      # 512-sample buffer ~= 11 ms latency
            pygame.mixer.init()
            pygame.mixer.set_num_channels(self.NUM_CHANNELS)
            # Channels 0..7 are reserved for boost loops (one per car) so one-shots can never steal
            # a loop's channel and silently cut it (find_channel(True) skips reserved channels).
            pygame.mixer.set_reserved(self.LOOP_CHANNELS + 1)
            self._loop_free = [pygame.mixer.Channel(i) for i in range(self.LOOP_CHANNELS)]
            # Only the files actually present are loaded: the manifest lists every event's files, and any
            # subset of them can be dropped into data/sounds/ (an event with no file just stays silent).
            sounds = {}
            for name, files in manifest["sounds"].items():
                if name.startswith("engine_"):
                    continue                                # old pre-pitched engine sets: unused now
                got = [pygame.mixer.Sound(os.path.join(SOUND_DIR, fn)) for fn in files
                       if os.path.isfile(os.path.join(SOUND_DIR, fn))]
                if got:
                    sounds[name] = got
            self.sounds = sounds
            import numpy as np
            self._boost_variants = []
            if "boost_loop" in sounds:
                # boost loop pre-pitched for the other cars (the local one is pitched continuously)
                loop = pygame.sndarray.array(sounds["boost_loop"][0]).astype("f4")
                for cents in self.BOOST_PITCH_STEPS:
                    r = 2.0 ** (cents / 1200.0)
                    t = np.arange(int(len(loop) / r)) * r
                    y = np.stack([np.interp(t, np.arange(len(loop)), loop[:, c], period=len(loop)) for c in range(2)], 1)
                    self._boost_variants.append(pygame.sndarray.make_sound(np.ascontiguousarray(y.astype("<i2"))))
            else:
                loop = np.zeros((2, 2), "f4")                  # no boost loop file: silent loop
            try:
                import engine_synth
                synth = engine_synth.EngineSynth()          # .ok False without data/sounds/engine_src/*.wav
                engine_ok = synth.ok
            except Exception:
                engine_synth, engine_ok = None, False
            if engine_synth is not None and (engine_ok or "boost_loop" in sounds):
                self._stream = EngineStream(pygame, pygame.mixer.Channel(self.STREAM_CH), sr,
                                            synth if engine_ok else _SilentSynth(), engine_synth.EngineModel(),
                                            np.ascontiguousarray(loop / 32768.0))
            if not sounds and self._stream is None:
                print("[audio] no sound files in {} (only the manifest): running silent".format(SOUND_DIR))
                return
            self.ok = True                                  # publish last: calls start working now
            print("[audio] {} sounds loaded (engine {}), muted={} volume={:.2f}".format(
                len(self.sounds), "on" if engine_ok else "off", self.muted, self.volume))
        except Exception as e:                             # no device / missing files -> silent
            print("[audio] disabled: {!r}".format(e))

    def wait_loaded(self, timeout=30.0):
        """Tests: block until the background load finished (no-op when sound is disabled)."""
        t = getattr(self, "_loader", None)
        if t is not None:
            t.join(timeout)

    # ---- settings ---------------------------------------------------------------------------- #

    def toggle_mute(self):
        self.muted = not self.muted
        _save_settings({"muted": self.muted})
        if self.muted:
            self.stop_all()

    def change_volume(self, delta):
        self.set_volume(self.volume + delta)

    def set_volume(self, v):
        self.volume = min(1.0, max(0.0, float(v)))
        _save_settings({"volume": round(self.volume, 3)})

    def set_active(self, active):
        """Window focus. Inactive = stop everything and refuse to (re)start anything: Qt still repaints
        the widget once on focus loss, which used to restart the boost loop with nobody to stop it."""
        if active == self.active:
            return
        self.active = active
        if not active:
            self.stop_all()

    # ---- spatial ----------------------------------------------------------------------------- #

    def set_listener(self, pos, right):
        self.listener = ((float(pos[0]), float(pos[1]), float(pos[2])),
                         (float(right[0]), float(right[1]), float(right[2])))

    def _gains(self, pos, gain, local):
        g = gain * self.volume
        if local or pos is None or self.listener is None:
            # 2D ("my car") sound: centred, slight pan toward the source so it still reads spatially.
            if pos is None or self.listener is None:
                return g, g
            pan = 0.35 * self._pan(pos)
        else:
            lp, _ = self.listener
            d = math.sqrt((pos[0] - lp[0]) ** 2 + (pos[1] - lp[1]) ** 2 + (pos[2] - lp[2]) ** 2)
            att = 1.0 / (1.0 + (d / self.REF_DIST) ** 1.6)
            g *= max(self.MIN_GAIN, att)
            pan = self._pan(pos)
        a = (pan + 1.0) * (math.pi / 4.0)                  # constant-power pan
        return g * math.cos(a) * 1.41421356, g * math.sin(a) * 1.41421356

    def _pan(self, pos):
        (lx, ly, lz), (rx, ry, rz) = self.listener
        dx, dy, dz = pos[0] - lx, pos[1] - ly, pos[2] - lz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-3:
            return 0.0
        return max(-1.0, min(1.0, (dx * rx + dy * ry + dz * rz) / d))

    def distance(self, pos):
        if self.listener is None or pos is None:
            return 0.0
        lp = self.listener[0]
        return math.sqrt((pos[0] - lp[0]) ** 2 + (pos[1] - lp[1]) ** 2 + (pos[2] - lp[2]) ** 2)

    # ---- one-shots --------------------------------------------------------------------------- #

    def _pick(self, name):
        vs = self.sounds.get(name)
        if not vs:
            return None
        if len(vs) == 1:
            return vs[0]
        last = self._last_pick.get(name, -1)
        k = random.randrange(len(vs) - 1)
        if k >= last:
            k += 1                                          # never the same variant twice in a row
        self._last_pick[name] = k
        return vs[k]

    def _cat_gain(self, name):
        c = self._cat_cache.get(name)
        if c is None:
            c = next((cat for pre, cat in self.CATEGORIES if name.startswith(pre)), "")
            self._cat_cache[name] = c
        return self.cat.get(c, 1.0)

    def play(self, name, pos=None, gain=1.0, local=False):
        if not self.ok or self.muted or not self.active or gain <= 0.0:
            return None
        gain *= self._cat_gain(name)
        if gain <= 0.0:
            return None
        snd = self._pick(name)
        if snd is None:
            return None
        ch = self._pg.mixer.find_channel(True)             # steal the oldest if all are busy
        if ch is None:
            return None
        l, r = self._gains(pos, gain, local)
        ch.play(snd)
        ch.set_volume(min(1.0, l), min(1.0, r))
        return ch

    # ---- boost loops ------------------------------------------------------------------------- #

    def update_boost(self, boosting):
        """boosting: {car_key: (pos, local[, speed])} for every car boosting THIS frame. Alpha Boost:
        start layer at once, loop after 300 ms (the local car's loop is in the synth stream, pitched
        continuously with speed), release = loop fade 100 ms + start-layer fade 300 ms + tail."""
        if not self.ok:
            return
        if self.muted or not self.active:
            if self._boost:
                self.stop_all()
            return
        now = time.time()
        for key, val in boosting.items():
            pos, local = val[0], val[1]
            speed = val[2] if len(val) > 2 else 0.0
            st = self._boost.get(key)
            if st is None:
                st = self._boost[key] = {"t0": now, "ch": None,
                                         "start_ch": self.play("boost_start", pos, 0.9, local)}
            if (st["ch"] is None and not local and now - st["t0"] >= self.BOOST_LOOP_DELAY and self._loop_free
                    and self._boost_variants):
                k = min(range(len(self.BOOST_PITCH_STEPS)),
                        key=lambda i: abs(self.BOOST_PITCH_STEPS[i] - 611.0 * min(1.0, speed / 2300.0)))
                st["ch"] = self._loop_free.pop()
                st["ch"].play(self._boost_variants[k], loops=-1, fade_ms=40)
            l, r = self._gains(pos, (0.8 if local else 0.7) * self.cat["boost"], local)
            if st["ch"] is not None:
                st["ch"].set_volume(min(1.0, l), min(1.0, r))
            if local and self._stream is not None:
                self._stream.set_boost(now - st["t0"] >= self.BOOST_LOOP_DELAY, min(1.0, l), min(1.0, r), speed)
            st["pos"], st["local"] = pos, local
        for key in [k for k in self._boost if k not in boosting]:
            st = self._boost.pop(key)
            if st.get("ch") is not None:
                st["ch"].fadeout(100)
                self._loop_free.append(st["ch"])
            if st.get("start_ch") is not None:
                try:
                    st["start_ch"].fadeout(300)
                except Exception:
                    pass
            if st.get("local") and self._stream is not None:
                self._stream.set_boost(False, 0.0, 0.0, 0.0)
            self.play("boost_stop", st.get("pos"), 0.8, st.get("local", False))

    # ---- engine ------------------------------------------------------------------------------ #

    def update_engine(self, car):
        """car = dict(pos, v_fwd, on_ground, throttle, steer, boosting, dt) for the spectated car, or
        None -> silence. Feeds the RPM model; the stream thread renders the audio."""
        if not self.ok or self._stream is None:
            return
        if car is None or self.muted or not self.active:
            self._stream.set_engine(None, 0.0, 0.0)
            return
        p = self._stream.model.update(car.get("dt", 1 / 60.0), car["on_ground"], car["v_fwd"],
                                      car.get("throttle"), car.get("steer"), car.get("boosting", False),
                                      car.get("speed"))
        l, r = self._gains(car["pos"], self.ENGINE_GAIN * self.cat["engine"], True)
        self._stream.set_engine(p, l, r)

    def stop_all(self):
        if not self.ok:
            return
        if self._stream is not None:
            self._stream.stop()
        self._pg.mixer.stop()
        for st in self._boost.values():
            if st.get("ch") is not None:
                self._loop_free.append(st["ch"])
        self._boost.clear()


class _SilentSynth:
    """Stand-in when the engine source loops aren't installed (the boost loop still streams)."""
    ok = False

    def render(self, n, params, gl, gr):
        import numpy as np
        return np.zeros((n, 2), "f4")


class EngineStream:
    """Streams the synthesized engine (+ the local car's boost loop) to one mixer channel in short
    blocks from a feeder thread, so pitch and level glide continuously with RPM and speed."""
    BLOCK = 2400                     # 50 ms at 48 kHz; one block queued ahead (margin vs GIL stalls)

    def __init__(self, pygame, channel, sr, synth, model, boost_loop):
        import numpy as np
        self.np = np
        self.pg = pygame
        self.ch = channel
        self.sr = sr
        self.synth = synth
        self.model = model
        self.boost_loop = boost_loop
        self._lock = threading.Lock()
        self._eng = None             # (params, l, r)
        self._boost = (False, 0.0, 0.0, 0.0)
        self._b_phase = 0.0
        self._b_last = (0.0, 0.0, 1.0)
        self._running = False
        self._t = threading.Thread(target=self._loop, name="rsv-engine-stream", daemon=True)
        self._t.start()

    def set_engine(self, params, l, r):
        with self._lock:
            self._eng = None if params is None else (dict(params), l, r)
            if params is not None:
                self._running = True

    def set_boost(self, on, l, r, speed):
        with self._lock:
            self._boost = (bool(on), l, r, speed)

    def stop(self):
        with self._lock:
            self._running = False
            self._eng = None
            self._boost = (False, 0.0, 0.0, 0.0)
        try:
            self.ch.stop()
        except Exception:
            pass

    def render_block(self, n, eng, boost):
        """Pure function of the given state (used by the live thread AND offline_render)."""
        np = self.np
        b_on, bl, br, bspeed = boost
        if eng is None:
            out = self.synth.render(n, self.model.params(), 0.0, 0.0)
        else:
            out = self.synth.render(n, eng[0], eng[1], eng[2])
        g0l, g0r, r0 = self._b_last
        g1l, g1r = (bl, br) if b_on else (0.0, 0.0)
        r1 = 2.0 ** (611.0 * min(1.0, bspeed / 2300.0) / 1200.0) if b_on else r0
        if max(g0l, g0r, g1l, g1r) > 1e-4:
            ramp = np.arange(1, n + 1, dtype="f8") / n
            if not b_on:                                   # release over ~100 ms, not one block
                k = min(1.0, n / (0.1 * self.sr))
                g1l, g1r = g0l * (1.0 - k), g0r * (1.0 - k)
            src = self.boost_loop
            pos = self._b_phase + np.cumsum(r0 + (r1 - r0) * ramp)
            self._b_phase = float(pos[-1] % len(src))
            i0 = np.floor(pos).astype(np.int64)
            fr = (pos - i0).astype("f4")[:, None]
            a = src[i0 % len(src)]
            s = a + (src[(i0 + 1) % len(src)] - a) * fr
            out[:, 0] += s[:, 0] * (g0l + (g1l - g0l) * ramp).astype("f4")
            out[:, 1] += s[:, 1] * (g0r + (g1r - g0r) * ramp).astype("f4")
        self._b_last = (g1l, g1r, r1)
        return out

    def _loop(self):
        np = self.np
        while True:
            time.sleep(0.005)
            with self._lock:
                running = self._running
                eng = self._eng
                boost = self._boost
            if not running:
                continue
            try:
                if self.ch.get_queue() is None:
                    out = self.render_block(self.BLOCK, eng, boost)
                    snd = self.pg.sndarray.make_sound(
                        np.ascontiguousarray((np.clip(out, -1.0, 1.0) * 32767.0).astype("<i2")))
                    if self.ch.get_busy():
                        self.ch.queue(snd)
                    else:
                        self.ch.play(snd)
            except Exception as e:
                print("[audio] engine stream error: {!r}".format(e))
                time.sleep(0.5)

r"""offline_render.py -- render gameplay to mp4 WITH SOUND, headless on the discrete GPU, faster than
real time.

Instead of screen-recording the live window, this replays a recorded packet stream through the exact
same renderer (RSVRenderer) and event/sound logic on a VIRTUAL clock, in a standalone OpenGL context
(which lands on the RTX on this laptop, unlike the vis window). Frames are converted to yuv420p on the
GPU (shader pass), read back asynchronously (double-buffered PBO) and piped into ffmpeg's NVENC
encoder, so the CPU only runs the renderer's own Python. The sound is mixed offline from the very same
play()/boost/engine calls the live vis makes (same gains, panning, per-category sliders), then muxed
in as AAC.

Speed: one clip is split into time segments rendered by parallel worker processes (each replays a
short low-rate warm-up first so particles, trails and the camera are continuous across the cut); the
parent concatenates the video losslessly and mixes ONE soundtrack from all workers' sound events (loop
phase is global, so there is no click at the joins). Pop-reset batches give each worker whole clips.

Sources:
  --replay <file.json.gz>   the vis's recent-packet ring (GUI "Clip 12s" button / 'C' in the vis)
  --pop <dir|file.bin>      the trainer's GGLPOP1 pop-reset trajectories (GUI "Record Pop Resets")

  python src/offline_render.py --replay ring.json.gz --out clip.mp4 [--done clip_done.txt]
  python src/offline_render.py --pop <checkpoint>/pop_reset_clips [--keep-bin]

Knobs: RSV_CLIP_SIZE (1920x1080), RSV_CLIP_FPS (60), RSV_CLIP_WORKERS (2), RSV_CLIP_VOLUME (0.7 =
export level, independent of the vis's listening volume), RSV_FFMPEG (explicit ffmpeg.exe; default
bin/ffmpeg/ffmpeg.exe, else any ffmpeg with h264_nvenc, else libx264).
"""
import argparse
import glob
import gzip
import json
import math
import os
import queue
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# ---- virtual clock ------------------------------------------------------------------------------ #
# The renderer reads time.time()/time.monotonic() for interpolation, effect ages and event timing.
# This process only renders, so the clock is replaced wholesale BEFORE the renderer is imported and
# advanced one video frame at a time. (Real elapsed time here = time.perf_counter.)


class _VClock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


VCLOCK = _VClock()
time.time = VCLOCK
time.monotonic = VCLOCK

# numpy's OpenBLAS pre-commits a scratch buffer per CPU thread at import: ~480 MB of commit per
# process for nothing (the renderer only does tiny matrix math). Must be set BEFORE numpy loads;
# workers inherit it. (With training running the machine has only ~5 GB of commit free.)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["RSV_SOUND"] = "0"          # the live mixer is replaced by OfflineAudio below
os.environ["RSV_CLIP_DISABLE"] = "1"   # no frame-ring capture inside the offline renderer
os.environ.pop("RSV_CONTROL_PATH", None)  # never follow the GUI's auto-switch; the timeline decides

import numpy as np  # noqa: E402

SAMPLE_RATE = 48000
SOUND_DIR = os.path.join(ROOT, "data", "sounds")
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
ABOVE_NORMAL = 0x00008000 if os.name == "nt" else 0
WARMUP_S = 1.5        # replayed (not encoded) before each parallel segment
WARMUP_FPS = 20.0


def log(msg):
    print("[offline] " + msg, flush=True)


# ---- safety: a clip must NEVER be able to take training down ---------------------------------------
# 2026-09-25: a clip render pushed Windows' commit charge (RAM + page file, ONE pool shared by every
# process) over its limit while training ran near it -> ffmpeg 0xc000012d and the TRAINER's next
# allocation failed. Measured cost of a 12 s clip: ~1.8 GB + ~0.5 GB per worker. So:
#   * before rendering, only as many workers as fit while keeping MEM_SAFETY_GB of commit free;
#     none fit -> the clip is skipped with a message, never forced;
#   * the whole render (parent, workers, ffmpeg) runs in a Windows Job Object with a hard memory cap
#     (exceeding it fails the RENDER, not the machine) and kill-on-close (no orphaned workers);
#   * a watchdog kills the job if anything stalls; normal priority (training runs above normal);
#   * the RTX is only used when it has VRAM to spare; otherwise the render goes to the iGPU.
MEM_SAFETY_GB = float(os.environ.get("RSV_CLIP_MEM_SAFETY_GB", "2.5"))   # always left free for training
MEM_BASE_GB, MEM_WORKER_GB = 0.6, 0.8        # measured with BLAS pinned: 0.4 GB (1 worker), 1.6 (2), 3.2 (4)
VRAM_WORKER_MB, VRAM_SAFETY_MB = 500, 1500
_JOB = None


def _commit_gb():
    """-> (committed, limit) in GB for the whole system."""
    import ctypes
    from ctypes import wintypes

    class PERF(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]
    pi = PERF()
    pi.cb = ctypes.sizeof(PERF)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return 0.0, 1e9
    return pi.CommitTotal * pi.PageSize / 2 ** 30, pi.CommitLimit * pi.PageSize / 2 ** 30


def plan_workers(requested):
    """-> (workers, reason). 0 workers = do not render now."""
    if os.name != "nt":
        return requested, ""
    used, limit = _commit_gb()
    free = limit - used
    k = requested
    while k >= 1 and MEM_BASE_GB + MEM_WORKER_GB * k + MEM_SAFETY_GB > free:
        k -= 1
    if k < 1:
        return 0, ("not enough memory headroom: {:.1f} GB of commit free, a clip needs {:.1f} GB and "
                   "{:.0f} GB is always kept free for training".format(free, MEM_BASE_GB + MEM_WORKER_GB, MEM_SAFETY_GB))
    return k, "{:.1f} GB commit free".format(free)


def _rtx_ok(workers):
    """True if the NVIDIA GPU has VRAM to spare for `workers` GL contexts (else use the iGPU)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5, creationflags=CREATE_NO_WINDOW)
        free_mb = float(q.stdout.strip().splitlines()[0])
        return free_mb >= VRAM_SAFETY_MB + VRAM_WORKER_MB * workers
    except Exception:
        return False


def _enter_job(limit_gb):
    """Put this process (and every child it spawns) in a Job Object with a hard memory cap and
    kill-on-close. Returns the job handle (kept open for the process lifetime) or None."""
    global _JOB
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]
        # explicit 64-bit handle types: without them GetCurrentProcess()'s pseudo-handle (-1) is
        # truncated and AssignProcessToJobObject silently fails
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.Basic.LimitFlags = 0x00000200 | 0x00002000        # JOB_MEMORY | KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = int(limit_gb * 2 ** 30)
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return None
        _JOB = job
        return job
    except Exception:
        return None


def _kill_job():
    if _JOB is not None:
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32")
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.TerminateJobObject(_JOB, 1)
        except Exception:
            pass


# ---- ffmpeg ------------------------------------------------------------------------------------- #
_FFMPEG = None


def find_ffmpeg():
    """-> (path, video_codec_args). Prefers an ffmpeg with NVENC (encode on the RTX)."""
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG
    cands = [os.environ.get("RSV_FFMPEG"), os.path.join(ROOT, "bin", "ffmpeg", "ffmpeg.exe"),
             shutil.which("ffmpeg")]
    x264 = None
    for c in cands:
        if not c or not os.path.isfile(c):
            continue
        try:
            enc = subprocess.run([c, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                 timeout=10, creationflags=CREATE_NO_WINDOW).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "h264_nvenc" in enc:
            _FFMPEG = (c, ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"])
            return _FFMPEG
        if x264 is None and "libx264" in enc:
            x264 = (c, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    if x264 is None:
        raise RuntimeError("no usable ffmpeg found (set RSV_FFMPEG)")
    _FFMPEG = x264
    return _FFMPEG


def _run_ffmpeg(args):
    ff, _ = find_ffmpeg()
    return subprocess.run([ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args],
                          stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW).returncode


# ---- offline audio -------------------------------------------------------------------------------- #
import audio as rl_audio  # noqa: E402


def _decode(path):
    """Any audio file -> float32 (n, 2) at SAMPLE_RATE (PyAV)."""
    import av
    chunks = []
    with av.open(path) as c:
        res = av.AudioResampler(format="flt", layout="stereo", rate=SAMPLE_RATE)
        for fr in c.decode(c.streams.audio[0]):
            for o in res.resample(fr):
                chunks.append(o.to_ndarray().reshape(-1, 2))
        for o in res.resample(None):
            chunks.append(o.to_ndarray().reshape(-1, 2))
    if not chunks:
        return np.zeros((1, 2), "f4")
    return np.ascontiguousarray(np.concatenate(chunks, 0), "f4")


# Decoding the oggs through PyAV costs ~60 ms per file (a clip needs 50+), so the whole bank is decoded
# ONCE into data/sounds_cache/ (int16 stereo, ~75 MB, memory-mapped) and rebuilt automatically when
# the manifest changes.
CACHE_DIR = os.path.join(ROOT, "data", "sounds_cache")
_BANK = None


def _manifest_sig():
    """Changes when the manifest OR the set of installed sound files changes (files dropped in later)."""
    st = os.stat(os.path.join(SOUND_DIR, "manifest.json"))
    n = sum(1 for f in os.listdir(SOUND_DIR) if f.lower().endswith((".ogg", ".wav")))
    return "{}-{}-{}".format(int(st.st_mtime), st.st_size, n)


def sound_bank(build=True):
    """-> (int16 array (N, 2), {file: (offset, length)}) or None."""
    global _BANK
    if _BANK is not None:
        return _BANK
    idx_p, bank_p = os.path.join(CACHE_DIR, "index.json"), os.path.join(CACHE_DIR, "bank.npy")
    try:
        idx = json.load(open(idx_p, "r"))
        if idx.get("sig") == _manifest_sig():
            _BANK = (np.load(bank_p, mmap_mode="r"), idx["files"])
            return _BANK
    except (OSError, ValueError):
        pass
    if not build:
        return None
    t = time.perf_counter()
    manifest = json.load(open(os.path.join(SOUND_DIR, "manifest.json"), "r"))
    parts, files, off = [], {}, 0
    for fns in manifest["sounds"].values():
        for fn in fns:
            if not os.path.isfile(os.path.join(SOUND_DIR, fn)):
                continue                                  # not installed: that sound stays silent
            a = np.clip(_decode(os.path.join(SOUND_DIR, fn)) * 32767.0, -32768, 32767).astype("i2")
            files[fn] = (off, len(a))
            parts.append(a)
            off += len(a)
    if not parts:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(bank_p, np.concatenate(parts, 0))
    with open(idx_p, "w") as f:
        json.dump({"sig": _manifest_sig(), "files": files}, f)
    log("sound cache built in {:.1f}s ({} files)".format(time.perf_counter() - t, len(files)))
    _BANK = (np.load(bank_p, mmap_mode="r"), files)
    return _BANK


class OfflineAudio(rl_audio.Audio):
    """Drop-in for audio.Audio that RECORDS what the live mixer would play (same gains, panning and
    per-category sliders) as plain events, and mixes events (possibly merged from several parallel
    workers) into one stereo buffer."""

    def __init__(self, clock, export_volume=0.7, seed=1):
        # deliberately not calling Audio.__init__ (it would start pygame)
        self.clock = clock
        self.export_volume = export_volume
        self.cat = {"ball": 1.0, "post": 1.0, "demo": 1.0, "impact": 1.0, "boost": 1.0, "engine": 1.0, "flips": 1.0,
                    "reset": 1.0}
        self._cat_cache = {}
        self._last_pick = {}
        self.listener = None
        self.active = True
        mpath = os.path.join(SOUND_DIR, "manifest.json")
        manifest = json.load(open(mpath, "r")) if os.path.exists(mpath) else {"sounds": {}}
        # only the files actually installed (any subset can be dropped into data/sounds/)
        self.files = {k: [fn for fn in v if os.path.isfile(os.path.join(SOUND_DIR, fn))]
                      for k, v in manifest["sounds"].items()}
        self.files = {k: v for k, v in self.files.items() if v}
        try:
            import engine_synth
            engine_ok = all(os.path.isfile(os.path.join(engine_synth.SRC_DIR, "{}.wav".format(l[0])))
                            for l in engine_synth.LAYERS)
        except Exception:
            engine_ok = False
        self.silent = not self.files and not engine_ok   # nothing installed: video-only clips
        self.engine_rates = list(manifest.get("engine_rates", []))
        self.sounds = {k: [None] * len(v) for k, v in self.files.items()}   # decoded lazily
        self.layers = [(layer, k) for layer in ("engine_a", "engine_b") for k in range(len(self.files.get(layer, [])))]
        self._rng = random.Random(seed)
        self.reset()

    # Clips use the user's own mix: master volume + per-category sliders (render_hud copies them from
    # rsv_settings.json every frame), but NEVER the vis's mute -- a clip always has sound.
    ok = True
    muted = False

    @property
    def volume(self):
        return self.export_volume

    @volume.setter
    def volume(self, v):
        if os.environ.get("RSV_CLIP_VOLUME") is None:
            self.export_volume = max(0.0, min(1.0, float(v)))

    def reset(self):
        self.shots = []          # (t, name, k, l, r)
        self.loops = []          # boost loops: dict(key, t0, t1, env=[(t, l, r, speed)])
        self.releases = []       # (key, t_start, t_release): fade the boost start layer 300 ms
        self._boost = {}
        self.engine = []         # (t, inputs dict | None) per frame -> EngineModel + EngineSynth

    def _arr(self, name, k):
        a = self.sounds[name][k]
        if a is None:
            fn = self.files[name][k]
            bank = sound_bank()
            if bank is not None and fn in bank[1]:
                o, n = bank[1][fn]
                a = np.asarray(bank[0][o:o + n], "f4") * (1.0 / 32767.0)
            else:
                a = _decode(os.path.join(SOUND_DIR, fn))
            self.sounds[name][k] = a
        return a

    def _pick_idx(self, name):
        """Variant choice is a pure function of (event time, name) -- NOT a running RNG -- so a clip
        split across parallel workers picks exactly the same recordings as a single-process render."""
        n = len(self.files.get(name, []))
        if n == 0:
            return None
        if n == 1:
            return 0
        h = zlib.crc32("{}@{}".format(name, int(round(self.clock() * 1000.0))).encode())
        return h % n

    # ---- interface used by the renderer ---------------------------------------------------- #
    def set_active(self, active):
        pass

    def toggle_mute(self):
        pass

    def set_volume(self, v):
        pass

    def stop_all(self):
        pass

    def play(self, name, pos=None, gain=1.0, local=False):
        if self.silent:
            return
        gain *= self._cat_gain(name)
        if gain <= 0.0:
            return None
        k = self._pick_idx(name)
        if k is None:
            return None
        l, r = self._gains(pos, gain, local)
        self.shots.append((self.clock(), name, k, min(1.0, l), min(1.0, r)))
        return None

    def update_boost(self, boosting):
        """Alpha Boost, as audio.Audio: start layer now, loop from +300 ms (continuous speed pitch),
        release = loop fade 100 ms, start-layer fade 300 ms, tail."""
        if self.silent:
            return
        t = self.clock()
        for key, val in boosting.items():
            pos, local = val[0], val[1]
            speed = float(val[2]) if len(val) > 2 else 0.0
            st = self._boost.get(key)
            if st is None:
                self.play("boost_start", pos, 0.9, local)
                st = self._boost[key] = {"key": str(key), "t_start": t, "t0": t + self.BOOST_LOOP_DELAY,
                                         "t1": None, "env": []}
                self.loops.append(st)
            l, r = self._gains(pos, (0.8 if local else 0.7) * self.cat["boost"], local)
            st["env"].append((t, min(1.0, l), min(1.0, r), speed))
            st["pos"], st["local"] = pos, local
        for key in [k for k in self._boost if k not in boosting]:
            st = self._boost.pop(key)
            st["t1"] = t
            self.releases.append((st["key"], st["t_start"], t))
            self.play("boost_stop", st.get("pos"), 0.8, st.get("local", False))

    def update_engine(self, car):
        if self.silent:
            return
        t = self.clock()
        if car is None:
            self.engine.append((t, None))
            return
        l, r = self._gains(car["pos"], self.ENGINE_GAIN * self.cat["engine"], True)
        self.engine.append((t, {"on_ground": bool(car["on_ground"]), "v_fwd": float(car["v_fwd"]),
                                "throttle": car.get("throttle"), "steer": car.get("steer"),
                                "boosting": bool(car.get("boosting", False)), "speed": car.get("speed"),
                                "l": l, "r": r}))

    # ---- export / merge ---------------------------------------------------------------------- #
    def export(self, a, b):
        """Only the events inside [a, b) (a worker's own segment, warm-up excluded). Boost loops that
        were already running at a / still running at b are marked continuous (no fades at the join)."""
        loops = []
        for st in self.loops:
            env = [e for e in st["env"] if a <= e[0] < b]
            if not env:
                continue
            t1 = st["t1"]
            loops.append({"key": st["key"], "t_start": st["t_start"], "t0": max(st["t0"], a),
                          "cont_start": st["t0"] < a,
                          "t1": t1 if (t1 is not None and t1 < b) else b,
                          "cont_end": t1 is None or t1 >= b, "env": env})
        return {"shots": [s for s in self.shots if a <= s[0] < b],
                "loops": loops,
                "releases": [x for x in self.releases if a <= x[2] < b],
                "engine": [e for e in self.engine if a <= e[0] < b]}

    def load(self, parts):
        self.reset()
        for p in parts:
            self.shots += [tuple(s) for s in p["shots"]]
            self.loops += p["loops"]
            self.releases += [tuple(x) for x in p.get("releases", [])]
            self.engine += [tuple(e) for e in p["engine"]]
        self.engine.sort(key=lambda e: e[0])
        # one boost continuing across a worker join is two pieces: stitch them back into one loop
        # so its phase (and pitch glide) is continuous
        self.loops.sort(key=lambda l: (l["key"], l["t_start"], l["t0"]))
        merged = []
        for lp in self.loops:
            m = merged[-1] if merged else None
            if (m is not None and m["key"] == lp["key"] and abs(m["t_start"] - lp["t_start"]) < 1e-6
                    and m.get("cont_end") and lp.get("cont_start")):
                m["env"] = m["env"] + lp["env"]
                m["t1"], m["cont_end"] = lp["t1"], lp.get("cont_end")
            else:
                merged.append(dict(lp))
        self.loops = merged

    # ---- mixdown ------------------------------------------------------------------------------ #
    def mix(self, t0, duration):
        n = int(math.ceil(duration * SAMPLE_RATE)) + 1
        out = np.zeros((n, 2), "f4")

        def idx(t):
            return int(round((t - t0) * SAMPLE_RATE))

        # boost start layers are faded out 300 ms after release (Wwise Stop, 300 ms transition)
        rel = {}
        for key, t_start, t_rel in self.releases:
            rel[round(t_start, 6)] = t_rel
        for t, name, k, l, r in self.shots:
            arr = self._arr(name, int(k))
            i = idx(t)
            if i >= n or i + len(arr) <= 0:
                continue
            a = arr[max(0, -i):]
            if name == "boost_start" and round(t, 6) in rel:
                j = max(0, idx(rel[round(t, 6)]) - max(0, i))
                if j < len(a):
                    a = a.copy()
                    m = min(len(a) - j, int(0.3 * SAMPLE_RATE))
                    a[j:j + m] *= np.linspace(1.0, 0.0, m, dtype="f4")[:, None]
                    a[j + m:] = 0.0
            i = max(0, i)
            m = min(len(a), n - i)
            out[i:i + m, 0] += a[:m, 0] * l
            out[i:i + m, 1] += a[:m, 1] * r

        def pitched(src, i0, i1, rate):
            """src looped from phase 0 at sample i0 with a per-sample playback rate -> (i1-i0, 2)."""
            pos = np.cumsum(rate) - rate[0]
            k0 = np.floor(pos).astype(np.int64)
            fr = (pos - k0).astype("f4")[:, None]
            a = src[k0 % len(src)]
            return a + (src[(k0 + 1) % len(src)] - a) * fr

        if "boost_loop" in self.files:
            src = self._arr("boost_loop", 0)
            for st in self.loops:
                env = st["env"]
                t_a = st["t0"]
                t_b = st["t1"] if st["t1"] is not None else t0 + duration
                end = t_b if st.get("cont_end") else t_b + 0.1          # 100 ms release fade
                i0, i1 = max(0, idx(t_a)), min(n, idx(end))
                if i1 <= i0 or not env:
                    continue
                tt = [e[0] for e in env]
                ts = t0 + np.arange(i0, i1) / SAMPLE_RATE
                gl = np.interp(ts, tt, [e[1] for e in env]).astype("f4")
                gr = np.interp(ts, tt, [e[2] for e in env]).astype("f4")
                sp = np.interp(ts, tt, [e[3] for e in env])
                if not st.get("cont_end"):
                    fade = np.clip((t_b + 0.1 - ts) / 0.1, 0.0, 1.0).astype("f4")
                    gl *= fade
                    gr *= fade
                rate = 2.0 ** (611.0 * np.minimum(1.0, sp / 2300.0) / 1200.0)
                y = pitched(src, i0, i1, rate)
                out[i0:i1, 0] += y[:, 0] * gl
                out[i0:i1, 1] += y[:, 1] * gr

        if self.engine:
            import engine_synth
            syn, model = engine_synth.EngineSynth(), engine_synth.EngineModel()
            if syn.ok:
                ev = self.engine
                for j, (t, inp) in enumerate(ev):
                    t_next = ev[j + 1][0] if j + 1 < len(ev) else t + 1.0 / 60.0
                    a_i, b_i = max(0, idx(t)), min(n, idx(t_next))
                    if b_i <= a_i:
                        continue
                    if inp is None:
                        y = syn.render(b_i - a_i, model.params(), 0.0, 0.0)
                    else:
                        p = model.update(t_next - t, inp["on_ground"], inp["v_fwd"], inp.get("throttle"),
                                         inp.get("steer"), inp.get("boosting", False), inp.get("speed"))
                        y = syn.render(b_i - a_i, p, inp["l"], inp["r"])
                    out[a_i:b_i] += y

        # soft limiter above 0.7 (demo + goal + bumps can stack)
        a = np.abs(out)
        over = a > 0.7
        out[over] = np.sign(out[over]) * (0.7 + 0.3 * np.tanh((a[over] - 0.7) / 0.3))
        return out


# ---- sources --------------------------------------------------------------------------------------- #
def load_replay(path):
    """-> dict(packets=[(t, json)], timeline=[(t, idx, cam_manual)], pov=None, t_start)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    packets = []
    for t, s in d["packets"]:
        try:
            packets.append((float(t), json.loads(s)))
        except ValueError:
            pass
    timeline = [(float(t), (None if i is None else int(i)), m) for t, i, m in d.get("timeline", [])]
    return {"packets": packets, "timeline": timeline, "pov": None, "t_start": d.get("t_start")}


def load_pop_bundle(path):
    """GGLPOP1 trajectory -> source dict. Reconstructs has_flip (HasFlipOrJump with the 1.25 s jump
    air-time window) from the stored onGround/hasJumped/hasFlipped/hasDoubleJumped so the flip reset
    indicator + sound fire exactly like on a live stream."""
    with open(path, "rb") as f:
        if f.read(8) != b"GGLPOP1\x00":
            return None
        _ver, _ts, cc, n = struct.unpack("<IIII", f.read(16))
        if n < 1:
            return None
        w = 1 + 9 + cc * 20
        nf, _team, pov, _ttg = struct.unpack("<IIIf", f.read(16))
        raw = f.read(4 * nf * w)
        if len(raw) < 4 * nf * w:
            return None
    frames = np.frombuffer(raw, np.float32).reshape(nf, w)
    t_first = float(frames[0, 0])
    air = [0.0] * cc
    prev_j = [False] * cc
    prev_t = None
    packets = []
    for fr in frames:
        t = t_first - float(fr[0])                   # column 0 = seconds remaining
        dt = 0.0 if prev_t is None else max(t - prev_t, 0.0)
        prev_t = t
        cars = []
        for c in range(cc):
            o = 10 + c * 20
            og, hj, hf, hdj = fr[o + 16] > .5, fr[o + 17] > .5, fr[o + 18] > .5, fr[o + 19] > .5
            air[c] = (air[c] + dt if prev_j[c] else 0.0) if (hj and not og) else 0.0
            prev_j[c] = hj
            cars.append({
                "team_num": 0 if c < cc // 2 else 1,
                "phys": {"pos": fr[o:o + 3].tolist(), "forward": fr[o + 3:o + 6].tolist(),
                         "up": fr[o + 6:o + 9].tolist(), "vel": fr[o + 9:o + 12].tolist(),
                         "ang_vel": fr[o + 12:o + 15].tolist()},
                "boost_amount": float(fr[o + 15]), "on_ground": bool(og),
                "has_flip": bool(og or (not hf and not hdj and air[c] < 1.25)),
                "is_demoed": False,
            })
        packets.append((t, {"gamemode": "soccar",
                            "ball_phys": {"pos": fr[1:4].tolist(), "vel": fr[4:7].tolist(),
                                          "ang_vel": fr[7:10].tolist()},
                            "cars": cars, "boost_pad_states": [True] * 34}))
    return {"packets": packets, "timeline": [], "pov": int(pov), "t_start": None}


def load_source(spec):
    return load_replay(spec["replay"]) if "replay" in spec else load_pop_bundle(spec["pop"])


def clip_frames(src, fps):
    """-> (t0, n_frames) of the encoded clip."""
    packets = src["packets"]
    t0 = packets[0][0] if src.get("t_start") is None else max(float(src["t_start"]), packets[0][0])
    t_end = packets[-1][0] + 0.15
    return t0, max(1, int((t_end - t0) * fps))


# ---- renderer --------------------------------------------------------------------------------------- #
# GPU colour conversion: the rendered RGB frame -> one R8 image laid out exactly like an I420
# (yuv420p) frame in memory -- Y plane (W x H) then U, V (W/2 x H/2 each, two rows packed per line) --
# vertically flipped and BT.709 limited-range, so ffmpeg does no CPU conversion at all.
YUV_VERT = """
#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
YUV_FRAG = """
#version 330
uniform sampler2D src;
uniform ivec2 size;          // W, H of the source frame
out float f_y;
vec3 px(int x, int yTop) { return texelFetch(src, ivec2(x, size.y - 1 - yTop), 0).rgb; }
void main() {
    int x = int(gl_FragCoord.x);
    int m = int(gl_FragCoord.y);             // memory row (GL row 0 is read back first)
    int W = size.x, H = size.y;
    if (m < H) {
        vec3 c = px(x, m);
        f_y = (16.0 + 219.0 * dot(c, vec3(0.2126, 0.7152, 0.0722))) / 255.0;
        return;
    }
    int q = m - H;                            // chroma rows: first H/4 lines U, next H/4 lines V
    bool isV = q >= H / 4;
    if (isV) q -= H / 4;
    int cr = q * 2 + (x >= W / 2 ? 1 : 0);    // chroma row (0 .. H/2-1)
    int cc = x % (W / 2);                     // chroma column
    vec3 c = 0.25 * (px(2 * cc, 2 * cr) + px(2 * cc + 1, 2 * cr) + px(2 * cc, 2 * cr + 1) + px(2 * cc + 1, 2 * cr + 1));
    float v = isV ? dot(c, vec3(0.5, -0.4542, -0.0458)) : dot(c, vec3(-0.1146, -0.3854, 0.5));
    f_y = (128.0 + 224.0 * v) / 255.0;
}
"""


def _parse_size(s):
    w, h = (int(v) for v in s.lower().split("x"))
    return w - w % 4, h - h % 4


class OfflineRenderer:
    def __init__(self, size=None, fps=None):
        import moderngl
        import main as rsv
        self.mgl = moderngl
        self.W, self.H = _parse_size(size or os.environ.get("RSV_CLIP_SIZE", "1920x1080"))
        self.fps = float(fps or os.environ.get("RSV_CLIP_FPS", "60"))
        t = time.perf_counter()
        self.ctx = moderngl.create_standalone_context(require=330)
        self.r = rsv.RSVRenderer()
        self.r.init_gl(self.ctx)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((self.W, self.H), 3)],
                                        depth_attachment=self.ctx.depth_renderbuffer((self.W, self.H)))
        self.yuv_prog = self.ctx.program(vertex_shader=YUV_VERT, fragment_shader=YUV_FRAG)
        self.yuv_prog["src"].value = 0
        self.yuv_prog["size"].value = (self.W, self.H)
        self.yuv_vao = self.ctx.vertex_array(self.yuv_prog, [])
        self.yuv_fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((self.W, self.H * 3 // 2), 1)])
        self.frame_bytes = self.W * self.H * 3 // 2
        # double-buffered pixel-pack buffers: frame N's readback DMA overlaps frame N+1's rendering
        self.pbos = [self.ctx.buffer(reserve=self.frame_bytes) for _ in range(2)]
        self.audio = OfflineAudio(VCLOCK, float(os.environ.get("RSV_CLIP_VOLUME", "0.7")))
        self.r.audio = self.audio
        self.gpu = str(self.ctx.info.get("GL_RENDERER", "?")).split("/")[0]
        self.ffmpeg, self.vcodec = find_ffmpeg()
        log("GPU: {} | encoder: {} | init {:.1f}s".format(self.gpu, self.vcodec[1], time.perf_counter() - t))

    def reset_scene(self):
        import state_manager
        import events
        import fx as rl_fx
        from ribbon import RibbonEmitter
        from states import GameState
        state_manager.global_state_manager.state = GameState()
        state_manager.forced_spectate_idx = None
        state_manager.scoreboard = None
        state_manager.gail_hud = None
        state_manager.hud_text = ""
        events.g_detector = events.EventDetector()
        events.event_queue.clear()
        r = self.r
        r.prev_state = None
        r.car_ribbons = []
        r.ball_ribbon = RibbonEmitter()
        r.fx = rl_fx.FX(self.ctx)
        r.wheel_rig.cars.clear()
        r._pad_prev = None
        r._pad_pick_t, r._pad_spawn_t = [], []
        r._car_cam_offset_smooth = r._car_cam_dir_smooth = None
        r.car_cam_time = 0
        r.cam_manual = None
        r._boost_last = {}
        r._cel_w = 0.0
        r._cel_last = None
        for a in ("_last_hit_snd", "_ball_in_goal", "_kickoff_pick_done"):
            if hasattr(r, a):
                delattr(r, a)
        self.audio.reset()

    def _feed(self, t, j):
        import state_manager
        import events
        st = state_manager.global_state_manager.state
        st.read_from_json(j)
        st.recv_interval = min(max(t - st.recv_time, 1e-3), 0.25) if st.recv_time > 0 else 1.0 / 30.0
        st.recv_time = t
        if isinstance(j, dict):
            state_manager.scoreboard = j.get("scoreboard")
            state_manager.gail_hud = j.get("gail_hud")
            state_manager.hud_text = j.get("hud", "")
        events.g_detector.process(j, st.boost_pad_locations, t)

    def _camera(self, src, T, ti):
        import state_manager
        if src.get("pov") is not None:
            state_manager.forced_spectate_idx = src["pov"]
            return ti
        tl = src.get("timeline") or []
        if not tl:
            return ti
        while ti + 1 < len(tl) and tl[ti + 1][0] <= T:
            ti += 1
        _t, idx, cam_manual = tl[ti]
        if idx is not None and idx >= 0:
            state_manager.forced_spectate_idx = int(idx)
        else:
            state_manager.forced_spectate_idx = None
            self.r.spectate_idx = -1
        self.r.cam_manual = cam_manual
        return ti

    def render_range(self, src, t0, f_a, f_b, out_video, warmup=WARMUP_S):
        """Encode frames [f_a, f_b) of the clip (frame f at t0 + f/fps) to out_video. A warm-up of
        `warmup` seconds before f_a is replayed at WARMUP_FPS but not encoded. -> audio export dict."""
        mgl = self.mgl
        # replay on the same smoothed playback clock as the live vis (state_manager.PlayoutClock),
        # computed over the WHOLE clip so every segment worker agrees
        if "_sched" not in src:
            import state_manager
            src["_sched"] = [(s_, p[1]) for s_, p in
                             zip(state_manager.playout_times([p[0] for p in src["packets"]]), src["packets"])]
        packets = src["_sched"]
        self.reset_scene()
        T_a = t0 + f_a / self.fps
        T_b = t0 + f_b / self.fps
        T_w = max(t0, T_a - warmup) if f_a > 0 else T_a
        # prime: everything up to the warm-up start, so interpolation/events begin settled
        pi = 0
        VCLOCK.t = T_w
        while pi < len(packets) and packets[pi][0] <= T_w:
            self._feed(*packets[pi])
            pi += 1
        if pi == 0:
            self._feed(*packets[0])
            pi = 1
        ti = 0
        step_w = 1.0 / WARMUP_FPS
        self.r.last_render_time = T_w - (step_w if T_w < T_a else 1.0 / self.fps)
        T = T_w
        while T < T_a - 1e-9:                         # warm-up (not encoded)
            VCLOCK.t = T
            while pi < len(packets) and packets[pi][0] <= T:
                self._feed(*packets[pi])
                pi += 1
            ti = self._camera(src, T, ti)
            self.r.paint(self.W, self.H, self.fbo, clip_capture=False)
            T = min(T + step_w, T_a)

        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "{}x{}".format(self.W, self.H),
               "-r", str(self.fps), "-i", "-", *self.vcodec,
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               out_video]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
        q = queue.Queue(maxsize=6)

        def writer():
            while True:
                b = q.get()
                if b is None:
                    break
                try:
                    enc.stdin.write(b)
                except OSError:
                    break
        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        pending = None
        try:
            for f in range(f_a, f_b):
                T = t0 + f / self.fps
                VCLOCK.t = T
                while pi < len(packets) and packets[pi][0] <= T:
                    self._feed(*packets[pi])
                    pi += 1
                ti = self._camera(src, T, ti)
                self.r.paint(self.W, self.H, self.fbo, clip_capture=False)
                pbo = self.pbos[f & 1]
                self.fbo.color_attachments[0].use(location=0)
                self.yuv_fbo.use()
                self.ctx.viewport = (0, 0, self.W, self.H * 3 // 2)
                self.ctx.disable(mgl.DEPTH_TEST | mgl.BLEND | mgl.CULL_FACE)
                self.yuv_vao.render(mgl.TRIANGLES, vertices=3)
                self.yuv_fbo.read_into(pbo, components=1, alignment=1)   # async into the PBO
                if pending is not None:
                    q.put(pending.read())                                 # previous frame: DMA done
                pending = pbo
            if pending is not None:
                q.put(pending.read())
        finally:
            q.put(None)
            wt.join()
            try:
                enc.stdin.close()
            except OSError:
                pass
            enc.wait()
        if enc.returncode != 0 or not os.path.isfile(out_video):
            raise RuntimeError("video encode failed (ffmpeg exit {})".format(enc.returncode))
        return self.audio.export(T_a, T_b)


# ---- assembly ------------------------------------------------------------------------------------- #
def _mux(videos, audio_parts, t0, n_frames, fps, out_path, sound_ref=None):
    """Concatenate segment videos (lossless) + mix all sound events into one AAC track -> out_path."""
    aud = sound_ref or OfflineAudio(VCLOCK, float(os.environ.get("RSV_CLIP_VOLUME", "0.7")))
    aud.load(audio_parts)
    base = os.path.splitext(out_path)[0]
    tmp_a = None
    if not aud.silent:                                # (a build without sounds makes video-only clips)
        tmp_a = base + ".audio.f32"
        aud.mix(t0, n_frames / fps).tofile(tmp_a)
    if len(videos) == 1:
        vin = ["-i", videos[0]]
        lst = None
    else:
        lst = base + ".concat.txt"
        with open(lst, "w", encoding="utf-8") as f:
            for v in videos:
                f.write("file '{}'\n".format(v.replace("\\", "/").replace("'", "'\\''")))
        vin = ["-f", "concat", "-safe", "0", "-i", lst]
    if tmp_a is None:
        rc = _run_ffmpeg([*vin, "-c:v", "copy", "-an", "-movflags", "+faststart", out_path])
    else:
        rc = _run_ffmpeg([*vin, "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", tmp_a,
                          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                          "-shortest", "-movflags", "+faststart", out_path])
    for p in [tmp_a, lst, *videos]:
        if p:
            try:
                os.remove(p)
            except OSError:
                pass
    if rc != 0:
        raise RuntimeError("mux failed (ffmpeg exit {})".format(rc))
    return len(aud.shots)


def _spawn_worker(job, job_dir):
    path = os.path.join(job_dir, "job_{}.json".format(job["id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f)
    logf = open(os.path.join(job_dir, "worker_{}.log".format(job["id"])), "w", encoding="utf-8")
    # SHIM_MCCOMPAT=0x800000001 makes the NVIDIA Optimus shim put THIS process's OpenGL on the dGPU
    # (read at process start, per process -- the vis window itself is unaffected). Without it the
    # driver hands python.exe the AMD 780M -- which is what happens when the RTX lacks VRAM headroom.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if job.get("use_rtx", True):
        env["SHIM_MCCOMPAT"] = "0x800000001"
    else:
        env.pop("SHIM_MCCOMPAT", None)
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", path],
                         stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env,
                         creationflags=CREATE_NO_WINDOW)
    p.logf = logf
    return p


def _collect(procs, job_dir, timeout_s):
    """wait for all workers (killing them all if one fails or the deadline passes); echo their
    [offline] lines (and any traceback) into our own log."""
    deadline = time.perf_counter() + timeout_s
    while any(p.poll() is None for p in procs):
        failed = any(p.poll() not in (None, 0) for p in procs)
        if failed or time.perf_counter() > deadline:
            log("worker failed or render timed out after {:.0f}s -- stopping all workers".format(timeout_s)
                if not failed else "a worker failed -- stopping the others")
            for p in procs:
                if p.poll() is None:
                    p.kill()          # its ffmpeg child dies with the job (kill-on-close / terminate)
            break
        time.sleep(0.05)
    for p in procs:
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            pass
        p.logf.close()
    for fn in sorted(glob.glob(os.path.join(job_dir, "worker_*.log"))):
        try:
            txt = open(fn, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for line in txt.splitlines():
            if line.startswith("[offline]") or "Error" in line or "Traceback" in line or line.startswith("  File"):
                print("  " + os.path.basename(fn)[:-4] + ": " + line, flush=True)


def render_clip(spec, out_path, workers=None):
    """Render one clip, split across `workers` parallel processes. spec = {"replay": path} | {"pop": path}."""
    t_real = time.perf_counter()
    fps = float(os.environ.get("RSV_CLIP_FPS", "60"))
    src = load_source(spec)
    if src is None or len(src["packets"]) < 2:
        raise RuntimeError("not enough packets in {}".format(spec))
    t0, n = clip_frames(src, fps)
    k = workers or int(os.environ.get("RSV_CLIP_WORKERS", "2"))
    k = max(1, min(k, n // int(fps * 1.5) or 1))      # segments of at least ~1.5 s
    k, why = plan_workers(k)
    if k < 1:
        raise MemoryError(why)
    use_rtx = _rtx_ok(k)
    log("{} worker(s) ({}), GPU: {}".format(k, why, "RTX" if use_rtx else "iGPU (RTX VRAM is busy)"))
    job_dir = tempfile.mkdtemp(prefix="rsv_clip_", dir=os.path.dirname(os.path.abspath(out_path)))
    try:
        bounds = [round(n * i / k) for i in range(k + 1)]
        procs, jobs = [], []
        for i in range(k):
            job = {"id": i, "mode": "segment", "spec": spec, "t0": t0, "f_a": bounds[i], "f_b": bounds[i + 1],
                   "use_rtx": use_rtx,
                   "video": os.path.join(job_dir, "seg_{:02d}.mp4".format(i)),
                   "audio": os.path.join(job_dir, "seg_{:02d}.json".format(i))}
            jobs.append(job)
            procs.append(_spawn_worker(job, job_dir))
        if os.path.exists(os.path.join(SOUND_DIR, "manifest.json")) and sound_bank(build=False) is None:
            sound_bank()                    # first run: build the cache while the workers render
        _collect(procs, job_dir, 120.0)
        bad = [j["id"] for j, p in zip(jobs, procs) if p.returncode != 0 or not os.path.isfile(j["audio"])]
        for b_ in bad:                                   # full tail of each failed worker's log
            try:
                tail = open(os.path.join(job_dir, "worker_{}.log".format(b_)), encoding="utf-8", errors="replace").read()[-3000:]
            except OSError:
                tail = "(no log)"
            log("worker {} exit {} log tail:\n{}".format(b_, procs[b_].returncode, tail))
        if bad:
            raise RuntimeError("worker(s) {} failed".format(bad))
        parts = []
        for j in jobs:
            with open(j["audio"], "r", encoding="utf-8") as f:
                parts.append(json.load(f))
        t_m = time.perf_counter()
        n_snd = _mux([j["video"] for j in jobs], parts, t0, n, fps, out_path)
        log("workers done after {:.1f}s, audio mix + mux {:.1f}s".format(t_m - t_real, time.perf_counter() - t_m))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    log("{}  {} frames ({:.1f}s @ {:.0f}fps) in {:.1f}s on {} worker(s), {} sounds".format(
        os.path.basename(out_path), n, n / fps, fps, time.perf_counter() - t_real, k, n_snd))
    return out_path


def render_pop_batch(bins, workers=None, keep_bin=False):
    """Whole pop-reset clips distributed round-robin over parallel workers."""
    t_real = time.perf_counter()
    k = max(1, min(workers or int(os.environ.get("RSV_CLIP_WORKERS", "2")), len(bins)))
    k, why = plan_workers(k)
    if k < 1:
        log("NOT rendering: " + why)
        return 0
    use_rtx = _rtx_ok(k)
    log("{} worker(s) ({}), GPU: {}".format(k, why, "RTX" if use_rtx else "iGPU (RTX VRAM is busy)"))
    job_dir = tempfile.mkdtemp(prefix="rsv_pop_", dir=os.path.dirname(os.path.abspath(bins[0])))
    try:
        procs = [_spawn_worker({"id": i, "mode": "pop", "bins": bins[i::k], "keep_bin": keep_bin,
                                "use_rtx": use_rtx}, job_dir) for i in range(k)]
        _collect(procs, job_dir, 60.0 + 45.0 * len(bins) / k)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    ok = sum(1 for b in bins if os.path.isfile(b[:-4] + ".mp4"))
    log("done: {}/{} mp4(s) in {:.1f}s on {} worker(s)".format(ok, len(bins), time.perf_counter() - t_real, k))
    return ok


def run_worker(job_path):
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    t_imp = time.perf_counter()
    rr = OfflineRenderer()
    if job["mode"] == "segment":
        src = load_source(job["spec"])
        t_r = time.perf_counter()
        part = rr.render_range(src, job["t0"], job["f_a"], job["f_b"], job["video"])
        log("segment {}: frames {}-{} rendered in {:.1f}s (process up {:.1f}s before init)".format(
            job["id"], job["f_a"], job["f_b"], time.perf_counter() - t_r, t_imp - _T_PROC0))
        with open(job["audio"], "w", encoding="utf-8") as f:
            json.dump(part, f)
        return 0
    fails = 0
    for b in job["bins"]:                               # mode "pop": whole clips, one after another
        try:
            src = load_pop_bundle(b)
            if src is None:
                log("{}: unreadable / empty -- skipped".format(os.path.basename(b)))
                fails += 1
                continue
            t0, n = clip_frames(src, rr.fps)
            out = b[:-4] + ".mp4"
            vid = b[:-4] + ".video.mp4"
            t = time.perf_counter()
            part = rr.render_range(src, t0, 0, n, vid)
            n_snd = _mux([vid], [part], t0, n, rr.fps, out, sound_ref=rr.audio)
            log("{}  {} frames in {:.1f}s, {} sounds".format(os.path.basename(out), n, time.perf_counter() - t, n_snd))
            if not job.get("keep_bin"):
                os.remove(b)
        except Exception as e:
            log("{}: FAILED {!r}".format(os.path.basename(b), e))
            fails += 1
    return 1 if fails else 0


def _write_done(done_path, text):
    if not done_path:
        return
    try:
        with open(done_path, "w") as f:
            f.write(text)
    except OSError:
        pass


_T_PROC0 = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="gzip JSON packet ring written by the vis")
    ap.add_argument("--pop", help="GGLPOP1 .bin file or folder of pop_reset_*.bin")
    ap.add_argument("--out", help="output mp4 (replay mode)")
    ap.add_argument("--done", help="write the finished mp4 path here (the GUI polls it)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--keep-bin", action="store_true", help="pop mode: keep .bin after its mp4 exists")
    ap.add_argument("--keep-replay", action="store_true")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--build-sound-cache", action="store_true")
    args = ap.parse_args()

    if args.build_sound_cache:
        sound_bank()
        return 0

    if args.worker:
        return run_worker(args.worker)

    # no Windows hard-error popups ("the application failed to start...") from us or any child:
    # a failure must stay a logged, silent failure of the render (children inherit the error mode)
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        except Exception:
            pass
    # the whole render tree lives in one capped job (workers + ffmpeg inherit it)
    k_req = args.workers or int(os.environ.get("RSV_CLIP_WORKERS", "2"))
    _enter_job(MEM_BASE_GB + MEM_WORKER_GB * k_req + 0.8)

    if args.replay:
        out = args.out or args.replay.replace(".json.gz", "") + ".mp4"
        try:
            render_clip({"replay": os.path.abspath(args.replay)}, out, args.workers)
        except Exception as e:
            log("FAILED: {!r}".format(e))
            _write_done(args.done, "ERROR: " + (str(e) if isinstance(e, MemoryError) else "render failed ({!r})".format(e)))
            if not args.keep_replay:
                try:
                    os.remove(args.replay)
                except OSError:
                    pass
            return 1
        _write_done(args.done, out)
        if not args.keep_replay:
            try:
                os.remove(args.replay)
            except OSError:
                pass
        return 0

    if args.pop:
        bins = [os.path.abspath(args.pop)] if os.path.isfile(args.pop) else \
            sorted(glob.glob(os.path.join(os.path.abspath(args.pop), "pop_reset_*.bin")))
        pending = [b for b in bins if not os.path.isfile(b[:-4] + ".mp4")]
        if not pending:
            log("nothing to render ({} clip(s), all already have an mp4)".format(len(bins)))
            return 0
        ok = render_pop_batch(pending, args.workers, args.keep_bin)
        return 0 if ok == len(pending) else 1

    ap.error("give --replay or --pop")


if __name__ == "__main__":
    sys.exit(main())

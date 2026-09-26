"""Rolling gameplay clip recorder for RocketSimVis.

Keeps a ring buffer of the last few seconds of *rendered* frames (read off the GL framebuffer,
so it captures exactly what's on screen) and, on demand, encodes the most recent CLIP_SECONDS to
an mp4. Two triggers: the 'C' hotkey in the vis window, and a small signal file (written by the
training GUI's "Clip 12s" button) that we poll.

Performance design (so an always-on buffer barely costs FPS):
  * The screen is MULTISAMPLED (samples=4). glReadPixels on an MSAA framebuffer is illegal and
    hard-crashes the driver, so we first blit/resolve it into a plain single-sample FBO.
  * Readback uses a DOUBLE-BUFFERED PBO: read_into() issues an async DMA into a pixel buffer and
    returns immediately; we collect the PREVIOUS frame's buffer (DMA already finished) so the GL
    thread never stalls waiting on the GPU.
  * All CPU work (vertical flip, RGB->BGR, downscale, JPEG encode) happens on a WORKER THREAD, not
    the render thread.
  * Capture is throttled to CAPTURE_FPS.
Any GL hiccup self-disables capture so the visualizer keeps running no matter what; set
RSV_CLIP_DISABLE=1 to turn capture off entirely. On completion a clip_done.txt is written (the GUI
polls it to flip its button back from "Saving").
"""

import os
import time
import queue
import threading
from collections import deque

import numpy as np
# cv2 (~0.5 s to import) is imported lazily on the worker threads that use it, not at startup.

CLIP_SECONDS = float(os.environ.get("RSV_CLIP_SECONDS", "12"))   # how much gameplay the clip keeps
# "replay" (default): dump the last CLIP_SECONDS of received packets + camera choices and re-render
# them with offline_render.py -- headless on the RTX, 1080p60, with sound, NVENC -- instead of
# screen-grabbing this window. "frames": the old JPEG frame ring (no sound, 20 fps, window-res).
CLIP_MODE = os.environ.get("RSV_CLIP_MODE", "replay").lower()
CAPTURE_FPS = 20.0      # frames/sec pulled into the ring buffer (plenty for a clip)
MAX_WIDTH = 1280        # downscale wider frames to bound memory / encode cost
JPEG_QUALITY = 85


class ClipRecorder:
    def __init__(self):
        self._capture_interval = 1.0 / CAPTURE_FPS
        self._last_capture = 0.0
        self._frames = deque()  # (timestamp, jpeg_bytes_ndarray, (h, w))
        self._lock = threading.Lock()
        self._disabled = bool(int(os.environ.get("RSV_CLIP_DISABLE", "0") or "0"))

        # GL resources for the async readback (created lazily on first capture / on resize).
        self._resolve_fbo = None
        self._resolve_tex = None
        self._pbos = [None, None]
        self._gl_size = None
        self._pbo_idx = 0
        self._pending = False
        self._pending_ts = 0.0

        # Encode worker (off the GL thread).
        self._work_q = queue.Queue(maxsize=4)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # rough capture-overhead readout for RSV_PERF logging
        self.last_capture_ms = 0.0

        # Output folder + GUI signal/done files. The training GUI passes RSV_CLIP_DIR so both sides
        # agree; standalone (RUN.bat) launches fall back to a "clips" folder next to the project.
        self.out_dir = os.environ.get("RSV_CLIP_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clips")
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except OSError:
            pass
        self._signal_path = os.path.join(self.out_dir, "clip_request.txt")
        self._done_path = os.path.join(self.out_dir, "clip_done.txt")
        self._last_signal = self._read_file(self._signal_path)  # ignore pre-existing request
        self._last_signal_poll = 0.0

        print("[clip] recorder ready (disabled={}), clips -> {}".format(self._disabled, self.out_dir))

    # ---- buffering (GL thread) ------------------------------------------- #

    def capture(self, ctx, width, height, src_fb=None):
        if self._disabled or width <= 0 or height <= 0 or CLIP_MODE == "replay":
            return
        now = time.time()
        if now - self._last_capture < self._capture_interval:
            return
        self._last_capture = now
        t0 = time.perf_counter()
        try:
            self._grab(ctx, width, height, now, src_fb if src_fb is not None else ctx.screen)
        except Exception as e:
            self._disabled = True
            print("[clip] capture disabled after error: {!r}".format(e))
            self._free_gl()
            return
        self.last_capture_ms = (time.perf_counter() - t0) * 1000.0

    def _grab(self, ctx, width, height, now, src_fb):
        # Downscale ON THE GPU to the stored clip size (<= MAX_WIDTH wide) before reading back: a
        # 3200x2000 window was being read in full (19 MB per capture) and resized on the CPU.
        out_w = min(width, MAX_WIDTH)
        out_h = max(1, int(round(height * out_w / width)))
        if self._gl_size != (out_w, out_h):
            self._free_gl()
            size = out_w * out_h * 3
            self._resolve_tex = ctx.texture((out_w, out_h), 3)
            self._resolve_fbo = ctx.framebuffer(color_attachments=[self._resolve_tex])
            self._pbos = [ctx.buffer(reserve=size, dynamic=True),
                          ctx.buffer(reserve=size, dynamic=True)]
            self._gl_size = (out_w, out_h)
            self._pbo_idx = 0
            self._pending = False
        src_w, src_h = width, height
        width, height = out_w, out_h

        # Scaled, filtered blit (window framebuffer is single-sample now, so this is legal).
        from OpenGL import GL as _gl
        _gl.glBindFramebuffer(_gl.GL_READ_FRAMEBUFFER, src_fb.glo)
        _gl.glBindFramebuffer(_gl.GL_DRAW_FRAMEBUFFER, self._resolve_fbo.glo)
        _gl.glBlitFramebuffer(0, 0, src_w, src_h, 0, 0, out_w, out_h, _gl.GL_COLOR_BUFFER_BIT, _gl.GL_LINEAR)
        _gl.glBindFramebuffer(_gl.GL_FRAMEBUFFER, src_fb.glo)

        cur = self._pbo_idx
        other = 1 - cur
        # Collect the readback issued LAST frame (its DMA has had a full frame to complete) before
        # kicking off a new one — this is what keeps the GL thread from stalling on the GPU.
        if self._pending:
            raw = self._pbos[other].read()
            try:
                self._work_q.put_nowait((self._pending_ts, raw, width, height))
            except queue.Full:
                pass  # worker behind — drop this frame rather than block rendering
        self._resolve_fbo.read_into(self._pbos[cur], components=3, alignment=1)
        self._pending = True
        self._pending_ts = now
        self._pbo_idx = other

    def _free_gl(self):
        for obj in (self._resolve_fbo, self._resolve_tex, self._pbos[0], self._pbos[1]):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._resolve_fbo = self._resolve_tex = None
        self._pbos = [None, None]
        self._gl_size = None
        self._pending = False

    # ---- encode worker (CPU thread) -------------------------------------- #

    def _worker_loop(self):
        while True:
            item = self._work_q.get()
            if item is None:
                break
            ts, raw, w, h = item
            try:
                self._process(ts, raw, w, h)
            except Exception as e:
                print("[clip] worker error: {!r}".format(e))

    def _process(self, ts, raw, w, h):
        import cv2
        if len(raw) != w * h * 3:
            return
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        frame = frame[::-1, :, ::-1]            # vertical flip (GL bottom-up) + RGB->BGR in one view
        if w > MAX_WIDTH:
            nh = int(round(h * (MAX_WIDTH / w)))
            frame = cv2.resize(frame, (MAX_WIDTH, nh), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame)
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        fh, fw = frame.shape[:2]
        cutoff = ts - CLIP_SECONDS
        with self._lock:
            self._frames.append((ts, enc, (fh, fw)))
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()

    # ---- triggers --------------------------------------------------------- #

    def poll_signal(self):
        now = time.time()
        if now - self._last_signal_poll < 0.25:
            return
        self._last_signal_poll = now
        sig = self._read_file(self._signal_path)
        if sig is not None and sig != self._last_signal:
            self._last_signal = sig
            self.save_clip()

    @staticmethod
    def _read_file(path):
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except OSError:
            return None

    def save_clip(self):
        if CLIP_MODE == "replay" and not self._disabled:
            self._save_replay()
            return
        if self._disabled:
            print("[clip] capture is disabled — nothing to save")
            return
        with self._lock:
            frames = list(self._frames)
        if len(frames) < 2:
            print("[clip] not enough buffered frames yet")
            return
        threading.Thread(target=self._encode, args=(frames,), daemon=True).start()

    def _save_replay(self):
        """Snapshot the packet/camera rings to <out_dir>/.clip_*.json.gz and hand them to a detached
        offline_render.py process (it writes the mp4 + clip_done.txt when finished)."""
        import gzip
        import json
        import subprocess
        import sys
        import state_manager
        now = time.time()
        with state_manager.ring_lock:
            packets = list(state_manager.packet_ring)
            cams = list(state_manager.cam_ring)
        t_start = now - CLIP_SECONDS
        packets = [(t, d) for t, d in packets if t >= t_start - 1.0]     # 1 s pre-roll for events
        if len(packets) < 2:
            print("[clip] not enough received packets yet")
            return
        stamp = time.strftime("clip_%Y%m%d_%H%M%S")
        replay = os.path.join(self.out_dir, "." + stamp + ".json.gz")
        out = os.path.join(self.out_dir, stamp + ".mp4")
        try:
            with gzip.open(replay, "wt", encoding="utf-8", compresslevel=1) as f:
                json.dump({"version": 1, "t_start": max(t_start, packets[0][0]),
                           "packets": [[t, d.decode("utf-8", "replace")] for t, d in packets],
                           "timeline": [[t, i, m] for t, i, m in cams]}, f)
            logf = open(os.path.join(self.out_dir, "offline_render.log"), "a", encoding="utf-8")
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_render.py")
            subprocess.Popen([sys.executable, script, "--replay", replay, "--out", out, "--done", self._done_path],
                             stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                             creationflags=0x08000000 if os.name == "nt" else 0)
            print("[clip] rendering {:.1f}s ({} packets) on the dGPU -> {}".format(
                now - max(t_start, packets[0][0]), len(packets), out))
        except Exception as e:
            print("[clip] ERROR starting the offline render: {!r}".format(e))

    def _encode(self, frames):
        import cv2
        t0, t1 = frames[0][0], frames[-1][0]
        span = max(t1 - t0, 1e-3)
        fps = float(np.clip((len(frames) - 1) / span, 1.0, 60.0))
        h, w = frames[0][2]

        path = os.path.join(self.out_dir, time.strftime("clip_%Y%m%d_%H%M%S.mp4"))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            print("[clip] ERROR: could not open VideoWriter for {}".format(path))
            return
        for _, enc, _ in frames:
            fr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if fr is None:
                continue
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h))
            writer.write(fr)
        writer.release()
        print("[clip] saved {} frames ({:.1f}s @ {:.1f}fps) -> {}".format(
            len(frames), span, fps, path))
        # Tell the GUI the clip finished (it polls this to flip the button back from "Saving").
        try:
            with open(self._done_path, "w") as f:
                f.write(path)
        except OSError:
            pass

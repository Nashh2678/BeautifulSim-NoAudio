from states import *
from threading import Lock
import time

class StateManager:
    state: GameState = GameState()

global_state_manager = StateManager()
global_state_mutex = Lock()

# When True, the state-set POSE EDITOR (pose_editor.py) owns the scene: the UDP socket listener
# stops applying incoming gamestates so keyboard edits aren't overwritten. Toggled with B in the
# RocketSimVis window.
edit_mode = False

# Optional HUD text sent by the PLAY driver (play.py) in the gamestate JSON's "hud" field — shown
# in the RocketSimVis overlay (live reward value, record/playback status, slow-mo speed).
hud_text = ""

# A play/state-test tool sets this (via the "capture_input" JSON flag) so RocketSimVis suspends its
# OWN camera keybinds (left-click POV switch, space camera menu) — they'd otherwise fight the
# gameplay controls. Refreshed every frame the tool sends; is_input_captured() is the recency check.
input_capture_time = 0.0


def is_input_captured():
    return (time.time() - input_capture_time) < 0.5

# Only the State-Set Editor (which sends "allow_pose_edit") enables the B/V pose keys + posing in
# RocketSimVis. The play driver and the training renderer do NOT, so their views aren't hijacked.
pose_edit_time = 0.0


def is_pose_edit_allowed():
    return (time.time() - pose_edit_time) < 1.5

# Pose-editor camera override: (eye_xyz, look_at_xyz) while posing — RocketSimVis sits the camera
# behind the selected object (billiard/chase POV). None = use the normal camera.
pose_cam = None

# Optional car index a streaming tool wants the camera locked to (sent via the "spectate_idx" JSON
# field, e.g. ngp/view_credit.py aiming the POV at the highest-credit player). None = leave the
# user's own spectate choice alone. Applied each frame in main.render(); additive/opt-in.
forced_spectate_idx = None

# ---- replay rings for offline clip rendering (clip_recorder.py -> offline_render.py) ---------------- #
# Every received packet (raw JSON bytes + receive time) and every camera change of the last
# REPLAY_SECONDS, so a clip can be re-rendered faithfully -- on the RTX, at 60 fps, with sound --
# instead of screen-recording this window. Storing the raw bytes costs ~nothing per packet.
import os as _os
from collections import deque as _deque

REPLAY_SECONDS = float(_os.environ.get("RSV_CLIP_SECONDS", "12")) + 2.0
packet_ring = _deque()        # (recv_time, bytes)
cam_ring = _deque()           # (time, spectate_idx, cam_manual) -- appended on change only
ring_lock = Lock()


def record_packet(t, data):
    with ring_lock:
        packet_ring.append((t, data))
        cut = t - REPLAY_SECONDS
        while packet_ring and packet_ring[0][0] < cut:
            packet_ring.popleft()


def record_cam(t, idx, cam_manual):
    if cam_ring and cam_ring[-1][1] == idx and cam_ring[-1][2] == cam_manual:
        return
    with ring_lock:
        cam_ring.append((t, idx, cam_manual))
        cut = t - REPLAY_SECONDS
        while len(cam_ring) > 1 and cam_ring[1][0] < cut:     # keep the entry in force at the cut
            cam_ring.popleft()


# ---- smooth packet playback (jitter buffer) ------------------------------------------------------- #
# Packets arrive unevenly (a 30 Hz trainer feed measured 24..43 ms apart, with a ~100-160 ms hiccup every few
# seconds while training shares the machine). Each packet gets a playback time on a steady clock behind
# arrival, and the render thread applies it when that time comes (interpolating over the steady interval).
# The delay adapts: it jumps up to cover the latest lateness seen and decays back over ~45 s, so after the
# first hiccup the next ones are absorbed instead of freezing the ball for 50-150 ms. Measured on a recorded
# 4-minute training feed: visible holds 49 -> 8, for ~95 ms of display delay (irrelevant when spectating).
class PlayoutClock:
    LAG_DECAY = 45.0        # s: how long a hiccup keeps the buffer deep
    LAG_MAX = 0.15          # s: never buffer more than this (a longer gap is a pause, not jitter)
    MARGIN = 0.004

    def __init__(self):
        self.reset()

    def reset(self):
        self.T = None           # smoothed packet interval
        self.r = None           # steady reference clock tracking the MEAN arrival time
        self.lag = 2.0 / 30.0   # current buffer depth beyond the reference
        self.last_a = None      # last arrival time
        self.last_s = None      # last playback time

    def schedule(self, a):
        """arrival time -> playback time (monotonic, never before arrival)."""
        T = self.T if self.T is not None else 1.0 / 30.0
        if self.last_a is None or a - self.last_a > max(0.3, 4.0 * T) or a < self.last_a:
            self.r = a                                    # first packet / after a pause: resync
            s = a + self.lag + self.MARGIN
        else:
            dt = a - self.last_a
            self.T = T = dt if self.T is None else 0.98 * T + 0.02 * min(max(dt, 0.002), 0.25)
            self.r += T
            dev = a - self.r                              # > 0: this packet is late vs the steady clock
            self.r += 0.02 * dev
            self.lag = min(max(dev, self.lag * (1.0 - T / self.LAG_DECAY), 0.3 * T), self.LAG_MAX)
            s = self.r + self.lag + self.MARGIN
            s = max(s, a, self.last_s + 0.25 * T)         # never before arrival, always moving forward
        self.last_a, self.last_s = a, s
        return s


def playout_times(arrivals):
    c = PlayoutClock()
    return [c.schedule(a) for a in arrivals]


playout_clock = PlayoutClock()
pending_packets = _deque()    # (playback_time, json dict), filled by the socket thread
pending_lock = Lock()


def queue_packet(s, j):
    with pending_lock:
        pending_packets.append((s, j))
        while len(pending_packets) > 240:      # not rendering (hidden window): keep only recent ones
            pending_packets.popleft()


def apply_due_packets(now):
    """Render thread, holding global_state_mutex: apply every queued packet whose playback time has
    come (after a long stall only the last two, so prev/next interpolation stays consistent)."""
    with pending_lock:
        due = []
        while pending_packets and pending_packets[0][0] <= now:
            due.append(pending_packets.popleft())
    if not due:
        return
    st = global_state_manager.state
    for s, j in due[-2:]:
        try:
            st.read_from_json(j)
        except Exception:
            import traceback
            print("ERROR reading received JSON:")
            traceback.print_exc()
            continue
        st.recv_interval = min(max(s - st.recv_time, 1e-3), 0.25) if st.recv_time > 0 else 1.0 / 30.0
        st.recv_time = s

"""
State-set POSE EDITOR for RocketSimVis.

Lets you pose the ball + cars with the KEYBOARD directly inside the RocketSimVis window (move,
rotate, set velocity), instead of typing coordinates into the GigaLearnCPP State-Set Editor. The
editor then only controls randomization ranges + which episode types the set is for, and reads the
live pose from here to save it.

Flow:
  - Press B in RocketSimVis to toggle "edit mode". While on, the UDP-received state is ignored
    (state_manager.edit_mode) and this module OWNS global_state_manager.state.
  - Keys (see KEYMAP in main.py / the on-screen HUD) move/rotate/velocity the SELECTED object.
  - After every change the current pose is streamed to the editor on UDP 127.0.0.1:9274, in the
    same JSON shape the editor uses, so "Save state set" captures exactly what's on screen.

Coordinates are RocketSim uu: X = side wall, Y = forward/back (goals at ±Y), Z = up.
"""

import socket
import json
import math
import os
import copy
import traceback

import state_manager
from states import CarState
from pyrr import Vector3

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_editor_log.txt")


def _log(msg):
    try:
        with open(_LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

try:
    import keyboard          # global key state — works regardless of Qt focus (the reliable path)
except Exception:
    keyboard = None
try:
    import mouse             # left/right click binds (boost/jump) in test mode
except Exception:
    mouse = None
try:
    import RocketSim as rs   # physics for TEST mode (play from the posed state)
except Exception:
    rs = None


def _kp(name):
    if keyboard is None:
        return False
    try:
        return keyboard.is_pressed(name)
    except Exception:
        return False


# Test-mode driving binds (ZQSD, same as stateset/play.py). mouse_* = mouse buttons. Pitch/yaw come
# from throttle/steer in the air (standard RL KBM) — the t/g/f/h keys are posing-only.
TEST_BINDS = {
    "throttle": "z", "reverse": "s", "steer_left": "q", "steer_right": "d",
    "jump": "mouse_right", "boost": "mouse_left", "powerslide": "left shift",
    "air_roll_left": "a", "air_roll_right": "e", "ball_cam": "space",
    "ball_reset_front": "&", "ball_reset_dribble": "é",
}


import ctypes
_MOUSE_VK = {"left": 0x01, "right": 0x02, "middle": 0x04}


def _mouse_down(button):
    # Real hardware state (no lag under rapid clicking — unlike mouse.is_pressed). See play.py.
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(_MOUSE_VK.get(button, 0)) & 0x8000)
    except Exception:
        return False


def _bind_down(name):
    n = TEST_BINDS.get(name, "")
    if not n:
        return False
    try:
        if n.startswith("mouse_"):
            return _mouse_down(n.split("_", 1)[1])
        return _kp(n)
    except Exception:
        return False


_rs_inited = False


def _ensure_rs_init():
    """RocketSim needs init() (collision meshes) before any Arena. The path comes from the
    RS_COLLISION_MESHES env var, which the State-Set Editor sets when it launches RocketSimVis."""
    global _rs_inited
    if rs is None or _rs_inited:
        return _rs_inited
    try:
        meshes = os.environ.get("RS_COLLISION_MESHES", "")
        rs.init(meshes) if meshes else rs.init()
        _rs_inited = True
    except Exception:
        _log("rs.init failed:\n" + traceback.format_exc())
    return _rs_inited

EDITOR_IP = "127.0.0.1"
EDITOR_PORT = 9274          # the State-Set Editor listens here for live poses

POS_STEP = 50.0            # uu per key tap (Shift = ×5 via main.py)
Z_STEP = 30.0
ANG_STEP = 15.0           # degrees per tap
VEL_STEP = 100.0          # uu/s per tap
BALL_RADIUS = 92.75
CAR_REST_Z = 17.0


def angle_to_fwd_up(yaw_deg, pitch_deg, roll_deg):
    """Yaw/Pitch/Roll (deg) -> (forward, up) tuples, matching RocketSim Angle::ToRotMat exactly."""
    p, y, r = math.radians(pitch_deg), math.radians(yaw_deg), math.radians(roll_deg)
    CP, SP = math.cos(p), math.sin(p)
    CY, SY = math.cos(y), math.sin(y)
    CR, SR = math.cos(r), math.sin(r)
    forward = (CP * CY, CP * SY, SP)
    up = (-CY * SP * CR - SR * SY, -SY * SP * CR + SR * CY, CP * CR)
    return forward, up


def _set_pos(phys, x, y, z):
    phys.prev_pos = Vector3((x, y, z))
    phys.next_pos = Vector3((x, y, z))


def _set_vel(phys, x, y, z):
    phys.prev_vel = Vector3((x, y, z))
    phys.next_vel = Vector3((x, y, z))


class PoseEditor:
    def __init__(self):
        self.active = False
        self.sel = -1                 # -1 = ball, 0..n-1 = car index
        self.angles = {}              # car idx -> [yaw, pitch, roll] degrees
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._edges = {}              # edge-detection state for tap keys
        # TEST mode: play from the posed state with real RocketSim physics (V toggles).
        self.test_mode = False
        self.test_arena = None
        self.test_cars = []
        self.test_started = False     # frozen until the first gameplay input
        self._controlled = None       # the car you drive in test mode
        self._pose_snapshot = None
        self._angles_snapshot = None
        self._ctrl_down_since = None   # for the hold-Ctrl-to-snap-flat behavior
        self._last_test_time = None    # real-time physics stepping in test mode
        # Scroll-to-set-initial-speed (along the car nose / the ball's aim). The ball has no
        # orientation, so when it's selected F/H/T/G aim its "billiard shot" direction instead.
        self._scroll_accum = 0.0
        self._sel_speed = 0.0          # current armed speed of the selected object (uu/s)
        self.ball_aim = [90.0, 0.0]    # yaw, pitch of the ball's shot direction (deg)
        if mouse is not None:
            try:
                mouse.hook(self._on_mouse)
            except Exception:
                pass

    def _on_mouse(self, event):
        # mouse.hook delivers move/button/wheel events on a background thread. Only wheel events have
        # a .delta; accumulate it for the poll loop to consume (float += is fine for this).
        d = getattr(event, "delta", None)
        if d is not None:
            self._scroll_accum += d

    def _edge(self, name, cur):
        was = self._edges.get(name, False)
        self._edges[name] = cur
        return cur and not was

    def poll_keys(self):
        # Crash-proof wrapper: an error here must NEVER take down the RocketSimVis window (it's a
        # QTimer slot, and PyQt aborts the app on an unhandled slot exception). Errors are logged.
        try:
            self._poll_keys_impl()
        except Exception:
            _log("poll_keys error:\n" + traceback.format_exc())

    def _poll_keys_impl(self):
        """Polled ~30x/s from RocketSimVis (a QTimer). Uses the global `keyboard` state so it works
        no matter which widget has Qt focus. Tap keys (B/N/R) are edge-detected; move/rotate/velocity
        repeat while held."""
        # The B/V pose keys ONLY work while the State-Set Editor is connected (it flags its stream).
        # In the play-driver / training-render views they're disabled, and any lingering edit mode is
        # force-exited so those views render normally.
        if not state_manager.is_pose_edit_allowed():
            if self.active:
                self.active = False
                self.test_mode = False
                state_manager.edit_mode = False
                state_manager.pose_cam = None
            return
        if self._edge("b", _kp("b")):
            self.toggle()
        if not self.active:
            return
        if self._edge("v", _kp("v")):
            self.toggle_test()
        # While editing/testing, suspend RocketSimVis's own camera keybinds (left-click POV, space
        # menu) — they fight the posing/gameplay controls. Keep the scene "live" for the renderer.
        import time as _t
        state_manager.input_capture_time = _t.time()
        with state_manager.global_state_mutex:
            self.recv_time_keepalive(state_manager.global_state_manager.state)
        if self.test_mode:
            self._test_tick()
            return
        # --- POSE controls (same key layout as driving): Z/S = fwd/back (Y), Q/D = left/right (X),
        # Shift/Ctrl = up/down (Z); rotate with T/G = nose down/up, F/H = yaw left/right, A/E = roll. ---
        # Left click (or N) cycles which object you're setting (ball/car). Both edges evaluated.
        lc = self._edge("lclick", _mouse_down("left"))
        nk = self._edge("n", _kp("n"))
        if lc or nk:
            self.select_next()
        if self._edge("backspace", _kp("backspace")):
            self.reset_selected()
        # Movement is RELATIVE to the chase camera (behind the aim direction): Z = into the screen
        # (away from camera), S = toward, Q/D strafe left/right — regardless of how the object is
        # oriented. (World-axis movement is what felt "reversed".) Shift/Ctrl stay world up/down.
        fwd = self._aim_dir()
        n = math.hypot(fwd[0], fwd[1]) or 1.0
        fx, fy = fwd[0] / n, fwd[1] / n
        f = (1 if _kp("z") else 0) - (1 if _kp("s") else 0)
        r = (1 if _kp("q") else 0) - (1 if _kp("d") else 0)
        dz = (1 if _kp("shift") else 0) - (1 if _kp("ctrl") else 0)
        if f or r or dz:
            self.move((fx * f + fy * r) * 12, (fy * f - fx * r) * 12, dz * 8)
        # Arrows pan/yaw (left/right) and pitch (up=nose down, down=nose up) — rotates the car AND the
        # chase camera together, and the speed direction follows. A/E roll.
        dyaw = ((1 if _kp("right") else 0) - (1 if _kp("left") else 0)) * 0.6
        dpitch = ((1 if _kp("down") else 0) - (1 if _kp("up") else 0)) * 0.6
        droll = ((1 if _kp("e") else 0) - (1 if _kp("a") else 0)) * 0.6
        if self.sel < 0:
            # Ball has no orientation: F/H/T/G aim its "billiard shot" direction instead.
            self.ball_aim[0] += dyaw
            self.ball_aim[1] = max(-89.0, min(89.0, self.ball_aim[1] + dpitch))
        elif dyaw or dpitch or droll:
            self.rotate(dyaw, dpitch, droll)
        # Scroll sets the initial speed (uu/s); it's applied along the aim direction every poll.
        if abs(self._scroll_accum) > 1e-6:
            self._sel_speed = max(0.0, self._sel_speed + self._scroll_accum * 100.0)
            self._scroll_accum = 0.0
        self._apply_speed_and_camera()
        self._ground_snap_check()   # hold Ctrl on a grounded car -> snap flat (clip to ground)

    # ---- mode ----
    def toggle(self):
        self.active = not self.active
        state_manager.edit_mode = self.active
        if self.active:
            self.ensure_objects()
            self._init_sel_speed()
            self.send_pose()
        else:
            self.test_mode = False
            state_manager.pose_cam = None   # restore RocketSimVis's normal camera

    def ensure_objects(self):
        """Seed a default scene (ball at center + one blue/orange car) if the arena is empty."""
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            _set_pos(st.ball_state, 0, 0, BALL_RADIUS)
            _set_vel(st.ball_state, 0, 0, 0)
            st.ball_state.ang_vel = Vector3((0, 0, 0))
            st.ball_state.has_rot = False
            if len(st.car_states) == 0:
                for team, x, y in ((0, -256, -2300), (1, 256, 2300)):
                    c = CarState()
                    c.team_num = team
                    _set_pos(c.phys, x, y, CAR_REST_Z)
                    _set_vel(c.phys, 0, 0, 0)
                    c.boost_amount = 33.0
                    c.on_ground = True
                    st.car_states.append(c)
            for i, car in enumerate(st.car_states):
                self.angles.setdefault(i, [90.0 if car.team_num == 0 else -90.0, 0.0, 0.0])
                self._apply_angle(car, i)
            st.boost_pad_states = [True] * len(st.boost_pad_locations)  # show all boost pads
            self.recv_time_keepalive(st)

    @staticmethod
    def recv_time_keepalive(st):
        # render() shows "Connected" off recv_time; keep it positive so the scene draws normally.
        import time
        st.recv_time = time.time()
        st.recv_interval = 1 / 60.0

    def _apply_angle(self, car, idx):
        fwd, up = angle_to_fwd_up(*self.angles[idx])
        car.phys.prev_forward = Vector3(fwd)
        car.phys.next_forward = Vector3(fwd)
        car.phys.prev_up = Vector3(up)
        car.phys.next_up = Vector3(up)
        car.phys.has_rot = True

    # ---- selection ----
    def select_next(self):
        with state_manager.global_state_mutex:
            n = len(state_manager.global_state_manager.state.car_states)
        # order: ball (-1) -> car0 -> ... -> carN-1 -> ball
        self.sel = -1 if self.sel >= n - 1 else self.sel + 1
        self._init_sel_speed()
        self.send_pose()

    def _init_sel_speed(self):
        with state_manager.global_state_mutex:
            v = self._sel_phys(state_manager.global_state_manager.state).next_vel
        self._sel_speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _aim_dir(self):
        if self.sel < 0:
            f, _ = angle_to_fwd_up(self.ball_aim[0], self.ball_aim[1], 0.0)
        else:
            a = self.angles.get(self.sel, [90.0, 0.0, 0.0])
            f, _ = angle_to_fwd_up(a[0], a[1], a[2])
        return f

    def _apply_speed_and_camera(self):
        d = self._aim_dir()
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            ph = self._sel_phys(st)
            _set_vel(ph, d[0] * self._sel_speed, d[1] * self._sel_speed, d[2] * self._sel_speed)
            p = ph.next_pos
            state_manager.pose_cam = ((p.x - d[0] * 300.0, p.y - d[1] * 300.0, p.z - d[2] * 300.0 + 60.0),
                                      (p.x, p.y, p.z))

    def _sel_phys(self, st):
        if self.sel < 0:
            return st.ball_state
        if 0 <= self.sel < len(st.car_states):
            return st.car_states[self.sel].phys
        self.sel = -1
        return st.ball_state

    # ---- edits (deltas) ----
    def move(self, dx, dy, dz):
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            ph = self._sel_phys(st)
            p = ph.next_pos
            # Clip to the soccar field — real wall extents + the GOAL MOUTHS, so you CAN state-set into
            # the net (a plain box blocked it). (RocketSim's Python API doesn't expose a mesh point
            # query, so the back wall + goal opening + crossbar are modeled from CommonValues.)
            is_ball = self.sel < 0
            rad = BALL_RADIUS if is_ball else 40.0
            SIDE, BACK, CEIL = 4096.0, 5120.0, 2044.0
            GOAL_HALF_X, GOAL_TOP_Z, NET_DEPTH = 892.755, 642.775, 5900.0
            zmin = 93.0 if is_ball else 17.0
            nx, ny, nz = p.x + dx, p.y + dy, p.z + dz
            nz = max(zmin, min(CEIL - rad, nz))
            # Past the back wall only through the goal mouth (within the posts + under the crossbar).
            can_enter_net = abs(nx) < (GOAL_HALF_X - rad) and nz < (GOAL_TOP_Z - rad)
            ymax = (NET_DEPTH - rad) if can_enter_net else (BACK - rad)
            ny = max(-ymax, min(ymax, ny))
            if abs(ny) > (BACK - rad):          # inside the net: stay within the goal frame width
                nx = max(-(GOAL_HALF_X - rad), min(GOAL_HALF_X - rad, nx))
            else:                                # on the field: full side-wall width
                nx = max(-(SIDE - rad), min(SIDE - rad, nx))
            _set_pos(ph, nx, ny, nz)
        self.send_pose()

    def add_vel(self, dx, dy, dz):
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            v = self._sel_phys(st).next_vel
            _set_vel(self._sel_phys(st), v.x + dx, v.y + dy, v.z + dz)
        self.send_pose()

    def rotate(self, dyaw, dpitch, droll):
        if self.sel < 0:
            return  # ball has no orientation
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            if not (0 <= self.sel < len(st.car_states)):
                return
            a = self.angles.setdefault(self.sel, [0.0, 0.0, 0.0])
            a[0] += dyaw; a[1] += dpitch; a[2] += droll
            self._apply_angle(st.car_states[self.sel], self.sel)
        self.send_pose()

    def reset_selected(self):
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            ph = self._sel_phys(st)
            if self.sel < 0:
                _set_pos(ph, 0, 0, BALL_RADIUS)
            else:
                _set_pos(ph, 0, 0, CAR_REST_Z)
                self.angles[self.sel] = [90.0, 0.0, 0.0]
                self._apply_angle(st.car_states[self.sel], self.sel)
            _set_vel(ph, 0, 0, 0)
        self.send_pose()

    def _ground_snap_check(self):
        # Holding Ctrl (down) on a grounded car for ~0.4s clips it flat to the floor (pitch/roll -> 0,
        # z -> 17), so you can't leave a car tilted/rolled while it's touching the ground.
        import time as _tt
        if self.sel >= 0 and _kp("ctrl"):
            with state_manager.global_state_mutex:
                st = state_manager.global_state_manager.state
                z = st.car_states[self.sel].phys.next_pos.z if self.sel < len(st.car_states) else 999.0
            if z <= 30.0:
                if self._ctrl_down_since is None:
                    self._ctrl_down_since = _tt.time()
                elif _tt.time() - self._ctrl_down_since >= 0.4:
                    self._snap_car_to_ground()
                    self._ctrl_down_since = _tt.time()
            else:
                self._ctrl_down_since = None
        else:
            self._ctrl_down_since = None

    def _snap_car_to_ground(self):
        if self.sel < 0:
            return
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            if self.sel >= len(st.car_states):
                return
            a = self.angles.setdefault(self.sel, [90.0, 0.0, 0.0])
            a[1] = 0.0; a[2] = 0.0   # pitch, roll flat
            self._apply_angle(st.car_states[self.sel], self.sel)
            ph = st.car_states[self.sel].phys
            _set_pos(ph, ph.next_pos.x, ph.next_pos.y, 17.0)
        self.send_pose()

    # ---- TEST mode: play from the posed state with real physics ----
    def toggle_test(self):
        if rs is None or not _ensure_rs_init():
            _log("test mode unavailable: RocketSim/meshes not ready")
            return
        self.test_mode = not self.test_mode
        if self.test_mode:
            self._snapshot_pose()
            self._build_test_arena()
        else:
            self.test_arena = None
            self.test_cars = []
            self.test_started = False
            self._restore_pose()

    def _snapshot_pose(self):
        with state_manager.global_state_mutex:
            self._pose_snapshot = copy.deepcopy(state_manager.global_state_manager.state)
        self._angles_snapshot = dict(self.angles)

    def _restore_pose(self):
        if self._pose_snapshot is not None:
            with state_manager.global_state_mutex:
                state_manager.global_state_manager.state = self._pose_snapshot
                self.recv_time_keepalive(state_manager.global_state_manager.state)
            self.angles = dict(self._angles_snapshot or {})

    def _build_test_arena(self):
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            ball = (st.ball_state.next_pos, st.ball_state.next_vel, st.ball_state.ang_vel)
            cars = [(c.team_num, c.phys.next_pos, c.phys.next_vel,
                     list(self.angles.get(i, [90.0, 0.0, 0.0])), c.boost_amount)
                    for i, c in enumerate(st.car_states)]
        arena = rs.Arena(rs.GameMode.SOCCAR)
        bs = arena.ball.get_state()
        bs.pos = rs.Vec(ball[0].x, ball[0].y, ball[0].z)
        bs.vel = rs.Vec(ball[1].x, ball[1].y, ball[1].z)
        bs.ang_vel = rs.Vec(ball[2].x, ball[2].y, ball[2].z)
        arena.ball.set_state(bs)
        self.test_cars = []
        for team_num, pos, vel, ang, boost in cars:
            car = arena.add_car(rs.Team.BLUE if team_num == 0 else rs.Team.ORANGE)
            cs = car.get_state()
            cs.pos = rs.Vec(pos.x, pos.y, pos.z)
            cs.vel = rs.Vec(vel.x, vel.y, vel.z)
            cs.ang_vel = rs.Vec(0, 0, 0)
            cs.rot_mat = rs.Angle(math.radians(ang[0]), math.radians(ang[1]),
                                  math.radians(ang[2])).as_rot_mat()
            cs.boost = float(boost)
            car.set_state(cs)
            self.test_cars.append(car)
        self.test_arena = arena
        self.test_started = False
        # You drive the SELECTED car (or the first one if the ball is selected).
        si = self.sel if 0 <= self.sel < len(self.test_cars) else 0
        self._controlled = self.test_cars[si] if self.test_cars else None

    def _read_play_controls(self):
        thr = (1.0 if _bind_down("throttle") else 0.0) - (1.0 if _bind_down("reverse") else 0.0)
        steer = (1.0 if _bind_down("steer_right") else 0.0) - (1.0 if _bind_down("steer_left") else 0.0)
        roll = (1.0 if _bind_down("air_roll_right") else 0.0) - (1.0 if _bind_down("air_roll_left") else 0.0)
        jump = _bind_down("jump"); boost = _bind_down("boost"); hb = _bind_down("powerslide")
        cc = rs.CarControls()
        cc.throttle, cc.steer = thr, steer
        cc.pitch, cc.yaw, cc.roll = -thr, steer, roll
        cc.jump, cc.boost, cc.handbrake = jump, boost, hb
        any_input = bool(thr or steer or roll or jump or boost or hb)
        return cc, any_input

    def _ball_reset(self, dribble):
        """RL freeplay ball reset: dribble = on top of the car, else ~100uu in front on the ground.
        Both inherit the car's velocity."""
        if self.test_arena is None or self._controlled is None:
            return
        cs = self._controlled.get_state()
        bs = self.test_arena.ball.get_state()
        if dribble:
            up = self._controlled.get_up_dir()
            bs.pos = rs.Vec(cs.pos.x + up.x * 100, cs.pos.y + up.y * 100, cs.pos.z + up.z * 100)
        else:
            fwd = self._controlled.get_forward_dir()
            bs.pos = rs.Vec(cs.pos.x + fwd.x * 220, cs.pos.y + fwd.y * 220, 93.0)
        bs.vel = rs.Vec(cs.vel.x, cs.vel.y, cs.vel.z)
        bs.ang_vel = rs.Vec(0, 0, 0)
        self.test_arena.ball.set_state(bs)

    def _test_tick(self):
        import time as _tt
        state_manager.pose_cam = None   # play with the normal chase camera while testing
        if self.test_arena is None:
            self._build_test_arena()
            if self.test_arena is None:
                return
        cc, any_input = self._read_play_controls()
        if self._edge("ball_front", _bind_down("ball_reset_front")):
            self._ball_reset(False); self.test_started = True
        if self._edge("ball_dribble", _bind_down("ball_reset_dribble")):
            self._ball_reset(True); self.test_started = True
        if not self.test_started and any_input:
            self.test_started = True   # the game begins on the first input
        if self._controlled is not None:
            self._controlled.set_controls(cc)
        # Step physics by REAL elapsed time (the poll fires ~120Hz but can jitter) so the sim runs at
        # true real time, not the poll rate. No physics at all until the first input (test_started).
        now = _tt.time()
        if self._last_test_time is None or not self.test_started:
            self._last_test_time = now
        if self.test_started:
            nsteps = int((now - self._last_test_time) * 120.0)
            nsteps = max(0, min(nsteps, 8))   # cap to avoid a spiral after a hitch
            self._last_test_time += nsteps / 120.0
            for _ in range(nsteps):
                self.test_arena.step(1)
        self._arena_to_state()

    def _arena_to_state(self):
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            bsim = self.test_arena.ball.get_state()
            _set_pos(st.ball_state, bsim.pos.x, bsim.pos.y, bsim.pos.z)
            _set_vel(st.ball_state, bsim.vel.x, bsim.vel.y, bsim.vel.z)
            st.ball_state.ang_vel = Vector3((bsim.ang_vel.x, bsim.ang_vel.y, bsim.ang_vel.z))
            for i, car in enumerate(self.test_cars):
                if i >= len(st.car_states):
                    break
                cs = car.get_state()
                ph = st.car_states[i].phys
                _set_pos(ph, cs.pos.x, cs.pos.y, cs.pos.z)
                _set_vel(ph, cs.vel.x, cs.vel.y, cs.vel.z)
                fwd, up = car.get_forward_dir(), car.get_up_dir()
                ph.prev_forward = ph.next_forward = Vector3((fwd.x, fwd.y, fwd.z))
                ph.prev_up = ph.next_up = Vector3((up.x, up.y, up.z))
                ph.has_rot = True
                st.car_states[i].boost_amount = cs.boost
                st.car_states[i].on_ground = bool(cs.is_on_ground)
                # Drive the boost-flame ribbon: boosting = boost input held AND boost remaining.
                st.car_states[i].is_boosting = bool(getattr(cs.last_controls, "boost", False)) and cs.boost > 0
            pads = self.test_arena.get_boost_pads()
            st.boost_pad_states = [bool(p.get_state().is_active) for p in pads]
            self.recv_time_keepalive(st)

    # ---- HUD + send ----
    def hud_text(self):
        if not self.active:
            return ""
        if self.test_mode:
            status = "press any control to START" if not self.test_started else "PLAYING"
            return ("\n######  TEST MODE — " + status + "  ######   [V = back to editing]\n"
                    "Drive: Z/S throttle  Q/D steer  A/E air-roll  L-click boost  R-click jump\n"
                    "L-Shift powerslide  Space ball-cam   1/2 ball reset (front/dribble)")
        who = "BALL" if self.sel < 0 else f"CAR {self.sel}"
        return ("\n######  EDIT MODE  ######   [V = test-drive,  B = exit]\n"
                f"Setting: {who}   (Left-click / N = switch object)\n"
                "Move: Z/S fwd-back  Q/D left-right  Shift/Ctrl up-down\n"
                "Rotate: arrows pan/pitch  A/E roll    Scroll = speed")

    def send_pose(self):
        with state_manager.global_state_mutex:
            st = state_manager.global_state_manager.state
            b = st.ball_state
            cars = []
            for i, c in enumerate(st.car_states):
                fwd, up = angle_to_fwd_up(*self.angles.get(i, [0.0, 0.0, 0.0]))
                cars.append({
                    "team": int(c.team_num),
                    "pos": [float(c.phys.next_pos.x), float(c.phys.next_pos.y), float(c.phys.next_pos.z)],
                    "vel": [float(c.phys.next_vel.x), float(c.phys.next_vel.y), float(c.phys.next_vel.z)],
                    "angle": list(self.angles.get(i, [0.0, 0.0, 0.0])),
                    "angvel": [0.0, 0.0, 0.0],
                    "boost": float(c.boost_amount),
                    "on_ground": bool(c.on_ground),
                })
            out = {
                "type": "pose",
                "ball": {
                    "pos": [float(b.next_pos.x), float(b.next_pos.y), float(b.next_pos.z)],
                    "vel": [float(b.next_vel.x), float(b.next_vel.y), float(b.next_vel.z)],
                    "angvel": [float(b.ang_vel.x), float(b.ang_vel.y), float(b.ang_vel.z)],
                },
                "cars": cars,
                "selected": self.sel,
            }
        try:
            self.sock.sendto(json.dumps(out).encode(), (EDITOR_IP, EDITOR_PORT))
        except Exception:
            pass


# Single shared instance used by main.py's keyPressEvent / render HUD.
g_pose_editor = PoseEditor()

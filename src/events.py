"""Game-event detection for RocketSimVis sounds / effects.

Senders only stream STATE (positions, velocities, on_ground, has_flip, ball_touched, pads...), so the
Rocket League events we want to hear/see are reconstructed here by diffing consecutive packets:

  jump         on_ground -> airborne with an impulse along the car's up axis
  doublejump   airborne, has_flip true->false, impulse mostly along up
  dodge        airborne, has_flip true->false, impulse mostly sideways/forward (a flip)
  flipreset    airborne, has_flip false->true with the ball in reach (RL's rule: wheels on the BALL)
  land         airborne -> on_ground, strength = speed into the surface
  ball_hit     a car's ball_touched flag (fallback: ball velocity jump with a car in reach)
  ball_bounce  ball velocity jump with no car touch (floor / wall / ceiling)
  bump         two cars in contact and both velocities jump (no ball touch)
  body         a car hits the arena with its body (roof / side / nose), not its wheels
  demo         is_demoed false->true
  pad          boost pad active->inactive
  goal         ball fully crossed a goal line
  supersonic   speed crosses 2200 uu/s upward

Runs on the SOCKET thread right after each packet is applied; events go into a deque that the render
thread drains (deque append/popleft are atomic). Each event carries the packet receive time so the
renderer can fire it when its interpolation actually reaches that packet (sound lines up with the
picture instead of leading it by one network interval).

Episode resets / state-setting teleports are detected and produce NO events (a reset would otherwise
look like 6 simultaneous demos, landings and pad pickups).
"""
import math
import os
from collections import deque

GOAL_Y = 5124.25 + 91.25        # ball centre past this |y| = fully in the goal
GOAL_HALF_W = 893.0 + 91.25
GOAL_H = 642.775 + 91.25
SUPERSONIC = 2200.0
TELEPORT = 900.0                # uu per packet -- same threshold as PhysState.is_teleporting

event_queue = deque(maxlen=512)

# RSV_EVENT_LOG=1 -> every flip-reset candidate (granted or rejected, with the reason) is appended to
# src/rsv_events.log so a miss on live data can be diagnosed from the exact streamed flags.
_EVENT_LOG = None
if os.environ.get("RSV_EVENT_LOG"):
    _EVENT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_events.log")


def _log(line):
    try:
        with open(_EVENT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _v(j, key, default=(0.0, 0.0, 0.0)):
    v = j.get(key)
    if v is None:
        return default
    return (float(v[0]), float(v[1]), float(v[2]))


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _len(a): return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


class _Car:
    __slots__ = ("pos", "vel", "ang", "up", "fwd", "on_ground", "has_flip", "demoed", "touched", "team", "speed",
                 "is_flipping", "world_contact", "jump", "stick", "stall")


def _car_snapshot(jc):
    c = _Car()
    ph = jc.get("phys", {})
    c.pos = _v(ph, "pos"); c.vel = _v(ph, "vel"); c.ang = _v(ph, "ang_vel")
    c.up = _v(ph, "up", (0.0, 0.0, 1.0)); c.fwd = _v(ph, "forward", (1.0, 0.0, 0.0))
    c.on_ground = bool(jc.get("on_ground", False))
    hf = jc.get("has_flip")
    if hf is None and jc.get("has_flipped_or_double_jumped") is not None:
        hf = not bool(jc["has_flipped_or_double_jumped"]) or c.on_ground
    c.has_flip = None if hf is None else bool(hf)
    c.demoed = bool(jc.get("is_demoed", False))
    t = jc.get("ball_touched")
    c.touched = None if t is None else bool(t)
    c.team = int(jc.get("team_num", 0))
    c.speed = _len(c.vel)
    f = jc.get("is_flipping")
    c.is_flipping = None if f is None else bool(f)
    w = jc.get("world_contact")
    c.world_contact = None if w is None else bool(w)
    ctl = jc.get("controls")
    if isinstance(ctl, dict) and ctl.get("jump") is not None:
        c.jump = bool(ctl["jump"])
        pitch, yaw, roll = float(ctl.get("pitch", 0)), float(ctl.get("yaw", 0)), float(ctl.get("roll", 0))
        c.stick = abs(pitch) + abs(yaw) + abs(roll)
        # a stall: RocketSim's dodge direction (-pitch, yaw + roll) is ~zero (air roll against yaw), so the
        # flip branch runs but the car gets no impulse and no rotation
        c.stall = abs(yaw + roll) < 0.1 and abs(pitch) < 0.1
    else:
        c.jump, c.stick, c.stall = None, 0.0, False
    return c


# ---- contact points on the car body (sparks): RLBot's Octane hitbox, offset from the car origin ----
HITBOX_HALF = (59.0, 42.1, 18.08)
HITBOX_OFF = (13.88, 0.0, 20.75)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _box_frame(c):
    f, u = c.fwd, c.up
    r = _cross(u, f)
    ctr = tuple(c.pos[k] + f[k] * HITBOX_OFF[0] + u[k] * HITBOX_OFF[2] for k in range(3))
    return ctr, (f, r, u)


def box_contact(c, point):
    """The point of car c's hitbox closest to `point`, and whether it is on the car's UNDERSIDE (the wheels'
    face) -> (contact, underside)."""
    ctr, axes = _box_frame(c)
    d = _sub(point, ctr)
    loc = [_dot(d, a) for a in axes]
    cl = [max(-h, min(h, l)) for l, h in zip(loc, HITBOX_HALF)]
    contact = tuple(ctr[k] + sum(axes[i][k] * cl[i] for i in range(3)) for k in range(3))
    excess = [abs(l) - h for l, h in zip(loc, HITBOX_HALF)]      # which face the point is beyond the most
    face = max(range(3), key=lambda i: excess[i])
    return contact, face == 2 and loc[2] < 0.0


def box_surface_contact(c, n):
    """The hitbox corner that is deepest along -n (the surface normal): where the body meets that surface."""
    ctr, axes = _box_frame(c)
    return tuple(ctr[k] - sum(axes[i][k] * HITBOX_HALF[i] * (1.0 if _dot(axes[i], n) > 0 else -1.0)
                              for i in range(3)) for k in range(3))


# ---- flip-reset grant: port of GigaLearnCPP's training-side flip-reset rule (GrantedFlipReset) ----
BALL_R = 91.25
SIDE_WALL_X, BACK_WALL_Y, GOAL_HALF_WIDTH, GOAL_HEIGHT = 4096.0, 5120.0, 893.0, 642.775
RESET_MIN_VERTICAL_SURFACE_DIST, RESET_MIN_BALL_Z = 150.0, 150.0


def _nearest_pinch_surface_dist(b):
    """NearestResetPinchSurface: ball-centre distance to the nearest solid VERTICAL surface (side wall,
    rounded 1152uu corner, backboard -- which is absent inside the goal mouth). None = none valid."""
    CORNER_R = 1152.0
    ax, ay = abs(b[0]), abs(b[1])
    corner_x, corner_y = SIDE_WALL_X - CORNER_R, BACK_WALL_Y - CORNER_R
    best = None
    if ay <= corner_y:
        best = SIDE_WALL_X - ax
    solid_backboard = b[2] >= GOAL_HEIGHT - BALL_R or ax >= GOAL_HALF_WIDTH - BALL_R
    if ax <= corner_x and solid_backboard:
        d = BACK_WALL_Y - ay
        best = d if best is None else min(best, d)
    dx, dy = ax - corner_x, ay - corner_y
    if dx > 0.0 and dy > 0.0:
        r = math.sqrt(dx * dx + dy * dy)
        if r > 1e-3:
            d = CORNER_R - r
            best = d if best is None else min(best, d)
    return best


def _reset_surface_eligible(b):
    d = _nearest_pinch_surface_dist(b)
    return b[2] >= RESET_MIN_BALL_Z and (d is None or d >= RESET_MIN_VERTICAL_SURFACE_DIST)


def _wheels_on_ball(c, ball_pos):
    d = _sub(ball_pos, c.pos)
    n = _len(d)
    if n > BALL_R + 110.0 or n < 1e-3:
        return False
    return -_dot(d, c.up) / n > 0.35                  # ball is below the car's floor pan


# arena planes (inward normal, offset) -- same set the wheel rig uses
_S2 = 1.0 / math.sqrt(2.0)
_PLANES = [((0.0, 0.0, 1.0), 0.0), ((0.0, 0.0, -1.0), -2044.0),
           ((-1.0, 0.0, 0.0), -4096.0), ((1.0, 0.0, 0.0), -4096.0),
           ((0.0, -1.0, 0.0), -5120.0), ((0.0, 1.0, 0.0), -5120.0),
           ((-_S2, -_S2, 0.0), -8064.0 * _S2), ((_S2, -_S2, 0.0), -8064.0 * _S2),
           ((-_S2, _S2, 0.0), -8064.0 * _S2), ((_S2, _S2, 0.0), -8064.0 * _S2)]


def _wheels_on_arena(c):
    """Car parked on floor/wall/ceiling: origin within 60uu of a plane with the roof pointing away
    from it. (The back wall is open inside the goal mouth.)"""
    x, y, z = c.pos
    in_mouth = abs(x) < 893.0 and z < 642.0
    for (nx, ny, nz), off in _PLANES:
        if in_mouth and nx == 0.0 and ny != 0.0:
            continue
        if nx * x + ny * y + nz * z - off < 60.0 and nx * c.up[0] + ny * c.up[1] + nz * c.up[2] > 0.6:
            return True
    return False


def _on_real_surface(c, ball_pos):
    """OnRealSurface: wheels on floor/wall/ceiling, NOT on the ball (isOnGround is also true for
    wheels resting on the ball -- that is 62.5% of real resets). GigaLearn's RenderSender does not
    stream world_contact, so arena-plane proximity stands in for it."""
    if c.world_contact is not None and c.world_contact:
        return True
    if not c.on_ground:
        return False
    return _len(_sub(ball_pos, c.pos)) > BALL_R + 130.0 or _wheels_on_arena(c)


# ---- "the reset came from the BALL, for sure" ------------------------------------------------ #
# Signed distance from a point to the arena shell INCLUDING its curved transitions (floor<->wall
# radius ~300, wall<->ceiling ~480, fitted to ArenaMeshCustom.obj) -- the flat planes above miss a
# car sitting on a curve, which is exactly where "reset on the wall while touching the ball" came from.
_FILLET_FLOOR, _FILLET_CEIL = 300.0, 480.0


def _fillet(h, v, r):
    if h < r and v < r:
        return r - math.hypot(r - h, r - v)
    return min(h, v)


def heading_into_goal(p, v, team, horizon=1.6, dt=1.0 / 60.0):
    """Would a free ball (gravity, floor bounces, side-wall bounces) cross `team`'s goal line inside the mouth
    within `horizon` s? Blue (0) defends -y, orange (1) +y."""
    sgn = -1.0 if (int(team) & 1) == 0 else 1.0
    x, y, z = p
    vx, vy, vz = v
    if vy * sgn <= 0.0:
        return False
    for _ in range(int(horizon / dt)):
        vz -= 650.0 * dt
        x += vx * dt; y += vy * dt; z += vz * dt
        if z < 91.25 and vz < 0.0:
            z, vz = 91.25, -vz * 0.6
        if abs(x) > 4096.0 - 91.25:
            vx = -vx
        if y * sgn >= 5124.25:                                   # reaches the goal line: inside the mouth?
            return abs(x) < 893.0 - 30.0 and z < 642.775 - 30.0
    return False


def arena_distance(p):
    x, y, z = abs(p[0]), abs(p[1]), p[2]
    h = min(4096.0 - x, (8064.0 - x - y) * 0.70710678)
    if not (x < 893.0 and z < 642.0):            # the back wall is open inside the goal mouth
        h = min(h, 5120.0 - y)
    return min(_fillet(h, z, _FILLET_FLOOR), _fillet(h, 2044.0 - z, _FILLET_CEIL))


def arena_normal(p, eps=4.0):
    """Unit normal (pointing into the field) of the closest arena surface, curves included."""
    gx = arena_distance((p[0] + eps, p[1], p[2])) - arena_distance((p[0] - eps, p[1], p[2]))
    gy = arena_distance((p[0], p[1] + eps, p[2])) - arena_distance((p[0], p[1] - eps, p[2]))
    gz = arena_distance((p[0], p[1], p[2] + eps)) - arena_distance((p[0], p[1], p[2] - eps))
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    return (0.0, 0.0, 1.0) if n < 1e-6 else (gx / n, gy / n, gz / n)


def goal_frame_distance(p):
    """Distance from a point to the goal frame (both posts and the crossbar, both goals): the edges where
    the back wall meets the goal mouth (|x| = 893, |y| = 5120, z <= 642.775 and z = 642.775, |x| <= 893)."""
    x, y, z = abs(p[0]), abs(p[1]), p[2]
    dy = y - BACK_WALL_Y
    post = math.sqrt((x - GOAL_HALF_WIDTH) ** 2 + dy * dy + max(0.0, z - GOAL_HEIGHT) ** 2)
    bar = math.sqrt(max(0.0, x - GOAL_HALF_WIDTH) ** 2 + dy * dy + (z - GOAL_HEIGHT) ** 2)
    return min(post, bar)


def bounce_goal_frame_distance(pa, va, pb, vb, dt, steps=8):
    """Closest approach to the goal frame around a bounce between two packets dt apart: along the straight
    chord AND along the two velocity rays (forward from the previous packet, backward from this one). A fast
    ball bounces between packets, so the chord cuts the corner and misses the post it actually hit."""
    d = swept_goal_frame_distance(pa, pb, steps)
    if 0.0 < dt < 0.2:
        for k in range(1, steps + 1):
            f = dt * k / steps
            d = min(d, goal_frame_distance((pa[0] + va[0] * f, pa[1] + va[1] * f, pa[2] + va[2] * f)),
                    goal_frame_distance((pb[0] - vb[0] * f, pb[1] - vb[1] * f, pb[2] - vb[2] * f)))
    return d


def swept_goal_frame_distance(a, b, steps=8):
    """Closest approach of the ball centre to the goal frame between two packets (a 2000 uu/s ball moves
    ~65 uu per 30 Hz packet, so the packet positions alone can straddle the actual contact)."""
    return min(goal_frame_distance((a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps,
                                    a[2] + (b[2] - a[2]) * k / steps)) for k in range(steps + 1))


POST_CONTACT = BALL_R + 30.0         # ball centre this close to a post/crossbar edge = it hit the frame
BODY_NEAR = 120.0                     # car origin within this of a surface can be touching it with its body
BODY_MIN_DV = 300.0                   # into-surface speed lost in one packet for a body impact


# Calibrated on 118 grants in 100 real bot pop-reset clips: ball centre 89..105 uu below the car's
# origin along -up (p95 lateral offset 47, max 88), car >= 198 uu from every arena surface.
RESET_ALONG = (80.0, 125.0)
RESET_LATERAL = 70.0
RESET_CAR_CLEARANCE = 120.0          # a wheel reaches ~80 uu from the car origin


def ball_under_wheels(c, ball_pos):
    d = _sub(ball_pos, c.pos)
    along = -_dot(d, c.up)
    if not (RESET_ALONG[0] <= along <= RESET_ALONG[1]):
        return False
    lat = (d[0] + c.up[0] * along, d[1] + c.up[1] * along, d[2] + c.up[2] * along)
    return _len(lat) <= RESET_LATERAL


def granted_flip_reset(c, p, bpos, pbpos, dodge_now):
    """-> (granted, reason). The step this car EARNS a flip reset off the ball: the training Learner's
    PopGranted (GigaLearnCPP Learner.cpp) -- flip availability rises (or a flip starts with no flip =
    reset spent instantly) while the ball was touched this/last step and the car is not on real
    geometry. NOTE: the reward-side WallPinch/150uu surface rule is deliberately NOT applied: it is a
    reward-shaping exclusion, and RL shows the indicator for resets next to walls/backboard too (that
    rule is what made the vis miss most live resets)."""
    if c.has_flip is None or p.has_flip is None:
        return False, "no has_flip field"
    rise = c.has_flip and not p.has_flip
    if c.is_flipping is not None and p.is_flipping is not None:
        instant = (not p.has_flip) and c.is_flipping and not p.is_flipping
    else:
        instant = (not p.has_flip) and (not c.has_flip) and dodge_now
    if not (rise or instant):
        return False, None
    # RocketSim only registers a ball hit (ball_touched) for car-BODY collisions: wheels resting on
    # the ball are raycasts and never set it, and a clean reset is often wheels-only. So the streamed
    # flag is OR-ed with geometry: ball close and under the car's floor pan.
    touched = bool(c.touched) or bool(p.touched) or _wheels_on_ball(c, bpos) or _wheels_on_ball(p, pbpos)
    if not touched:
        return False, "ball not touched this/last step"
    if _on_real_surface(c, bpos) or _on_real_surface(p, pbpos):
        return False, "car on real surface (floor/wall landing)"
    if bpos[2] < BALL_R + 25.0:
        return False, "ball on the floor"
    # Only a reset taken ON THE BALL, for sure: the ball must be right under the wheels (this step or
    # the last) and no wheel may be able to touch any arena surface (wall/curve/ceiling resets while
    # also touching the ball used to pass).
    if not (ball_under_wheels(c, bpos) or ball_under_wheels(p, pbpos)):
        return False, "ball not under the wheels"
    if min(arena_distance(c.pos), arena_distance(p.pos)) < RESET_CAR_CLEARANCE:
        return False, "car close enough to the arena to reset on it"
    return True, "instant" if instant else "rise"


class EventDetector:
    def __init__(self):
        self.prev_cars = None
        self.prev_ball = None       # (pos, vel)
        self.prev_pads = None
        self._pair_cool = {}
        self._reset_cool = {}
        self._body_cool = {}
        self.last_touch_team = None     # team of the last car to touch the ball (ball trail colour)
        self._goal_cool_until = 0.0
        self._save_cool_until = 0.0

    def reset(self):
        self.prev_cars = None
        self.prev_ball = None
        self.prev_pads = None

    def process(self, j, pad_locations, t):
        """j = the raw packet dict (already validated by GameState.read_from_json)."""
        try:
            self._process(j, pad_locations, t)
        except Exception as e:           # never let event detection break the socket thread
            print("[events] error: {!r}".format(e))
            self.reset()

    def _emit(self, t, kind, pos, **kw):
        ev = {"t": t, "kind": kind, "pos": pos}
        ev.update(kw)
        event_queue.append(ev)

    def _process(self, j, pad_locations, t):
        jb = j.get("ball_phys") or {}
        bpos, bvel = _v(jb, "pos"), _v(jb, "vel")
        cars = [_car_snapshot(jc) for jc in j.get("cars", [])]
        pads = j.get("boost_pad_states")

        pc, pb = self.prev_cars, self.prev_ball
        self.prev_cars, self.prev_ball = cars, (bpos, bvel)
        prev_t, self._prev_t = getattr(self, "_prev_t", t), t
        self._prev_t_last = prev_t
        prev_pads, self.prev_pads = self.prev_pads, (list(pads) if pads is not None else None)
        if pc is None or pb is None or len(pc) != len(cars):
            return

        # ---- reset / teleport guard ----
        ball_tp = _len(_sub(bpos, pb[0])) > TELEPORT
        car_tp = [(_len(_sub(c.pos, p.pos)) > TELEPORT) for c, p in zip(cars, pc)]
        if ball_tp or (cars and sum(car_tp) * 2 >= len(cars)):
            self.last_touch_team = None                     # kickoff / reset: nobody has touched it yet
            return                                          # episode reset / state-set: no events

        # ---- ball ----
        dvb = _sub(bvel, pb[1])
        dvb_mag = _len((dvb[0], dvb[1], dvb[2] + 21.7))     # remove ~1 packet of gravity
        touchers = [i for i, c in enumerate(cars) if c.touched and not c.demoed]
        have_touch_flag = any(c.touched is not None for c in cars)
        if not have_touch_flag and dvb_mag > 250.0:
            near = [(i, _len(_sub(c.pos, bpos))) for i, c in enumerate(cars) if not c.demoed]
            near = [x for x in near if x[1] < 260.0]
            if near:
                touchers = [min(near, key=lambda x: x[1])[0]]
        if touchers:
            i = touchers[0]
            self.last_touch_team = cars[i].team
            contact, under = box_contact(cars[i], bpos)
            self._emit(t, "ball_hit", bpos, car=i, team=cars[i].team, strength=dvb_mag, contact=contact,
                       underside=under)
            # a save: the defending team touched a ball that was going in, and now it isn't
            tm = int(cars[i].team) & 1
            if (t >= self._save_cool_until and heading_into_goal(pb[0], pb[1], tm)
                    and not heading_into_goal(bpos, bvel, tm)):
                self._save_cool_until = t + 1.5
                self._emit(t, "save", bpos, car=i, team=tm)
        elif dvb_mag > 300.0:
            if bounce_goal_frame_distance(pb[0], pb[1], bpos, bvel, t - self._prev_t_last) < POST_CONTACT:
                surf = "post"                               # goal post OR crossbar clang
            elif bpos[2] < 180.0:
                surf = "floor"
            else:
                surf = "wall"
            self._emit(t, "ball_bounce", bpos, surface=surf, strength=dvb_mag)

        # ---- goal ----
        if (abs(bpos[1]) > GOAL_Y >= abs(pb[0][1]) and abs(bpos[0]) < GOAL_HALF_W and bpos[2] < GOAL_H
                and t >= self._goal_cool_until):
            self._goal_cool_until = t + 3.0
            self._emit(t, "goal", bpos, team=0 if bpos[1] > 0 else 1)

        # ---- cars ----
        demo_now = set()
        self_impulse = set()           # cars whose velocity jump is their own jump/dodge/landing
        for i, (c, p) in enumerate(zip(cars, pc)):
            if car_tp[i]:
                continue
            if c.demoed:
                if not p.demoed:
                    demo_now.add(i)
                    # the demolisher: the closest other active car at the moment of the demo
                    near = [(q, _len(_sub(pc[q].pos, p.pos))) for q in range(len(cars))
                            if q != i and not cars[q].demoed and not pc[q].demoed]
                    near = [x for x in near if x[1] < 350.0]
                    by = min(near, key=lambda x: x[1])[0] if near else -1
                    self._emit(t, "demo", p.pos, car=i, team=c.team, vel=p.vel, by=by)
                continue
            if p.demoed:
                continue                                    # respawn
            dv = _sub(c.vel, p.vel)
            dv_up = _dot(dv, p.up)
            dv_h = _len(_sub(dv, (p.up[0] * dv_up, p.up[1] * dv_up, p.up[2] * dv_up)))

            # dodge signature (no is_flipping field): a flip adds ~500 uu/s along the car plane and a
            # sudden spin
            dodge_now = dv_h > 250.0 and _len(_sub(c.ang, p.ang)) > 3.5
            granted, why = granted_flip_reset(c, p, bpos, pb[0], dodge_now)
            if why is not None and _EVENT_LOG is not None:
                _log("t=%.3f car=%d granted=%s reason=%s on_ground=%s->%s has_flip=%s->%s touched=%s->%s "
                     "dist_ball=%.0f ball=(%.0f,%.0f,%.0f)" % (t, i, granted, why, p.on_ground, c.on_ground,
                     p.has_flip, c.has_flip, p.touched, c.touched, _len(_sub(bpos, c.pos)), bpos[0], bpos[1], bpos[2]))
            if granted and t >= self._reset_cool.get(i, 0.0):
                self._reset_cool[i] = t + 0.35
                to_car = _sub(c.pos, bpos)
                n = max(_len(to_car), 1e-3)
                nrm = (to_car[0] / n, to_car[1] / n, to_car[2] / n)
                contact = (bpos[0] + nrm[0] * BALL_R, bpos[1] + nrm[1] * BALL_R, bpos[2] + nrm[2] * BALL_R)
                self._emit(t, "flipreset", contact, car=i, team=c.team, up=c.up, normal=nrm,
                           car_pos=c.pos, ball_pos=bpos)
                if not p.has_flip and not c.has_flip:            # instantly spent: it was a flip too
                    self_impulse.add(i)
                    self._emit(t, "dodge", c.pos, car=i, team=c.team, up=c.up)
                    continue

            p_srf = _on_real_surface(p, pb[0])       # real floor/wall/ceiling contact, NOT wheels on the ball
            c_srf = _on_real_surface(c, bpos)
            if c.jump is not None and p.jump is not None:
                # The sender streams the input that PRODUCED this packet (GigaLearn: player.prevAction), so
                # a new jump press is a real jump/flip even when the car is back on the wall by the next
                # packet (wall dash: jump + dodge + re-land within one or two packets -- the state-diff
                # rules below never see it). Same rules as RocketSim: on the ground -> jump; airborne
                # with a flip left -> dodge if the stick is past the 0.5 deadzone, else double jump.
                if c.jump and not p.jump:
                    if p.on_ground:
                        self_impulse.add(i)
                        self._emit(t, "jump", c.pos, car=i, team=c.team, up=p.up)
                    elif p.has_flip:
                        self_impulse.add(i)
                        self._emit(t, "dodge" if c.stick >= 0.5 else "doublejump", c.pos, car=i, team=c.team, up=c.up,
                                   stall=bool(c.stall))
                if (not p_srf) and c_srf:
                    into = -_dot(p.vel, c.up)
                    if into > 140.0:
                        self_impulse.add(i)
                        self._emit(t, "land", c.pos, car=i, team=c.team, strength=into)
            elif p_srf and not c_srf:
                if dv_up > 170.0 and not c.touched:
                    self_impulse.add(i)
                    self._emit(t, "jump", c.pos, car=i, team=c.team, up=c.up)
            elif not p_srf and not c_srf:
                if c.has_flip is not None and p.has_flip is not None:
                    if p.has_flip and not c.has_flip:
                        if dv_h > 250.0:
                            self_impulse.add(i)
                            self._emit(t, "dodge", c.pos, car=i, team=c.team, up=c.up)
                        elif dv_up > 140.0:
                            self_impulse.add(i)
                            self._emit(t, "doublejump", c.pos, car=i, team=c.team, up=c.up)
                elif not c.touched and _len(_sub(c.ang, p.ang)) > 4.5 and _len(dv) > 300.0:
                    self_impulse.add(i)
                    self._emit(t, "dodge", c.pos, car=i, team=c.team, up=c.up)   # no has_flip field
            elif (not p_srf) and c_srf:
                into = -_dot(p.vel, c.up)
                if into > 140.0:
                    self_impulse.add(i)
                    self._emit(t, "land", c.pos, car=i, team=c.team, strength=into)

            # body impact: the car hits the arena (floor / wall / ceiling / curve) with its roof, side, nose or
            # tail -- a big loss of into-surface speed near a surface that is NOT a landing on the wheels
            if i not in self_impulse and not c.touched and t >= self._body_cool.get(i, 0.0):
                d_srf = arena_distance(c.pos)
                if d_srf < BODY_NEAR:
                    n = arena_normal(c.pos)
                    vin_p, vin_c = -_dot(p.vel, n), -_dot(c.vel, n)
                    # tilted < ~70 deg before the hit: the wheels touch first -> a landing, not a body hit. (Was
                    # 45 deg: tilted wheel landings at 45-50 deg played the body thud on top of the landing sound.)
                    wheels_down = _dot(p.up, n) > 0.35
                    if vin_p > BODY_MIN_DV and vin_p - vin_c > BODY_MIN_DV and not wheels_down:
                        self._body_cool[i] = t + 0.3
                        self_impulse.add(i)
                        corner = box_surface_contact(c, n)
                        dc = max(0.0, arena_distance(corner))
                        contact = (corner[0] - n[0] * dc, corner[1] - n[1] * dc, corner[2] - n[2] * dc)
                        self._emit(t, "body", contact, car=i, team=c.team, strength=vin_p - vin_c, normal=n)

            if c.speed >= SUPERSONIC > p.speed:
                self._emit(t, "supersonic", c.pos, car=i, team=c.team)

        # ---- bumps (car vs car) ----
        n = len(cars)
        for a in range(n):
            ca, pa = cars[a], pc[a]
            if ca.demoed or pa.demoed or car_tp[a] or ca.touched:
                continue
            for b in range(a + 1, n):
                cb, pbb = cars[b], pc[b]
                if cb.demoed or pbb.demoed or car_tp[b] or cb.touched or a in demo_now or b in demo_now:
                    continue
                if a in self_impulse or b in self_impulse:
                    continue
                sep = _sub(ca.pos, cb.pos)
                if _len(sep) > 240.0:
                    continue
                dva_v, dvb_v = _sub(ca.vel, pa.vel), _sub(cb.vel, pbb.vel)
                dva, dvb2 = _len(dva_v), _len(dvb_v)
                if max(dva, dvb2) < 150.0:      # a light push is ~170; a car's own braking is <= ~120
                    continue
                if _dot(_sub(dva_v, dvb_v), sep) <= 0.0:       # a real contact pushes them apart
                    continue
                if t < self._pair_cool.get((a, b), 0.0):
                    continue
                self._pair_cool[(a, b)] = t + 0.3
                mid = ((ca.pos[0] + cb.pos[0]) / 2, (ca.pos[1] + cb.pos[1]) / 2, (ca.pos[2] + cb.pos[2]) / 2)
                # where the two bodies meet: each car's hitbox point closest to the other's hitbox centre;
                # sparks only when it is body on body (not one car's wheels landing on the other)
                pa_c, under_a = box_contact(ca, _box_frame(cb)[0])
                pb_c, under_b = box_contact(cb, _box_frame(ca)[0])
                contact = tuple((pa_c[k] + pb_c[k]) / 2 for k in range(3))
                self._emit(t, "bump", mid, car=a, other=b, strength=max(dva, dvb2), contact=contact,
                           spark=not (under_a or under_b), normal=_sub(pa_c, pb_c))

        # ---- boost pads ----
        if pads is not None and prev_pads is not None and len(pads) == len(prev_pads):
            for k, (was, now) in enumerate(zip(prev_pads, pads)):
                if was and not now and k < len(pad_locations):
                    loc = pad_locations[k]
                    big = float(loc[2]) == 73.0
                    pos = (float(loc[0]), float(loc[1]), 0.0)
                    grabber = min(range(n), key=lambda q: _len(_sub(cars[q].pos, pos))) if n else -1
                    self._emit(t, "pad", pos, big=big, car=grabber)


g_detector = EventDetector()

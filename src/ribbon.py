from collections import deque

# A new point at most this often (s): at a high frame rate the head point is moved to the emitter instead of a new one
# being added every frame -- the trail looks the same, with a fraction of the points to age, upload and draw.
MIN_EMIT_DT = 1.0 / 90.0


class RibbonPoint:
    __slots__ = ("pos", "vel", "t0", "connected", "k", "col", "_em")

    def __init__(self, em, pos, vel, t0):
        self._em = em
        self.pos = pos              # [x, y, z] (plain floats)
        self.vel = vel              # None = static (every trail here)
        self.t0 = t0
        self.connected = True

    @property
    def time_active(self):
        return self._em.clock - self.t0


class RibbonEmitter:
    def __init__(self):
        self.points = deque()
        self.time_since_emit = 0
        self.clock = 0.0
        self._last_emit = -1e9

    def update(self, can_emit, emit_delay, emit_pos, emit_vel, lifetime, delta_time):
        self.clock += delta_time
        if self.time_since_emit < emit_delay:
            self.time_since_emit += delta_time
            if len(self.points) > 0:
                self.points[0].connected = False
        elif can_emit:
            self.time_since_emit = 0
            pos = [float(emit_pos[0]), float(emit_pos[1]), float(emit_pos[2])]
            vx, vy, vz = float(emit_vel[0]), float(emit_vel[1]), float(emit_vel[2])
            vel = None if (vx == 0.0 and vy == 0.0 and vz == 0.0) else (vx, vy, vz)
            pts = self.points
            if (len(pts) >= 2 and pts[0].connected and vel is None and pts[0].vel is None
                    and self.clock - self._last_emit < MIN_EMIT_DT):
                head = pts[0]                         # too soon for a new point: drag the head along
                head.pos = pos
                head.t0 = self.clock
            else:
                pts.appendleft(RibbonPoint(self, pos, vel, self.clock))
                self._last_emit = self.clock

        # only moving points need integrating (every trail in the vis is static: vel None)
        for point in self.points:
            v = point.vel
            if v is not None:
                p = point.pos
                p[0] += v[0] * delta_time
                p[1] += v[1] * delta_time
                p[2] += v[2] * delta_time

        # Remove dead points
        clock = self.clock
        while len(self.points) > 0 and clock - self.points[-1].t0 > lifetime:
            self.points.pop()

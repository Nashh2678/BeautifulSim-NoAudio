from collections import deque

from pyrr import Vector3

class RibbonPoint:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.time_active = 0
        self.connected = True

class RibbonEmitter:
    def __init__(self):
        self.points = deque()
        self.time_since_emit = 0

    def update(self, can_emit, emit_delay, emit_pos, emit_vel, lifetime, delta_time):
        if self.time_since_emit < emit_delay:
            self.time_since_emit += delta_time
            if len(self.points) > 0:
                self.points[0].connected = False
        elif can_emit:
            self.time_since_emit = 0

            new_point = RibbonPoint(emit_pos, emit_vel)
            self.points.insert(0, new_point)

        # `pos += vel * delta_time` with pyrr Vector3 allocates two temporaries PER POINT PER FRAME,
        # for every ribbon (one per car + the ball). Each point's velocity is constant, so advancing
        # the position with plain float arithmetic is numerically the same and allocation-free.
        for point in self.points:
            vx, vy, vz = point.vel[0], point.vel[1], point.vel[2]
            p = point.pos
            p[0] += vx * delta_time
            p[1] += vy * delta_time
            p[2] += vz * delta_time
            point.time_active += delta_time

        # Remove dead points
        while len(self.points) > 0 and self.points[-1].time_active > lifetime:
            self.points.pop()
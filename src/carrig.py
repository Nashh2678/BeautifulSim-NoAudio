"""Per-car wheel rig: suspension travel, wheel spin and front-wheel steering for the Octane mesh.

Senders stream only the car's rigid-body state, so wheels are reconstructed the way RocketSim's
raycast vehicle places them: each wheel hangs from an anchor at local z = 20.755 (RocketSim Octane
connection point) along the car's -up axis; a ray from the anchor against the arena surfaces gives
the suspension length (clamped to the +-12 uu travel), so the body visibly sinks on landings /
wavedashes and wheels droop in the air. Spin = forward speed / radius; steering is estimated from the
measured yaw rate (Ackermann: steer = atan(yaw_rate * wheelbase / speed)).
"""
import math
import struct

import numpy as np

_PACK16 = struct.Struct("16f").pack

ANCHOR_Z = 20.755
TRAVEL = 12.0
WHEELBASE = 85.0
S2 = 1.0 / math.sqrt(2.0)
# arena surfaces as (inward normal, offset): n . p >= offset inside the arena
PLANES = [((0.0, 0.0, 1.0), 0.0), ((0.0, 0.0, -1.0), -2044.0),
          ((-1.0, 0.0, 0.0), -4096.0), ((1.0, 0.0, 0.0), -4096.0),
          ((0.0, -1.0, 0.0), -5120.0), ((0.0, 1.0, 0.0), -5120.0),
          ((-S2, -S2, 0.0), -8064.0 * S2), ((S2, -S2, 0.0), -8064.0 * S2),
          ((-S2, S2, 0.0), -8064.0 * S2), ((S2, S2, 0.0), -8064.0 * S2)]


class _CarWheels:
    __slots__ = ("susp", "spin", "omega", "steer", "prev_fwd")

    def __init__(self, rest):
        self.susp = [float(v) for v in rest]
        self.spin = [0.0] * 4
        self.omega = [0.0] * 4
        self.steer = 0.0
        self.prev_fwd = None


class WheelRig:
    def __init__(self, centers, radii):
        self.c = np.asarray(centers, "f8")          # (4,3) mesh wheel centres in car space (rest pose)
        self.r = np.asarray(radii, "f8")
        self.rest = ANCHOR_Z - self.c[:, 2]         # suspension length at the mesh's rest pose
        self.front = self.c[:, 0] > 0.0
        self.cars = {}
        # plain-float copies for the per-frame path
        self._cx = [float(v) for v in self.c[:, 0]]
        self._cy = [float(v) for v in self.c[:, 1]]
        self._r = [float(v) for v in self.r]
        self._rest = [float(v) for v in self.rest]
        self._front = [bool(v) for v in self.front]

    def matrices(self, key, pos, fwd, up, vel, on_ground, dt):
        """-> list of 4 column-major float32 model matrices (bytes) for the wheel meshes.
        Plain float math packed with struct: numpy's per-call overhead dominated on 4x4 matrices."""
        st = self.cars.get(key)
        if st is None:
            st = self.cars[key] = _CarWheels(self.rest)
        fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        lx, ly, lz = uy * fz - uz * fy, uz * fx - ux * fz, ux * fy - uy * fx
        dt = min(max(dt, 0.0), 0.1)
        vf = float(vel[0]) * fx + float(vel[1]) * fy + float(vel[2]) * fz

        # ---- steering estimate from the measured yaw rate ----
        pf = st.prev_fwd
        if pf is not None and dt > 1e-4:
            s_ = (pf[1] * fz - pf[2] * fy) * ux + (pf[2] * fx - pf[0] * fz) * uy + (pf[0] * fy - pf[1] * fx) * uz
            yaw_rate = math.atan2(s_, pf[0] * fx + pf[1] * fy + pf[2] * fz) / dt
        else:
            yaw_rate = 0.0
        st.prev_fwd = (fx, fy, fz)
        if on_ground:
            target = math.atan(yaw_rate * WHEELBASE / max(abs(vf), 250.0)) * (1.0 if vf >= 0 else -1.0)
            target = max(-0.5, min(0.5, target))
        else:
            target = 0.0
        st.steer += (target - st.steer) * (1.0 - math.exp(-dt / 0.08))
        ks = 1.0 - math.exp(-dt / 0.03)
        kw = math.exp(-dt * 0.6)

        out = []
        for k in range(4):
            cx, cy, R, rest = self._cx[k], self._cy[k], self._r[k], self._rest[k]
            ax = px + fx * cx + lx * cy + ux * ANCHOR_Z
            ay = py + fy * cx + ly * cy + uy * ANCHOR_Z
            az = pz + fz * cx + lz * cy + uz * ANCHOR_Z
            t = None
            for (nx, ny, nz), off in PLANES:
                nd = -(nx * ux + ny * uy + nz * uz)
                if nd > -0.25:
                    continue
                tt = (off - (nx * ax + ny * ay + nz * az)) / nd
                if tt > -20.0 and (t is None or tt < t):
                    t = tt
            hang = rest + 0.35 * TRAVEL
            if t is not None and t - R < rest + TRAVEL:
                L = min(max(t - R, rest - TRAVEL), hang)
                st.omega[k] = vf / R
            else:
                L = hang
                st.omega[k] *= kw
            st.susp[k] += (L - st.susp[k]) * ks
            st.spin[k] = (st.spin[k] + st.omega[k] * dt) % (2.0 * math.pi)

            g = st.steer if self._front[k] else 0.0
            cg, sg = math.cos(g), math.sin(g)
            ca, sa = math.cos(st.spin[k]), math.sin(st.spin[k])
            # RzRy columns (local wheel axes in car space)
            c0 = (cg * ca, sg * ca, -sa)
            c1 = (-sg, cg, 0.0)
            c2 = (cg * sa, sg * sa, ca)
            # world axis = f*c[0] + left*c[1] + up*c[2]
            def w(c):
                return (fx * c[0] + lx * c[1] + ux * c[2], fy * c[0] + ly * c[1] + uy * c[2],
                        fz * c[0] + lz * c[1] + uz * c[2])
            X, Y, Z = w(c0), w(c1), w(c2)
            hz = ANCHOR_Z - st.susp[k]
            tx = px + fx * cx + lx * cy + ux * hz
            ty = py + fy * cx + ly * cy + uy * hz
            tz = pz + fz * cx + lz * cy + uz * hz
            out.append(_PACK16(X[0], X[1], X[2], 0.0, Y[0], Y[1], Y[2], 0.0, Z[0], Z[1], Z[2], 0.0, tx, ty, tz, 1.0))
        return out

    def forget(self, key):
        self.cars.pop(key, None)

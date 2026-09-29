"""Boost pad meshes, generated: smooth lathed (turned) shapes at 48 (small) / 72 (big) segments instead of the 24-sided OBJs, with rounded
edges and a real sphere for the big pad's orb. Same object-space dimensions, attributes and conventions as the OBJs
(BoostPad_{Small,Big}_{0,1}.obj, drawn with PAD_VERT / PAD_FRAG at scale 2.5), so the pad shader's recharge / ghost /
orb logic keeps working unchanged. The shader tells the gold parts from the metal base by the texture's saturation:
the meshes sample a 2-texel texture, u = 0.25 metal, u = 0.75 gold (PAD_TEX)."""
import math

import numpy as np

SEG = 96
METAL, GOLD = 0.25, 0.75
# texel 0: dark gun-metal base, texel 1: gold
PAD_TEX = np.array([[[70, 70, 78, 255], [255, 158, 26, 255]]], "u1")


def _arc(cr, cz, rad, a0, a1, n, part):
    """Profile points on a circular arc (centre cr, cz), angles in degrees, normals pointing out of the arc."""
    out = []
    for k in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * k / n)
        out.append((cr + rad * math.cos(a), cz + rad * math.sin(a), math.cos(a), math.sin(a), part))
    return out


def _line(r0, z0, r1, z1, part, n=1):
    """A straight profile segment with its own (flat) normal: hard edges at its ends."""
    dr, dz = r1 - r0, z1 - z0
    L = math.hypot(dr, dz) or 1.0
    nr, nz = dz / L, -dr / L          # outward for a profile walked outer-bottom -> top -> inward
    return [(r0 + dr * k / n, z0 + dz * k / n, nr, nz, part) for k in range(n + 1)]


def _lathe(strips, seg=SEG):
    """strips: lists of (r, z, nr, nz, u) profile points; each strip is smooth along itself, strips meet at hard edges.
    Returns (N, 8) float32 triangles: pos(3) normal(3) uv(2), counter-clockwise seen from outside (like the OBJs)."""
    ang = np.linspace(0.0, 2.0 * math.pi, seg + 1)
    ca, sa = np.cos(ang)[None, :], np.sin(ang)[None, :]
    out = []
    for st in strips:
        st = np.asarray(st, "f8")                              # (n, 5)
        g = np.empty((len(st), seg + 1, 8))
        g[..., 0] = st[:, 0:1] * ca; g[..., 1] = st[:, 0:1] * sa; g[..., 2] = st[:, 1:2]
        g[..., 3] = st[:, 2:3] * ca; g[..., 4] = st[:, 2:3] * sa; g[..., 5] = st[:, 3:4]
        g[..., 6] = st[:, 4:5]; g[..., 7] = 0.5
        a0, a1, b1, b0 = g[:-1, :-1], g[:-1, 1:], g[1:, 1:], g[1:, :-1]
        for t in ((a0, a1, b1), (a0, b1, b0)):
            t = np.stack(t, 2).reshape(-1, 3, 8)
            cr = np.cross(t[:, 1, :3] - t[:, 0, :3], t[:, 2, :3] - t[:, 0, :3])
            keep = (cr * cr).sum(1) > 1e-10                     # drop the degenerate ones on the axis
            t, cr = t[keep], cr[keep]
            flip = (cr * t[:, :, 3:6].sum(1)).sum(1) < 0
            t[flip] = t[flip][:, ::-1]
            out.append(t.reshape(-1, 8))
    return np.concatenate(out, 0).astype("f4")


def _base(R, zb, zt, Ri, zf):
    """The metal ring: outer wall, rounded rim, flat top, rounded inner lip, inner wall, floor."""
    b = 1.1 if R < 30 else 1.6
    return [
        _line(R, zb, R, zt - b, METAL),
        _arc(R - b, zt - b, b, 0.0, 90.0, 5, METAL),
        _line(R - b, zt, Ri + b * 0.6, zt, METAL),
        _arc(Ri + b * 0.6, zt - b * 0.6, b * 0.6, 90.0, 180.0, 3, METAL),
        _line(Ri, zt - b * 0.6, Ri, zf, METAL),
        _line(Ri, zf, 0.0, zf, METAL),
    ]


def small(active):
    s = _base(20.8, -3.1, 3.1, 18.0, 1.7)
    if active:
        # gold dome-topped plate rising out of the ring
        s += [_line(17.6, -0.4, 12.99, 6.2, GOLD),
              _arc(11.6, 5.4, 1.6, 30.0, 90.0, 4, GOLD),
              _line(11.6, 7.0, 0.0, 7.0, GOLD)]
    return _lathe(s, 48)


ORB_CZ, ORB_R = 29.65, 12.75


def big(active):
    s = _base(37.8, -8.0, 3.3, 32.6, 0.7)
    if active:
        s += [_line(24.5, -1.6, 15.6, 5.9, GOLD),
              _arc(14.2, 5.0, 1.8, 30.0, 90.0, 4, GOLD),
              _line(14.2, 6.8, 0.0, 6.8, GOLD)]
        # the orb: a real sphere (normals radial)
        s += [_arc(0.0, ORB_CZ, ORB_R, -90.0, 90.0, 28, GOLD)]
    return _lathe(s, 72)


def all_meshes():
    """name -> (N, 8) float32 triangle vertices, keyed like the OBJs they replace."""
    return {"BoostPad_Small_0.obj": small(False), "BoostPad_Small_1.obj": small(True),
            "BoostPad_Big_0.obj": big(False), "BoostPad_Big_1.obj": big(True)}

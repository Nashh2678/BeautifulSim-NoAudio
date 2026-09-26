"""Speed up pyrr.Vector3 arithmetic (import this before anything does vector math).

pyrr routes EVERY Vector3 operator (+ - * / neg, .length, .normalized) through `multipledispatch`,
which costs ~10-15 us per operation. The renderer does thousands of these per frame (interpolation,
camera, model matrices), and profiling showed that dispatch overhead was the single largest CPU cost.

For Vector3-with-Vector3/array/scalar operands the result of pyrr's dispatch is exactly the numpy
element-wise ufunc (Vector3 * Vector3 is element-wise in pyrr), so we bind those operators straight to
the ufuncs. Anything else (e.g. a Matrix or Quaternion operand) falls back to pyrr's original method,
so semantics are unchanged.
"""
import math

import numpy as np
from pyrr import Vector3
from pyrr.objects.base import BaseMatrix, BaseQuaternion

_orig = {name: getattr(Vector3, name) for name in
         ("__add__", "__sub__", "__mul__", "__truediv__", "__radd__", "__rsub__", "__rmul__", "__neg__")}
_SPECIAL = (BaseMatrix, BaseQuaternion)


def _mk(ufunc, name, reverse=False):
    orig = _orig[name]

    def op(self, other):
        if isinstance(other, _SPECIAL):
            return orig(self, other)
        return ufunc(other, self) if reverse else ufunc(self, other)
    op.__name__ = name
    return op


Vector3.__add__ = _mk(np.add, "__add__")
Vector3.__radd__ = _mk(np.add, "__radd__", True)
Vector3.__sub__ = _mk(np.subtract, "__sub__")
Vector3.__rsub__ = _mk(np.subtract, "__rsub__", True)
Vector3.__mul__ = _mk(np.multiply, "__mul__")
Vector3.__rmul__ = _mk(np.multiply, "__rmul__", True)
Vector3.__truediv__ = _mk(np.true_divide, "__truediv__")
Vector3.__neg__ = lambda self: np.negative(self)


def _length(self):
    x, y, z = float(self[0]), float(self[1]), float(self[2])
    return math.sqrt(x * x + y * y + z * z)


def _normalized(self):
    l = _length(self)
    return np.multiply(self, 1.0 / l) if l > 0 else np.multiply(self, 0.0)


def _squared_length(self):
    x, y, z = float(self[0]), float(self[1]), float(self[2])
    return x * x + y * y + z * z


Vector3.length = property(_length)
Vector3.normalized = property(_normalized)
Vector3.squared_length = property(_squared_length)
Vector3.normalise = Vector3.normalize = lambda self: self.__setitem__(slice(None), _normalized(self))


def cross(a, b):
    """Fast 3-vector cross product (np.cross is ~15 us on tiny arrays)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return Vector3((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Same matrix as pyrr Matrix44.look_at (verified to 1e-12) without pyrr's per-call overhead."""
    eye = np.asarray(eye, "f8"); target = np.asarray(target, "f8"); up = np.asarray(up, "f8")
    f = target - eye
    f /= math.sqrt(f.dot(f))
    s = np.array([f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]])
    s /= math.sqrt(s.dot(s))
    u = np.array([s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]])
    m = np.eye(4)
    m[:3, 0] = s; m[:3, 1] = u; m[:3, 2] = -f
    m[3, 0] = -s.dot(eye); m[3, 1] = -u.dot(eye); m[3, 2] = f.dot(eye)
    return m

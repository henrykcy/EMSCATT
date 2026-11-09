"""Scattering-centre *features*: geometry-driven kernels K(f, r_hat).

A scattering centre = feature kernel x smooth complex angular amplitude.
The kernel carries everything that geometry dictates (location, sliding of
the specular point, coherent length of distributed centres, physical-optics
magnitude), the fitted amplitude carries material / coating / diffraction-
coefficient / visibility effects.  Every feature belongs to a named *part* so
that fitted centres map onto physical components (fuselage, wing leading
edge, fin, inlet ...).

Feature types (GSC taxonomy of Lu et al. 2022, Table 1; sliding model of Guo
et al. 2013):

=================  =====================================  =====================
Feature            physical mechanism                     GSC type
=================  =====================================  =====================
PointFeature       tip / corner / cavity / localised      fixed (FF)
EdgeFeature        straight edge diffraction (ASC sinc)   fixed-distributed (FD)
EllipsoidFeature   doubly curved surface specular (GO)    sliding (SS)
CylinderFeature    singly curved surface specular (PO)    sliding-distributed (SD)
PlateFeature       flat panel (exact PO of polygon)       distributed (DD)
=================  =====================================  =====================

Kernel normalisation: for Point/Edge features the fitted amplitude is the
scattering amplitude in metres (sigma = |A|^2 at peak).  For Ellipsoid,
Cylinder and Plate the kernel is the PEC GO/PO response, so the fitted
amplitude is a dimensionless *reflectivity* (1 = ideal PEC; < 1 for RAM etc.).
"""
from __future__ import annotations

import numpy as np

from .conventions import Pose, unit, wavenumber
from .po import tri_unit_integral


def _sinc(t):
    return np.sinc(t / np.pi)


def _freq_factor(freqs, alpha, f_ref):
    if alpha == 0:
        return np.ones(np.size(freqs))
    return (np.asarray(freqs, float) / f_ref) ** alpha


class Feature:
    kind = "base"
    physical = False          # True -> amplitude is a dimensionless reflectivity
    refinable = ()            # names of continuous params the fitter may adjust

    def __init__(self, part="unassigned", name=None, alpha=0.0):
        self.part = part
        self.name = name or self.kind
        self.alpha = float(alpha)

    # to implement ---------------------------------------------------------
    def kernel(self, freqs, dirs, f_ref) -> np.ndarray:  # (Nf, Na)
        raise NotImplementedError

    def location(self, dirs) -> np.ndarray:  # (Na, 3) apparent phase centre
        raise NotImplementedError

    def transformed(self, pose: Pose) -> "Feature":
        raise NotImplementedError

    def params(self) -> dict:
        raise NotImplementedError

    # helpers --------------------------------------------------------------
    def _phase(self, freqs, dirs, x):
        k = wavenumber(freqs)
        return np.exp(2j * k[:, None] * (np.atleast_2d(dirs) @ np.asarray(x))[None, :])

    def with_alpha(self, alpha):
        d = self.to_dict()
        d["alpha"] = alpha
        return feature_from_dict(d)

    def get_vector(self) -> np.ndarray:
        return np.concatenate([np.atleast_1d(getattr(self, p)) for p in self.refinable]) \
            if self.refinable else np.zeros(0)

    def with_vector(self, v) -> "Feature":
        d = self.to_dict()
        i = 0
        for p in self.refinable:
            n = np.size(getattr(self, p))
            d[p] = list(np.asarray(v[i:i + n], float)) if n > 1 else float(v[i])
            i += n
        return feature_from_dict(d)

    @property
    def reference_point(self) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self):
        d = {"kind": self.kind, "part": self.part, "name": self.name, "alpha": self.alpha}
        d.update({k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in self.params().items()})
        return d

    def __repr__(self):
        return f"{type(self).__name__}({self.part}/{self.name}, alpha={self.alpha:g})"


class PointFeature(Feature):
    kind = "point"
    refinable = ("position",)

    def __init__(self, position, part="unassigned", name=None, alpha=0.0):
        super().__init__(part, name, alpha)
        self.position = np.asarray(position, float).reshape(3)

    def kernel(self, freqs, dirs, f_ref):
        return self._phase(freqs, dirs, self.position) * _freq_factor(freqs, self.alpha, f_ref)[:, None]

    def location(self, dirs):
        return np.broadcast_to(self.position, (np.atleast_2d(dirs).shape[0], 3)).copy()

    def transformed(self, pose):
        return PointFeature(pose.apply_points(self.position), self.part, self.name, self.alpha)

    def params(self):
        return {"position": self.position}

    @property
    def reference_point(self):
        return self.position


class EdgeFeature(Feature):
    """Straight distributed edge p1-p2: K = sinc(k L r.t) exp(j2k r.xc) (f/fc)^alpha."""
    kind = "edge"
    refinable = ("centre",)

    def __init__(self, p1=None, p2=None, part="unassigned", name=None, alpha=0.0, centre=None, axis=None,
                 length=None):
        super().__init__(part, name, alpha)
        if p1 is not None:
            p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
            centre, axis, length = (p1 + p2) / 2, unit(p2 - p1), np.linalg.norm(p2 - p1)
        self.centre = np.asarray(centre, float).reshape(3)
        self.axis = unit(np.asarray(axis, float).reshape(3))
        self.length = float(length)

    @property
    def endpoints(self):
        return self.centre - 0.5 * self.length * self.axis, self.centre + 0.5 * self.length * self.axis

    def kernel(self, freqs, dirs, f_ref):
        k = wavenumber(freqs)
        rt = np.atleast_2d(dirs) @ self.axis
        return (_sinc(k[:, None] * self.length * rt[None, :]) * self._phase(freqs, dirs, self.centre)
                * _freq_factor(freqs, self.alpha, f_ref)[:, None])

    def location(self, dirs):
        return np.broadcast_to(self.centre, (np.atleast_2d(dirs).shape[0], 3)).copy()

    def transformed(self, pose):
        return EdgeFeature(part=self.part, name=self.name, alpha=self.alpha,
                           centre=pose.apply_points(self.centre), axis=pose.apply_vectors(self.axis),
                           length=self.length)

    def params(self):
        return {"centre": self.centre, "axis": self.axis, "length": self.length}

    @property
    def reference_point(self):
        return self.centre


class EllipsoidFeature(Feature):
    """Sliding specular point of a (doubly curved) ellipsoidal surface, GO kernel.

    Specular point x_s = c + R (D^2 u)/sqrt(u^T D^2 u), u = R^T r_hat;
    K = -sqrt(pi) abc / (u^T D^2 u) * exp(j 2k r.x_s).  Covers spheres, nose
    radomes, fuselage sections, pods.
    """
    kind = "ellipsoid"
    physical = True

    def __init__(self, centre, semi_axes, rotation=None, part="unassigned", name=None, alpha=0.0):
        super().__init__(part, name, alpha)
        self.centre = np.asarray(centre, float).reshape(3)
        self.semi_axes = np.asarray(semi_axes, float).reshape(3)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, float).reshape(3, 3)

    def _u(self, dirs):
        return np.atleast_2d(dirs) @ self.rotation

    def location(self, dirs):
        u = self._u(dirs)
        D2 = self.semi_axes ** 2
        q = np.sum(u * u * D2, 1)
        xl = u * D2 / np.sqrt(q)[:, None]
        return self.centre + xl @ self.rotation.T

    def kernel(self, freqs, dirs, f_ref):
        k = wavenumber(freqs)
        d = np.atleast_2d(dirs)
        u = self._u(d)
        q = np.sum(u * u * self.semi_axes ** 2, 1)
        amp = -np.sqrt(np.pi) * np.prod(self.semi_axes) / q
        xs = self.location(d)
        ph = np.exp(2j * k[:, None] * np.sum(d * xs, 1)[None, :])
        return amp[None, :] * ph * _freq_factor(freqs, self.alpha, f_ref)[:, None]

    def transformed(self, pose):
        return EllipsoidFeature(pose.apply_points(self.centre), self.semi_axes, pose.rotation @ self.rotation,
                                self.part, self.name, self.alpha)

    def params(self):
        return {"centre": self.centre, "semi_axes": self.semi_axes, "rotation": self.rotation}

    @property
    def reference_point(self):
        return self.centre


class CylinderFeature(Feature):
    """Finite circular cylinder (open tube) - sliding specular line, PO kernel.

    K = -j e^{-j pi/4} L sqrt(k a cos b) sinc(k L r.t) exp(j2k r.(c + a n_perp))
    """
    kind = "cylinder"
    physical = True

    def __init__(self, p1=None, p2=None, radius=0.1, part="unassigned", name=None, alpha=0.0,
                 centre=None, axis=None, length=None):
        super().__init__(part, name, alpha)
        if p1 is not None:
            p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
            centre, axis, length = (p1 + p2) / 2, unit(p2 - p1), np.linalg.norm(p2 - p1)
        self.centre = np.asarray(centre, float).reshape(3)
        self.axis = unit(np.asarray(axis, float).reshape(3))
        self.length = float(length)
        self.radius = float(radius)

    def _perp(self, d):
        rt = d @ self.axis
        rp = d - rt[:, None] * self.axis
        cb = np.linalg.norm(rp, axis=1)
        n = rp / np.where(cb > 1e-12, cb, 1.0)[:, None]
        return rt, cb, n

    def location(self, dirs):
        d = np.atleast_2d(dirs)
        _, _, n = self._perp(d)
        return self.centre + self.radius * n

    def kernel(self, freqs, dirs, f_ref):
        k = wavenumber(freqs)[:, None]
        d = np.atleast_2d(dirs)
        rt, cb, n = self._perp(d)
        xs = self.centre + self.radius * n
        amp = (-1j * np.exp(-0.25j * np.pi) * self.length * np.sqrt(k * self.radius * cb[None, :])
               * _sinc(k * self.length * rt[None, :]))
        return amp * np.exp(2j * k * np.sum(d * xs, 1)[None, :]) * _freq_factor(freqs, self.alpha, f_ref)[:, None]

    def transformed(self, pose):
        return CylinderFeature(part=self.part, name=self.name, alpha=self.alpha,
                               centre=pose.apply_points(self.centre), axis=pose.apply_vectors(self.axis),
                               length=self.length, radius=self.radius)

    def params(self):
        return {"centre": self.centre, "axis": self.axis, "length": self.length, "radius": self.radius}

    @property
    def reference_point(self):
        return self.centre


class PlateFeature(Feature):
    """Flat convex polygonal panel - exact PEC PO response (distributed centre)."""
    kind = "plate"
    physical = True

    def __init__(self, vertices, normal=None, two_sided=False, part="unassigned", name=None, alpha=0.0):
        super().__init__(part, name, alpha)
        self.vertices = np.asarray(vertices, float).reshape(-1, 3)
        V = self.vertices
        n = unit(np.cross(V[1] - V[0], V[2] - V[0]))
        if normal is not None and np.dot(n, normal) < 0:
            self.vertices = V[::-1].copy()
            n = -n
        self.normal = n
        self.two_sided = bool(two_sided)

    def kernel(self, freqs, dirs, f_ref):
        k = wavenumber(freqs)
        d = np.atleast_2d(dirs)
        V = self.vertices
        nd = d @ self.normal
        lit = np.abs(nd) if self.two_sided else np.clip(nd, 0, None)
        out = np.zeros((k.size, d.shape[0]), complex)
        for i in range(1, V.shape[0] - 1):
            p0, e1, e2 = V[0], V[i] - V[0], V[i + 1] - V[0]
            A = 0.5 * np.linalg.norm(np.cross(e1, e2))
            kk = 2 * k[:, None]
            out += 2 * A * np.exp(1j * kk * (d @ p0)[None, :]) * tri_unit_integral(kk * (d @ e1), kk * (d @ e2))
        return -1j * k[:, None] / np.sqrt(np.pi) * lit[None, :] * out * _freq_factor(freqs, self.alpha, f_ref)[:, None]

    def location(self, dirs):
        return np.broadcast_to(self.vertices.mean(0), (np.atleast_2d(dirs).shape[0], 3)).copy()

    def transformed(self, pose):
        return PlateFeature(pose.apply_points(self.vertices), pose.apply_vectors(self.normal), self.two_sided,
                            self.part, self.name, self.alpha)

    def params(self):
        return {"vertices": self.vertices, "normal": self.normal, "two_sided": self.two_sided}

    @property
    def reference_point(self):
        return self.vertices.mean(0)


_KINDS = {c.kind: c for c in (PointFeature, EdgeFeature, EllipsoidFeature, CylinderFeature, PlateFeature)}


def feature_from_dict(d) -> Feature:
    d = dict(d)
    cls = _KINDS[d.pop("kind")]
    arr = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in d.items()}
    return cls(**arr)


class Part:
    """A named physical component region with candidate scattering features.

    ``features`` - geometry-derived candidates (edges, tips, curved surfaces...).
    ``region``   - optional (centre, half_sizes[, rotation]) oriented box used to
                   attribute *free* (grid) scattering centres to this part.
    """

    def __init__(self, name, features=(), region=None):
        self.name = name
        self.features = []
        for f in features:
            f.part = name
            self.features.append(f)
        self.region = None
        if region is not None:
            c, h = np.asarray(region[0], float), np.asarray(region[1], float)
            R = np.eye(3) if len(region) < 3 else np.asarray(region[2], float)
            self.region = (c, h, R)

    def add(self, feature):
        feature.part = self.name
        self.features.append(feature)
        return self

    def distance(self, x) -> float:
        """Distance from point x to the part (0 inside its region)."""
        x = np.asarray(x, float)
        ds = []
        if self.region is not None:
            c, h, R = self.region
            loc = np.abs((x - c) @ R)
            ds.append(np.linalg.norm(np.clip(loc - h, 0, None)))
        for f in self.features:
            if isinstance(f, (EdgeFeature, CylinderFeature)):
                t = np.clip((x - f.centre) @ f.axis, -f.length / 2, f.length / 2)
                ds.append(np.linalg.norm(x - (f.centre + t * f.axis)) - getattr(f, "radius", 0.0))
            elif isinstance(f, EllipsoidFeature):
                loc = (x - f.centre) @ f.rotation
                r = np.linalg.norm(loc / f.semi_axes)
                ds.append(max(0.0, (r - 1)) * f.semi_axes.min())
            elif isinstance(f, PlateFeature):
                ds.append(np.min(np.linalg.norm(f.vertices - x, axis=1)) * 0.5 +
                          abs((x - f.vertices[0]) @ f.normal))
            else:
                ds.append(np.linalg.norm(x - f.reference_point))
        return float(min(ds)) if ds else np.inf

    def transformed(self, pose: Pose, name=None) -> "Part":
        reg = None
        if self.region is not None:
            c, h, R = self.region
            reg = (pose.apply_points(c), h, pose.rotation @ R)
        p = Part(name or self.name, [f.transformed(pose) for f in self.features], reg)
        return p

    def bounds(self):
        pts = []
        for f in self.features:
            if isinstance(f, (EdgeFeature, CylinderFeature)):
                pts += list(f.endpoints) if hasattr(f, "endpoints") else [
                    f.centre - f.length / 2 * f.axis, f.centre + f.length / 2 * f.axis]
            elif isinstance(f, EllipsoidFeature):
                for s in (-1, 1):
                    pts += list(f.centre + s * f.rotation * f.semi_axes[None, :])
            elif isinstance(f, PlateFeature):
                pts += list(f.vertices)
            else:
                pts.append(f.reference_point)
        if self.region is not None:
            c, h, R = self.region
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(c + R @ (h * [sx, sy, sz]))
        if not pts:
            return None
        P = np.array(pts)
        return P.min(0), P.max(0)


def _in_region(part, x):
    if part.region is None:
        return False
    c, h, R = part.region
    return bool(np.all(np.abs((np.asarray(x) - c) @ R) <= h + 1e-12))


def assign_part(x, parts, max_distance=np.inf) -> str:
    """Attribute a location to a part: an explicit region containing the point wins
    (smallest such region); otherwise the nearest part within ``max_distance``."""
    if not parts:
        return "unassigned"
    inside = [p for p in parts if _in_region(p, x)]
    if inside:
        return min(inside, key=lambda p: float(np.prod(p.region[1]))).name
    d = [p.distance(x) for p in parts]
    i = int(np.argmin(d))
    return parts[i].name if d[i] <= max_distance else "unassigned"

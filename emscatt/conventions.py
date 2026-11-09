"""Coordinate, time-harmonic and polarisation conventions used throughout EMSCATT.

Conventions (read these before importing data!)
------------------------------------------------
* Time convention ``exp(+j*omega*t)`` (engineering).  Data produced with the
  physics convention ``exp(-i*omega*t)`` must be conjugated on import
  (:meth:`emscatt.data.ScatteringData.conjugate`).
* Monostatic geometry.  ``r_hat`` is the unit vector pointing *from the target
  towards the radar*.
* The complex scattering amplitude ``S`` is normalised so that the radar cross
  section is ``sigma = |S|**2`` (S in metres, sigma in m^2).
* A point scatterer located at ``x`` (object frame, phase reference at the
  origin) contributes ``A * exp(+j * 2k * r_hat . x)``.  Points closer to the
  radar therefore have a phase *advance*.
* Spherical angles: ``theta`` is the polar angle from +z, ``phi`` the azimuth
  from +x towards +y.  Aircraft-style azimuth/elevation helpers are provided
  (elevation measured from the xy-plane).
* Polarisation channels use the spherical basis (theta_hat, phi_hat) at the
  observation direction (Ludwig-II / BSA backscatter alignment).  A channel
  name ``"ab"`` means *receive a, transmit b*; the scattering matrix is
  ``[[tt, tp], [pt, pp]]``.  In a horizontal azimuth cut (theta = 90 deg)
  theta_hat = -z, so ``VV == tt`` and ``HH == pp``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0 = 299_792_458.0  #: speed of light in vacuum (m/s)

CHANNELS = ("tt", "tp", "pt", "pp")
_ALIASES = {
    "vv": "tt", "hh": "pp", "vh": "tp", "hv": "pt",
    "tt": "tt", "pp": "pp", "tp": "tp", "pt": "pt",
    "thth": "tt", "phph": "pp", "thph": "tp", "phth": "pt",
}


def canonical_channel(name: str) -> str:
    """Map a polarisation channel name (e.g. ``"VV"``, ``"HH"``) to ``tt/tp/pt/pp``."""
    key = name.lower().replace("-", "").replace("_", "").replace("theta", "th").replace("phi", "ph")
    if key not in _ALIASES:
        raise ValueError(f"Unknown polarisation channel {name!r}; use one of {CHANNELS} or VV/HH/VH/HV")
    return _ALIASES[key]


def wavenumber(freqs) -> np.ndarray:
    """Free-space wavenumber k = 2*pi*f/c (rad/m)."""
    return 2.0 * np.pi * np.asarray(freqs, dtype=float) / C0


def _rad(a, degrees):
    a = np.asarray(a, dtype=float)
    return np.deg2rad(a) if degrees else a


def dir_from_theta_phi(theta, phi, degrees: bool = True) -> np.ndarray:
    """Unit direction(s) from spherical angles. Broadcasts; returns (..., 3)."""
    t, p = np.broadcast_arrays(_rad(theta, degrees), _rad(phi, degrees))
    st = np.sin(t)
    return np.stack([st * np.cos(p), st * np.sin(p), np.cos(t)], axis=-1)


def theta_phi_from_dir(d, degrees: bool = True):
    d = np.asarray(d, dtype=float)
    theta = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
    phi = np.arctan2(d[..., 1], d[..., 0])
    if degrees:
        return np.rad2deg(theta), np.rad2deg(phi)
    return theta, phi


def dir_from_az_el(az, el, degrees: bool = True) -> np.ndarray:
    """Unit direction(s) from azimuth (from +x towards +y) and elevation (from xy-plane)."""
    a, e = np.broadcast_arrays(_rad(az, degrees), _rad(el, degrees))
    ce = np.cos(e)
    return np.stack([ce * np.cos(a), ce * np.sin(a), np.sin(e)], axis=-1)


def az_el_from_dir(d, degrees: bool = True):
    d = np.asarray(d, dtype=float)
    az = np.arctan2(d[..., 1], d[..., 0])
    el = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
    if degrees:
        return np.rad2deg(az), np.rad2deg(el)
    return az, el


def spherical_basis(d):
    """Return (theta_hat, phi_hat) at direction(s) ``d`` (..., 3).

    At the poles phi is taken as 0 so the basis remains defined.
    """
    d = np.asarray(d, dtype=float)
    theta, phi = theta_phi_from_dir(d, degrees=False)
    rho = np.hypot(d[..., 0], d[..., 1])
    phi = np.where(rho < 1e-12, 0.0, phi)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    th_hat = np.stack([ct * cp, ct * sp, -st], axis=-1)
    ph_hat = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    return th_hat, ph_hat


def unit(v, axis=-1):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def rotation_about_axis(axis, angle, degrees: bool = True) -> np.ndarray:
    """Right-handed rotation matrix (Rodrigues)."""
    a = unit(axis)
    t = float(_rad(angle, degrees))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def angle_between(d1, d2, degrees: bool = True):
    c = np.clip(np.sum(np.asarray(d1) * np.asarray(d2), axis=-1), -1.0, 1.0)
    a = np.arccos(c)
    return np.rad2deg(a) if degrees else a


def fibonacci_sphere(n: int) -> np.ndarray:
    """Quasi-uniform set of ``n`` unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.clip(1 - z * z, 0, None))
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


@dataclass
class Pose:
    """Rigid placement of a component frame inside a parent (global) frame.

    ``x_parent = R @ x_local + t``.  ``R`` must be orthogonal; improper
    rotations (det = -1, mirror images) are allowed, which is how e.g. a
    measured starboard wing is re-used as the port wing.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("Pose rotation must be an orthogonal matrix")

    # constructors -----------------------------------------------------------
    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    @classmethod
    def from_translation(cls, t) -> "Pose":
        return cls(np.eye(3), t)

    @classmethod
    def from_yaw_pitch_roll(cls, yaw=0.0, pitch=0.0, roll=0.0, translation=(0, 0, 0), degrees=True) -> "Pose":
        """Intrinsic z-y'-x'' (yaw, pitch, roll) rotation followed by translation."""
        R = (rotation_about_axis([0, 0, 1], yaw, degrees)
             @ rotation_about_axis([0, 1, 0], pitch, degrees)
             @ rotation_about_axis([1, 0, 0], roll, degrees))
        return cls(R, translation)

    @classmethod
    def mirror(cls, normal=(0, 1, 0), translation=(0, 0, 0)) -> "Pose":
        """Reflection through the plane through the origin with the given normal."""
        n = unit(normal)
        return cls(np.eye(3) - 2 * np.outer(n, n), translation)

    # operations -------------------------------------------------------------
    @property
    def is_proper(self) -> bool:
        return np.linalg.det(self.rotation) > 0

    @property
    def is_identity(self) -> bool:
        return np.allclose(self.rotation, np.eye(3)) and np.allclose(self.translation, 0)

    def apply_points(self, x):
        return np.asarray(x, dtype=float) @ self.rotation.T + self.translation

    def apply_vectors(self, v):
        return np.asarray(v, dtype=float) @ self.rotation.T

    def to_local_dirs(self, d_parent):
        """Direction(s) expressed in the component frame: R^T d."""
        return np.asarray(d_parent, dtype=float) @ self.rotation

    def compose(self, inner: "Pose") -> "Pose":
        """Return self o inner (apply ``inner`` first)."""
        return Pose(self.rotation @ inner.rotation, self.rotation @ inner.translation + self.translation)

    def inverse(self) -> "Pose":
        Rt = self.rotation.T
        return Pose(Rt, -Rt @ self.translation)

    def to_dict(self):
        return {"rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["rotation"]), np.array(d["translation"]))


def polarisation_transform(pose: Pose, dirs_parent) -> np.ndarray:
    """Matrix T (Na, 2, 2) mapping local (theta, phi) components to parent ones.

    For the scattering matrix in backscatter alignment, ``S_parent = T S_local T^T``.
    """
    dg = np.asarray(dirs_parent, dtype=float)
    dl = pose.to_local_dirs(dg)
    tg, pg = spherical_basis(dg)
    tl, pl = spherical_basis(dl)
    tl_g, pl_g = pose.apply_vectors(tl), pose.apply_vectors(pl)
    T = np.empty(dg.shape[:-1] + (2, 2))
    T[..., 0, 0] = np.sum(tg * tl_g, -1)
    T[..., 0, 1] = np.sum(tg * pl_g, -1)
    T[..., 1, 0] = np.sum(pg * tl_g, -1)
    T[..., 1, 1] = np.sum(pg * pl_g, -1)
    return T

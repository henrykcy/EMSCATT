"""Smooth angular amplitude patterns for scattering centres.

Each scattering centre's complex amplitude is allowed to vary slowly with
aspect through a normalised Gaussian radial-basis-function (RBF) expansion

    a(r_hat) = sum_j c_j * B_j(r_hat),  B_j = g_j / sum_i g_i,
    g_j = exp(-angle(r_hat, d_j)^2 / (2 w^2))

with anchors ``d_j`` on the arc (planar cut) or sphere covered by the data.
Normalisation makes the basis a partition of unity, so a constant pattern is
represented exactly and extrapolation outside the anchors is "hold nearest"
rather than decaying to zero.  This captures visibility/shadowing, diffraction
coefficient angle dependence and specular lobes without fixing a functional
form, which is the generalisation of the attributed scattering centre
``exp(-2*pi*f*gamma*sin(phi))`` term and of the rational amplitude function used
by Guo et al. for sliding centres.
"""
from __future__ import annotations

import numpy as np

from .conventions import fibonacci_sphere, unit
from .data import plane_normal


class AngularBasis:
    def __init__(self, anchors, width_rad: float, neighbours=None):
        self.anchors = unit(np.atleast_2d(np.asarray(anchors, float)))
        self.width = float(width_rad)
        if neighbours is None:
            neighbours = self._auto_neighbours()
        self.neighbours = np.asarray(neighbours, int).reshape(-1, 2)

    @property
    def size(self) -> int:
        return self.anchors.shape[0]

    def _auto_neighbours(self):
        n = self.size
        if n < 2:
            return np.zeros((0, 2), int)
        c = np.clip(self.anchors @ self.anchors.T, -1, 1)
        ang = np.arccos(c)
        np.fill_diagonal(ang, np.inf)
        nn = np.min(ang, axis=1)
        lim = 1.6 * np.median(nn)
        i, j = np.where(np.triu(ang <= lim, 1))
        return np.stack([i, j], 1)

    def evaluate(self, dirs) -> np.ndarray:
        """(Na, Nb) real basis matrix."""
        d = np.atleast_2d(np.asarray(dirs, float))
        if self.size == 1:
            return np.ones((d.shape[0], 1))
        ang = np.arccos(np.clip(d @ self.anchors.T, -1, 1))
        lg = -0.5 * (ang / self.width) ** 2
        lg -= lg.max(axis=1, keepdims=True)  # stable normalisation
        g = np.exp(lg)
        return g / g.sum(axis=1, keepdims=True)

    def difference_operator(self) -> np.ndarray:
        """(Ne, Nb) first-difference operator over neighbouring anchors."""
        D = np.zeros((self.neighbours.shape[0], self.size))
        if D.size:
            r = np.arange(D.shape[0])
            D[r, self.neighbours[:, 0]] = 1.0
            D[r, self.neighbours[:, 1]] = -1.0
        return D

    def transformed(self, R) -> "AngularBasis":
        return AngularBasis(self.anchors @ np.asarray(R).T, self.width, self.neighbours)

    # --- factories ------------------------------------------------------------
    @classmethod
    def constant(cls) -> "AngularBasis":
        return cls(np.array([[1.0, 0, 0]]), np.pi)

    @classmethod
    def for_directions(cls, dirs, spacing_deg: float = 15.0) -> "AngularBasis":
        """Anchors spaced ~``spacing_deg`` covering the data directions."""
        d = np.atleast_2d(np.asarray(dirs, float))
        sp = np.deg2rad(spacing_deg)
        if spacing_deg >= 180 or d.shape[0] < 3:
            return cls.constant()
        n = plane_normal(d)
        if n is not None:
            # planar cut: anchors on the covered arc of the great circle
            u1 = unit(d[0] - (d[0] @ n) * n)
            u2 = np.cross(n, u1)
            psi = np.sort(np.mod(np.arctan2(d @ u2, d @ u1), 2 * np.pi))
            gaps = np.diff(np.concatenate([psi, [psi[0] + 2 * np.pi]]))
            ig = int(np.argmax(gaps))
            if gaps[ig] < 2 * sp:  # full circle
                nb = max(int(np.ceil(2 * np.pi / sp)), 3)
                ang = np.arange(nb) * 2 * np.pi / nb
                nbr = np.stack([np.arange(nb), (np.arange(nb) + 1) % nb], 1)
                width = 2 * np.pi / nb
            else:
                start = psi[(ig + 1) % psi.size]
                span = 2 * np.pi - gaps[ig]
                nb = max(int(np.ceil(span / sp)) + 1, 2)
                ang = start + np.linspace(0, span, nb)
                nbr = np.stack([np.arange(nb - 1), np.arange(1, nb)], 1)
                width = span / (nb - 1) if nb > 1 else sp
                if span < 1e-9:
                    return cls.constant()
            anchors = np.cos(ang)[:, None] * u1 + np.sin(ang)[:, None] * u2
            return cls(anchors, width, nbr)
        # 3-D coverage: Fibonacci anchors near the data
        nfull = max(int(np.ceil(4 * np.pi / sp ** 2)), 4)
        cand = fibonacci_sphere(nfull)
        near = np.max(cand @ d.T, axis=1) >= np.cos(1.0 * sp)
        anchors = cand[near]
        if anchors.shape[0] < 2:
            return cls.constant()
        return cls(anchors, sp)

    def to_dict(self):
        return {"anchors": self.anchors.tolist(), "width": self.width,
                "neighbours": self.neighbours.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["anchors"]), d["width"], np.array(d["neighbours"], int))

"""Physical-optics (PO) monostatic solver for triangulated PEC surfaces.

Purpose: (i) generate independent, physically realistic synthetic "measured"
or "simulated" data for validating the scattering-centre workflow, (ii) give
quick component-level predictions, and (iii) provide the exact PO kernel of a
flat polygon used by :class:`emscatt.features.PlateFeature`.

Monostatic co-polarised PO (PEC) with the conventions of
:mod:`emscatt.conventions`:

    S(k, r) = -j k/sqrt(pi) * sum_lit (n . r) * Int_T exp(j 2k r . x) dA

Cross-polarisation is zero in monostatic PO, and tt == pp.  Only back-face
culling is applied (no occlusion/shadowing, no multiple bounces, no edge
diffraction or creeping waves) - ample for validation, but not a substitute
for a full-wave or SBR+PTD solver.
"""
from __future__ import annotations

import struct

import numpy as np

from .conventions import Pose, unit, wavenumber


# ---------------------------------------------------------------------------
# Robust triangle phase integral
# ---------------------------------------------------------------------------
def _sinc(t):
    return np.sinc(t / np.pi)


def _dd1(x, y):
    """First divided difference of exp(jx) (robust for x ~ y)."""
    return 1j * np.exp(0.5j * (x + y)) * _sinc(0.5 * (x - y))


def tri_unit_integral(a, b) -> np.ndarray:
    """f(a,b) = Int_0^1 Int_0^{1-u} exp(j(a u + b v)) dv du (vectorised, robust).

    Equals -F[0, a, b], the negated 2nd divided difference of F(x)=exp(jx).
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    x = np.stack([np.zeros_like(a), a, b], -1)
    x = np.sort(x, axis=-1)
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    spread = x2 - x0
    out = np.empty(a.shape, complex)
    big = spread >= 0.5
    if np.any(big):
        X0, X1, X2 = x0[big], x1[big], x2[big]
        out[big] = -(_dd1(X1, X2) - _dd1(X0, X1)) / (X2 - X0)
    small = ~big
    if np.any(small):
        X0, X1, X2 = x0[small], x1[small], x2[small]
        c = (X0 + X1 + X2) / 3
        y0, y1, y2 = X0 - c, X1 - c, X2 - c
        # h_n complete homogeneous symmetric polynomials of (y0,y1,y2)
        h2 = [np.ones_like(y2)]  # h_n(y2)
        h12 = [np.ones_like(y2)]  # h_n(y1,y2)
        h012 = [np.ones_like(y2)]
        acc = np.zeros(y0.shape, complex)
        fact = 2.0
        jp = -1.0 + 0j  # j^(n+2), n=0
        acc += jp * h012[0] / fact
        for n in range(1, 16):
            h2.append(h2[-1] * y2)
            h12.append(h2[-1] + y1 * h12[-1])
            h012.append(h12[-1] + y0 * h012[-1])
            fact *= (n + 2)
            jp *= 1j
            acc += jp * h012[-1] / fact
        out[small] = -np.exp(1j * c) * acc
    return out


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
class Mesh:
    """Triangle mesh. ``two_sided`` faces radiate from both sides (thin plates)."""

    def __init__(self, vertices, faces, two_sided=False, name="mesh"):
        self.vertices = np.asarray(vertices, float).reshape(-1, 3)
        self.faces = np.asarray(faces, int).reshape(-1, 3)
        ts = np.asarray(two_sided, bool)
        self.two_sided = np.broadcast_to(ts, (self.faces.shape[0],)).copy()
        self.name = name

    @property
    def tri(self):
        return self.vertices[self.faces]  # (Nt,3,3)

    @property
    def normals(self):
        t = self.tri
        return unit(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]))

    @property
    def areas(self):
        t = self.tri
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    @property
    def centroids(self):
        return self.tri.mean(axis=1)

    def transformed(self, pose: Pose) -> "Mesh":
        f = self.faces if pose.is_proper else self.faces[:, ::-1]  # keep outward normals
        return Mesh(pose.apply_points(self.vertices), f, self.two_sided, self.name)

    def __add__(self, other: "Mesh") -> "Mesh":
        return Mesh(np.vstack([self.vertices, other.vertices]),
                    np.vstack([self.faces, other.faces + self.vertices.shape[0]]),
                    np.concatenate([self.two_sided, other.two_sided]), f"{self.name}+{other.name}")

    @staticmethod
    def merge(meshes):
        out = meshes[0]
        for m in meshes[1:]:
            out = out + m
        return out

    def bounds(self):
        return self.vertices.min(0), self.vertices.max(0)

    # --- builders ---------------------------------------------------------
    @classmethod
    def ellipsoid(cls, semi_axes, n_u=64, n_v=32, centre=(0, 0, 0), name="ellipsoid"):
        a, b, c = semi_axes
        u = np.linspace(0, np.pi, n_v + 1)            # polar
        v = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        verts = [[0, 0, c]]
        for ui in u[1:-1]:
            for vj in v:
                verts.append([a * np.sin(ui) * np.cos(vj), b * np.sin(ui) * np.sin(vj), c * np.cos(ui)])
        verts.append([0, 0, -c])
        verts = np.array(verts) + np.asarray(centre)
        faces = []
        nr = n_v - 1
        idx = lambda i, j: 1 + i * n_u + (j % n_u)
        for j in range(n_u):
            faces.append([0, idx(0, j), idx(0, j + 1)])
        for i in range(nr - 1):
            for j in range(n_u):
                faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
                faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
        last = verts.shape[0] - 1
        for j in range(n_u):
            faces.append([last, idx(nr - 1, j + 1), idx(nr - 1, j)])
        m = cls(verts, faces, False, name)
        # ensure outward orientation
        if np.mean(np.sum(m.normals * (m.centroids - np.asarray(centre)), 1)) < 0:
            m.faces = m.faces[:, ::-1]
        return m

    @classmethod
    def cylinder(cls, p1, p2, radius, n_circ=32, n_len=1, caps=False, name="cylinder"):
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        t = unit(p2 - p1)
        e1 = unit(np.cross(t, [0.123, 0.456, 0.789]))
        e2 = np.cross(t, e1)
        ang = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
        ring = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
        verts, faces = [], []
        for i in range(n_len + 1):
            verts.append(p1 + (p2 - p1) * i / n_len + radius * ring)
        verts = np.vstack(verts)
        for i in range(n_len):
            for j in range(n_circ):
                a0, a1 = i * n_circ + j, i * n_circ + (j + 1) % n_circ
                b0, b1 = a0 + n_circ, a1 + n_circ
                faces += [[a0, a1, b1], [a0, b1, b0]]
        m = cls(verts, faces, False, name)
        if np.mean(np.sum(m.normals * (m.centroids - (p1 + p2) / 2), 1)) < 0:
            m.faces = m.faces[:, ::-1]
        if caps:
            m = m + cls.disk(p1, -t, radius, n_circ) + cls.disk(p2, t, radius, n_circ)
            m.name = name
        return m

    @classmethod
    def polygon(cls, vertices, normal=None, two_sided=False, name="plate"):
        """Flat convex polygon (fan triangulation). ``normal`` sets the lit side."""
        V = np.asarray(vertices, float)
        faces = [[0, i, i + 1] for i in range(1, V.shape[0] - 1)]
        m = cls(V, faces, two_sided, name)
        if normal is not None and np.mean(m.normals @ unit(normal)) < 0:
            m.faces = m.faces[:, ::-1]
        return m

    @classmethod
    def disk(cls, centre, normal, radius, n=32, two_sided=False, name="disk"):
        n_hat = unit(normal)
        e1 = unit(np.cross(n_hat, [0.123, 0.456, 0.789]))
        e2 = np.cross(n_hat, e1)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        V = np.asarray(centre) + radius * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
        return cls.polygon(V, n_hat, two_sided, name)

    @classmethod
    def box(cls, lo, hi, name="box"):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                      [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
        quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
        faces = [f for q in quads for f in ([q[0], q[1], q[2]], [q[0], q[2], q[3]])]
        return cls(c, faces, False, name)

    @classmethod
    def load_stl(cls, path, scale=1.0, two_sided=False):
        with open(path, "rb") as fh:
            data = fh.read()
        if data[:5].lower() == b"solid" and b"facet" in data[:1000]:
            txt = data.decode(errors="ignore").split()
            vs = [list(map(float, txt[i + 1:i + 4])) for i, w in enumerate(txt) if w == "vertex"]
            V = np.array(vs) * scale
        else:
            n = struct.unpack("<I", data[80:84])[0]
            rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
                [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            V = rec["v"].reshape(-1, 3).astype(float) * scale
        return cls(V, np.arange(V.shape[0]).reshape(-1, 3), two_sided, str(path))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def po_monostatic(mesh: Mesh, freqs, dirs, max_chunk=4_000_000) -> np.ndarray:
    """Monostatic co-pol PO scattering amplitude S (Nf, Na), sigma = |S|^2."""
    k = wavenumber(freqs)
    d = np.atleast_2d(np.asarray(dirs, float))
    T = mesh.tri
    p0, e1, e2 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    n, A = mesh.normals, mesh.areas
    out = np.zeros((k.size, d.shape[0]), complex)
    ndr = d @ n.T                                    # (Na, Nt)
    lit_fac = np.where(mesh.two_sided[None, :], np.abs(ndr), np.clip(ndr, 0, None))
    p0d, e1d, e2d = d @ p0.T, d @ e1.T, d @ e2.T      # (Na, Nt)
    for ia in range(d.shape[0]):
        lit = lit_fac[ia] > 0
        if not np.any(lit):
            continue
        w = (2 * A[lit] * lit_fac[ia, lit])
        step = max(1, max_chunk // max(1, lit.sum()))
        for i0 in range(0, k.size, step):
            kk = 2 * k[i0:i0 + step, None]
            f = tri_unit_integral(kk * e1d[ia, lit], kk * e2d[ia, lit])
            out[i0:i0 + step, ia] = np.sum(w * np.exp(1j * kk * p0d[ia, lit]) * f, axis=1)
    return -1j * k[:, None] / np.sqrt(np.pi) * out

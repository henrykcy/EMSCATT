"""Back-projection (ISAR) imaging - diagnostics and visual SC verification."""
from __future__ import annotations

import numpy as np

from .conventions import C0, unit
from .data import ScatteringData


def plane_grid(extent, spacing, normal=(0, 0, 1), centre=(0, 0, 0)):
    """Grid of points in the plane through ``centre`` with ``normal``.

    extent = (u_min, u_max, v_min, v_max) along in-plane axes (u1, u2).
    Returns X (Nu*Nv, 3), u (Nu,), v (Nv,), (u1, u2).  For normal = z, u1 = x, u2 = y.
    """
    n = unit(normal)
    if np.allclose(np.abs(n), [0, 0, 1]):
        u1, u2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]) * np.sign(n[2])
    else:
        u1 = unit(np.cross([0, 0, 1], n))
        u2 = np.cross(n, u1)
    u = np.arange(extent[0], extent[1] + 1e-9, spacing)
    v = np.arange(extent[2], extent[3] + 1e-9, spacing)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = np.asarray(centre) + U.reshape(-1, 1) * u1 + V.reshape(-1, 1) * u2
    return X, u, v, (u1, u2)


def backproject(data: ScatteringData, X, channel=None, window=True, oversample=16, chunk=4000):
    """Complex back-projection image I(x) = sum_{f,r} w W S e^{-j2k r.x} at points X (N,3)."""
    ch = data._ch(channel)
    S = data.S[ch] * data.w
    if window:
        S = S * np.hanning(data.nf + 2)[1:-1, None] if data.nf > 2 else S
    X = np.atleast_2d(X)
    out = np.zeros(X.shape[0], complex)
    uniform = data.nf > 2 and np.allclose(np.diff(data.freqs), data.freqs[1] - data.freqs[0], rtol=1e-6)
    if uniform:
        M = int(2 ** np.ceil(np.log2(data.nf * oversample)))
        dk2 = 4 * np.pi * (data.freqs[1] - data.freqs[0]) / C0
        period = 2 * np.pi / dk2
        P = np.fft.fft(S, n=M, axis=0)
        P = np.vstack([P, P[:1]])
    for i0 in range(0, X.shape[0], chunk):
        proj = X[i0:i0 + chunk] @ data.dirs.T
        if uniform:
            t = np.mod(proj, period) / period * M
            j0 = np.minimum(np.floor(t).astype(int), M - 1)
            fr = t - j0
            cols = np.arange(proj.shape[1])[None, :]
            val = (P[j0, cols] * (1 - fr) + P[j0 + 1, cols] * fr) * np.exp(-2j * data.k[0] * proj)
        else:
            val = sum(S[i][None, :] * np.exp(-2j * data.k[i] * proj) for i in range(data.nf))
        out[i0:i0 + chunk] = val.sum(1)
    return out / (data.nf * data.na)

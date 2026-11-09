"""Containers and loaders for complex monostatic scattering data.

A :class:`ScatteringData` block holds the complex scattering amplitude
``S[channel][i_freq, i_dir]`` on a (frequency x direction) grid.  Several
blocks with *different* grids (e.g. a wide-band simulated full sphere plus a
narrow-band measured azimuth cut) can be fitted together - simply pass a list
of blocks to the fitter.  Per-sample ``weights`` (e.g. 1/noise variance) let
you balance experimental and simulated sources.
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field

import numpy as np

from .conventions import (C0, canonical_channel, dir_from_az_el, dir_from_theta_phi,
                          wavenumber, az_el_from_dir, theta_phi_from_dir)


@dataclass
class ScatteringData:
    freqs: np.ndarray                      # (Nf,) Hz
    dirs: np.ndarray                       # (Na, 3) unit vectors target -> radar
    S: dict                                # channel -> complex (Nf, Na), sqrt(m^2)
    weights: np.ndarray | None = None      # (Nf, Na) non-negative
    name: str = "data"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.freqs = np.atleast_1d(np.asarray(self.freqs, dtype=float))
        self.dirs = np.atleast_2d(np.asarray(self.dirs, dtype=float))
        n = np.linalg.norm(self.dirs, axis=1)
        if not np.allclose(n, 1.0, atol=1e-6):
            raise ValueError("dirs must be unit vectors")
        S = {}
        for ch, v in self.S.items():
            v = np.asarray(v, dtype=complex)
            if v.ndim == 1:
                v = v.reshape(self.nf, self.na)
            if v.shape != (self.nf, self.na):
                raise ValueError(f"S[{ch}] has shape {v.shape}, expected {(self.nf, self.na)}")
            S[canonical_channel(ch)] = v
        self.S = S
        if self.weights is not None:
            self.weights = np.broadcast_to(np.asarray(self.weights, float), (self.nf, self.na)).copy()

    # --- basic properties -------------------------------------------------
    @property
    def nf(self):
        return self.freqs.size

    @property
    def na(self):
        return self.dirs.shape[0]

    @property
    def channels(self):
        return tuple(self.S.keys())

    @property
    def k(self):
        return wavenumber(self.freqs)

    @property
    def w(self) -> np.ndarray:
        return np.ones((self.nf, self.na)) if self.weights is None else self.weights

    def rcs(self, channel=None) -> np.ndarray:
        return np.abs(self.S[self._ch(channel)]) ** 2

    def rcs_dbsm(self, channel=None, floor_db=-200.0) -> np.ndarray:
        return np.maximum(10 * np.log10(self.rcs(channel) + 1e-30), floor_db)

    def _ch(self, channel):
        return self.channels[0] if channel is None else canonical_channel(channel)

    def energy(self, channels=None) -> float:
        chs = self.channels if channels is None else [canonical_channel(c) for c in channels]
        return float(sum(np.sum(self.w * np.abs(self.S[c]) ** 2) for c in chs))

    def plane_normal(self, tol=1e-6):
        """Normal of the plane containing all directions, or None for 3-D coverage."""
        return plane_normal(self.dirs, tol)

    def az_el(self, degrees=True):
        return az_el_from_dir(self.dirs, degrees)

    def theta_phi(self, degrees=True):
        return theta_phi_from_dir(self.dirs, degrees)

    # --- constructors -----------------------------------------------------
    @classmethod
    def from_theta_phi(cls, freqs, theta, phi, S, degrees=True, **kw):
        return cls(freqs, dir_from_theta_phi(theta, phi, degrees).reshape(-1, 3), S, **kw)

    @classmethod
    def from_az_el(cls, freqs, az, el, S, degrees=True, **kw):
        return cls(freqs, dir_from_az_el(az, el, degrees).reshape(-1, 3), S, **kw)

    @classmethod
    def from_rcs_db_phase(cls, freqs, dirs, rcs_dbsm: dict, phase_deg: dict, **kw):
        S = {ch: 10 ** (np.asarray(rcs_dbsm[ch]) / 20) * np.exp(1j * np.deg2rad(phase_deg[ch]))
             for ch in rcs_dbsm}
        return cls(freqs, dirs, S, **kw)

    # --- transformations --------------------------------------------------
    def copy(self, **changes):
        d = dict(freqs=self.freqs.copy(), dirs=self.dirs.copy(),
                 S={c: v.copy() for c, v in self.S.items()},
                 weights=None if self.weights is None else self.weights.copy(),
                 name=self.name, meta=dict(self.meta))
        d.update(changes)
        return ScatteringData(**d)

    def conjugate(self):
        """Convert between exp(-i w t) and exp(+j w t) time conventions."""
        return self.copy(S={c: np.conj(v) for c, v in self.S.items()})

    def with_phase_reference(self, p):
        """Data whose phase was referenced to point ``p`` (object frame, e.g. the
        turntable rotation centre) -> data referenced to the object origin."""
        ph = np.exp(2j * self.k[:, None] * (self.dirs @ np.asarray(p, float))[None, :])
        return self.copy(S={c: v * ph for c, v in self.S.items()})

    def subset(self, freq_mask=None, dir_mask=None, channels=None):
        fm = slice(None) if freq_mask is None else np.asarray(freq_mask)
        dm = slice(None) if dir_mask is None else np.asarray(dir_mask)
        chs = self.channels if channels is None else [canonical_channel(c) for c in channels]
        return ScatteringData(self.freqs[fm], self.dirs[dm],
                              {c: self.S[c][fm][:, dm] for c in chs},
                              None if self.weights is None else self.weights[fm][:, dm],
                              self.name, dict(self.meta))

    def __sub__(self, other: "ScatteringData"):
        if other.nf != self.nf or other.na != self.na or not np.allclose(other.freqs, self.freqs) \
                or not np.allclose(other.dirs, self.dirs):
            raise ValueError("Data grids differ")
        return self.copy(S={c: self.S[c] - other.S[c] for c in self.channels if c in other.S},
                         name=f"{self.name}-{other.name}")

    def add_noise(self, snr_db: float, rng=None):
        """Add complex white noise at ``snr_db`` below the mean signal power."""
        rng = np.random.default_rng(rng)
        out = {}
        for c, v in self.S.items():
            p = np.mean(np.abs(v) ** 2) * 10 ** (-snr_db / 10)
            out[c] = v + np.sqrt(p / 2) * (rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape))
        return self.copy(S=out)

    # --- I/O ---------------------------------------------------------------
    def save_npz(self, path):
        np.savez_compressed(path, freqs=self.freqs, dirs=self.dirs,
                            channels=np.array(self.channels),
                            S=np.stack([self.S[c] for c in self.channels]),
                            weights=self.w, name=self.name)

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        chs = [str(c) for c in z["channels"]]
        return cls(z["freqs"], z["dirs"], {c: z["S"][i] for i, c in enumerate(chs)},
                   z["weights"], str(z["name"]))

    @classmethod
    def load_csv(cls, path, *, freq_col="freq", freq_scale=1.0,
                 angle_cols=("theta", "phi"), angle_type="theta_phi", degrees=True,
                 channels=None, value_format="re_im", time_convention="+j", name=None,
                 delimiter=","):
        """Load a long-format CSV (one row per frequency/direction sample).

        Parameters
        ----------
        freq_col, freq_scale : column name and multiplier to Hz (1e9 for GHz).
        angle_cols, angle_type : ("theta","phi") with "theta_phi" or ("az","el") with "az_el".
        channels : dict mapping channel -> column prefix, e.g. {"VV": "vv"}. Columns are
            then ``vv_re``/``vv_im`` (value_format="re_im"), ``vv_db``/``vv_deg``
            (value_format="db_deg", RCS in dBsm and phase in degrees) or ``vv_lin``/``vv_deg``.
        time_convention : "+j" (exp(+jwt), default) or "-i" (data will be conjugated).
        The samples must form a full frequency x direction grid (any order).
        """
        with open(path, newline="") as fh:
            rows = list(csv.DictReader((l for l in fh if not l.startswith("#")), delimiter=delimiter))
        if not rows:
            raise ValueError(f"No data in {path}")
        cols = {k.strip(): k for k in rows[0].keys()}
        get = lambda r, c: float(r[cols[c]])
        if channels is None:
            raise ValueError("Specify channels, e.g. channels={'VV': 'vv'}")
        f = np.array([get(r, freq_col) * freq_scale for r in rows])
        a1 = np.array([get(r, angle_cols[0]) for r in rows])
        a2 = np.array([get(r, angle_cols[1]) for r in rows])
        fu, fi = np.unique(np.round(f, 3), return_inverse=True)
        key = np.round(a1, 9) + 1j * np.round(a2, 9)
        ku, ki = np.unique(key, return_inverse=True)
        if fu.size * ku.size != len(rows):
            raise ValueError("CSV samples do not form a complete frequency x direction grid")
        if angle_type == "theta_phi":
            dirs = dir_from_theta_phi(ku.real, ku.imag, degrees)
        elif angle_type == "az_el":
            dirs = dir_from_az_el(ku.real, ku.imag, degrees)
        else:
            raise ValueError("angle_type must be 'theta_phi' or 'az_el'")
        S = {}
        for ch, pre in channels.items():
            v = np.zeros((fu.size, ku.size), complex)
            if value_format == "re_im":
                vals = np.array([get(r, f"{pre}_re") + 1j * get(r, f"{pre}_im") for r in rows])
            elif value_format == "db_deg":
                vals = np.array([10 ** (get(r, f"{pre}_db") / 20) * np.exp(1j * np.deg2rad(get(r, f"{pre}_deg")))
                                 for r in rows])
            elif value_format == "lin_deg":
                vals = np.array([get(r, f"{pre}_lin") * np.exp(1j * np.deg2rad(get(r, f"{pre}_deg")))
                                 for r in rows])
            else:
                raise ValueError("value_format must be re_im, db_deg or lin_deg")
            v[fi, ki] = vals
            S[ch] = v
        out = cls(fu, dirs, S, name=name or str(path))
        if time_convention == "-i":
            out = out.conjugate()
        return out

    def save_csv(self, path, angle_type="theta_phi"):
        if angle_type == "theta_phi":
            a1, a2 = self.theta_phi()
            names = ("theta", "phi")
        else:
            a1, a2 = self.az_el()
            names = ("az", "el")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["freq", *names] + [f"{c}_{p}" for c in self.channels for p in ("re", "im")])
            for i, f in enumerate(self.freqs):
                for j in range(self.na):
                    w.writerow([f, a1[j], a2[j]] + [x for c in self.channels
                                                     for x in (self.S[c][i, j].real, self.S[c][i, j].imag)])


def estimate_noise_power(data: ScatteringData, extent_radius: float, channel=None, guard_cells=2.0):
    """Per-sample complex noise variance estimated from range gates outside the object.

    Requires uniformly spaced frequencies with an unambiguous range larger than
    the object.  Range bins farther than ``extent_radius`` + guard from the phase
    reference contain only noise (and weak sidelobe leakage); a median estimator
    makes the result robust to that leakage.  Returns None when not estimable.
    """
    if data.nf < 8:
        return None
    df = np.diff(data.freqs)
    if not np.allclose(df, df[0], rtol=1e-6):
        return None
    ch = data._ch(channel)
    period = C0 / (2 * df[0])                     # unambiguous range
    res = C0 / (2 * df[0] * data.nf)               # range bin
    d = np.arange(data.nf) * res
    d = np.minimum(d, period - d)                  # circular distance from 0
    out = d > extent_radius + guard_cells * res
    if out.sum() < 2:
        return None
    win = np.hanning(data.nf + 2)[1:-1]            # suppress range sidelobes of strong scatterers
    P = np.fft.fft(data.S[ch] * win[:, None], axis=0) / np.sqrt(np.sum(win ** 2))
    return float(np.median(np.abs(P[out]) ** 2) / np.log(2))


def plane_normal(dirs, tol=1e-6):
    d = np.asarray(dirs, float)
    if d.shape[0] < 3:
        return None
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    if s[-1] < tol * s[0] * np.sqrt(d.shape[0]):
        return vt[-1]
    return None


def as_blocks(data) -> list:
    if isinstance(data, ScatteringData):
        return [data]
    blocks = list(data)
    if not all(isinstance(b, ScatteringData) for b in blocks):
        raise TypeError("Expected ScatteringData or a list of them")
    return blocks


def sampling_check(data: ScatteringData, max_extent: float):
    """Warn when angular/frequency sampling is too coarse for an object of the given
    maximum extent (m) about the phase reference (Nyquist for the phase history)."""
    msgs = []
    lam = C0 / data.freqs.max()
    if data.na > 1:
        from .conventions import angle_between
        nn = []
        for i in range(min(data.na, 400)):
            a = angle_between(data.dirs[i], np.delete(data.dirs, i, 0), degrees=False)
            nn.append(a.min())
        dang = float(np.median(nn))
        need = lam / (4 * max_extent)
        if dang > need:
            msgs.append(f"angular step {np.rad2deg(dang):.3f} deg > Nyquist {np.rad2deg(need):.3f} deg "
                        f"for extent {max_extent} m")
    if data.nf > 1:
        df = float(np.median(np.diff(np.sort(data.freqs))))
        unamb = C0 / (2 * df)
        if unamb < 2 * max_extent:
            msgs.append(f"unambiguous range {unamb:.2f} m < 2 x extent ({2 * max_extent:.2f} m)")
    for m in msgs:
        warnings.warn(m)
    return msgs

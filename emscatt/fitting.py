"""Extraction of a scattering-centre model from complex scattering data.

Model (all channels share centre geometry, amplitudes are per channel):

    S_ch(f, r) = sum_m K_m(f, r) * sum_j c_{m,j,ch} B_{m,j}(r)

Given the kernels K_m (geometry) the model is *linear* in the amplitude
coefficients c, so extraction is a structured sparse-approximation problem:

1. Candidate dictionary: geometry-derived features of the user's named parts
   (edges, tips, curved surfaces, panels; each optionally with several GTD
   frequency exponents alpha) plus, optionally, a free grid of point
   centres covering the object for anything the parts do not explain
   (cavities, interactions, unknown features).
2. Greedy group selection (group-OMP / multi-measurement): at every step the
   candidate with the largest windowed matched-filter energy against the
   current residual is added (physical part features get a small preference),
   then *all* amplitudes are re-estimated jointly by regularised weighted
   least squares (smoothness across neighbouring angular anchors).
3. RELAX-style refinement of continuous positions of point/edge centres.
4. Pruning of negligible centres; attribution of free centres to the nearest
   part.

Point-candidate correlations use an FFT range-profile back-projection when
frequencies are uniformly spaced (equivalent to sub-aperture ISAR imaging),
which keeps dense candidate grids cheap.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .basis import AngularBasis
from .conventions import C0, canonical_channel, unit, wavenumber
from .data import ScatteringData, as_blocks, estimate_noise_power, plane_normal
from .features import EdgeFeature, Feature, Part, PointFeature, assign_part
from .model import ScatteringCentre, ScatteringCentreModel


@dataclass
class FitConfig:
    channels: tuple | None = None          # default: channels common to all blocks
    basis_spacing_deg: float = 15.0        # angular anchor spacing for point/edge centres
    physical_basis_spacing_deg: float = 45.0  # for physical (PO/GO) features: smoother
    smoothness: float = 3e-2               # anchor-difference penalty (relative)
    ridge: float = 1e-6                    # Tikhonov (relative)
    max_centres: int = 60
    target_residual_db: float = -30.0      # stop when residual energy / data energy below this
    min_gain_db: float = -50.0             # stop when a new centre removes less than this fraction
    alphas: tuple = (0.0,)                 # GTD frequency exponents tried for point/edge features
    free_points: bool = True               # add a free grid of point centres
    grid_spacing: float | None = None      # m; default from resolution
    grid_bounds: tuple | None = None       # (lo(3), hi(3)); default from parts
    grid_margin: float = 0.25              # m added around part bounds
    max_grid_points: int = 15000
    part_bias: float = 0.3                 # score preference for part features (0 = none)
    refine: bool = True
    refine_passes: int = 2
    part_feature_shift: float = 0.0        # m a part point/edge may move during refinement
    assign_radius: float = 0.5             # m; free centres further than this stay 'unassigned'
    prune_db: float = -45.0                # drop centres whose energy share is below this
    fft_oversample: int = 16
    noise_power: float | None = None       # per-sample complex noise variance; None = estimate
    noise_gain_factor: float = 4.0         # a centre must remove > factor x (expected noise fit) energy
    balance_aspects: bool = True           # equalise accuracy across aspect (dB-like fidelity)
    balance_dynamic_range_db: float = 40.0 # aspects weaker than peak sector by more are not boosted further
    verbose: bool = False


@dataclass
class FitResult:
    model: ScatteringCentreModel
    residual_db: float
    history: list
    residual: list                         # residual ScatteringData blocks
    config: FitConfig = field(repr=False, default=None)

    def report(self) -> str:
        lines = [f"Fit: {len(self.model.centres)} centres, residual energy {self.residual_db:.1f} dB "
                 f"(aspect-balanced), {self.model.meta.get('residual_db_unweighted', np.nan):.1f} dB (unweighted)"
                 + (f"; noise floor {self.model.meta['noise_db']:.1f} dB" if "noise_db" in self.model.meta else "")]
        shares = self.model.meta.get("energy_share_db", {})
        for p, v in shares.items():
            lines.append(f"   part {p:<20} energy share {v:7.1f} dB")
        return "\n".join(lines)


class _Atom:
    __slots__ = ("feature", "basis", "B", "K", "free", "D")

    def __init__(self, feature, basis, free):
        self.feature, self.basis, self.free = feature, basis, free
        self.B, self.K = [], []
        self.D = basis.difference_operator()


class SCFitter:
    def __init__(self, data, parts=None, config: FitConfig | None = None, reference=None):
        """``reference``: optional blocks (same grids as ``data``) whose power sets the
        aspect-balancing weights - e.g. the full-target data when ``data`` is a residual."""
        self.blocks = as_blocks(data)
        self.reference = None if reference is None else as_blocks(reference)
        self.parts = list(parts or [])
        self.cfg = config or FitConfig()
        chs = self.cfg.channels
        if chs is None:
            common = set(self.blocks[0].channels)
            for b in self.blocks[1:]:
                common &= set(b.channels)
            chs = tuple(c for c in self.blocks[0].channels if c in common)
        self.channels = tuple(canonical_channel(c) for c in chs)
        if not self.channels:
            raise ValueError("No common polarisation channel in the data blocks")
        self.raw_blocks = self.blocks
        allf = np.concatenate([b.freqs for b in self.blocks])
        self.f_ref = float(np.mean([allf.min(), allf.max()]))
        self.all_dirs = np.vstack([b.dirs for b in self.blocks])
        self.normal = plane_normal(self.all_dirs)
        self.point_basis = AngularBasis.for_directions(self.all_dirs, self.cfg.basis_spacing_deg)
        self.phys_basis = AngularBasis.for_directions(self.all_dirs, self.cfg.physical_basis_spacing_deg)
        self.noise = self._estimate_noise()
        if self.cfg.balance_aspects:
            refs = self.reference or self.blocks
            self.blocks = [self._balanced(b, nz, r) for b, nz, r in zip(self.blocks, self.noise, refs)]
        self.total_energy = sum(b.energy(self.channels) for b in self.blocks)
        self.atoms: list[_Atom] = []
        self.coeffs = None

    # ------------------------------------------------------------------ utils
    def _extent_radius(self):
        bs = [p.bounds() for p in self.parts if p.bounds() is not None]
        if self.cfg.grid_bounds is not None:
            bs.append(tuple(np.asarray(v, float) for v in self.cfg.grid_bounds))
        if not bs:
            return None
        lo = np.min([b[0] for b in bs], 0) - self.cfg.grid_margin
        hi = np.max([b[1] for b in bs], 0) + self.cfg.grid_margin
        return float(np.max(np.linalg.norm([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                            for z in (lo[2], hi[2])], axis=1)))

    def _estimate_noise(self):
        """Per-block noise variance (per sample, per channel) or None."""
        if self.cfg.noise_power is not None:
            return [float(self.cfg.noise_power)] * len(self.blocks)
        R = self._extent_radius()
        out = []
        for b in self.blocks:
            nz = None
            if "noise_power" in b.meta:
                nz = float(b.meta["noise_power"])
            elif R is not None:
                est = [estimate_noise_power(b, R, ch) for ch in self.channels]
                if all(e is not None for e in est):
                    nz = float(np.mean(est))
            out.append(nz)
        return out

    def _balanced(self, b: ScatteringData, noise=None, ref: ScatteringData | None = None) -> ScatteringData:
        """Re-weight so every aspect sector carries similar energy.

        RCS requirements are set per aspect sector in dB, so a strong specular
        flash must not dominate the fit at the expense of low-level aspects.
        Weight = 1 / (sector-smoothed power + floor), floor at
        ``balance_dynamic_range_db`` below the strongest sector.
        """
        ref = b if ref is None else ref
        P = sum(np.mean(b.w * np.abs(ref.S[ch]) ** 2, 0) for ch in self.channels)    # (Na,)
        B = self.point_basis.evaluate(b.dirs)
        Ps = B @ ((B.T @ P) / np.maximum(B.sum(0), 1e-30))
        floor = Ps.max() * 10 ** (-self.cfg.balance_dynamic_range_db / 10) + 1e-300
        if noise is not None:  # never boost aspects where the data are noise dominated
            floor = max(floor, 10.0 * noise * len(self.channels) * np.mean(b.w))
        wa = 1.0 / (Ps + floor)
        wa /= np.mean(wa)
        return b.copy(weights=b.w * wa[None, :])

    def _log(self, *a):
        if self.cfg.verbose:
            print(*a, flush=True)

    def _basis_for(self, feature):
        return self.phys_basis if feature.physical else self.point_basis

    def _make_atom(self, feature, free=False):
        at = _Atom(feature, self._basis_for(feature), free)
        for b in self.blocks:
            at.B.append(at.basis.evaluate(b.dirs))
            at.K.append(feature.kernel(b.freqs, b.dirs, self.f_ref))
        return at

    def _project_in_plane(self, x):
        if self.normal is None:
            return x
        return x - (x @ self.normal) * self.normal

    # --------------------------------------------------------------- candidates
    def _grid(self):
        cfg = self.cfg
        if cfg.grid_bounds is not None:
            lo, hi = map(lambda v: np.asarray(v, float), cfg.grid_bounds)
        else:
            bs = [p.bounds() for p in self.parts if p.bounds() is not None]
            if bs:
                lo = np.min([b[0] for b in bs], 0) - cfg.grid_margin
                hi = np.max([b[1] for b in bs], 0) + cfg.grid_margin
            else:
                fmin = [np.diff(np.sort(b.freqs)) for b in self.blocks if b.nf > 1]
                if not fmin:
                    raise ValueError("grid_bounds required for single-frequency data without parts")
                r = 0.45 * C0 / (2 * max(np.median(f) for f in fmin))
                lo, hi = -np.full(3, r), np.full(3, r)
        fmax = max(b.freqs.max() for b in self.blocks)
        bw = max(b.freqs.max() - b.freqs.min() for b in self.blocks)
        lam = C0 / fmax
        sp = np.deg2rad(min(self.cfg.basis_spacing_deg, 90))
        res = lam / (4 * sp)
        if bw > 0:
            res = min(res, C0 / (2 * bw))
        h = cfg.grid_spacing or res / 1.5
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        if self.normal is not None:
            n = self.normal
            u1 = unit(np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0]))
            u2 = np.cross(n, u1)
            axes = [u1, u2]
        else:
            axes = [np.eye(3)[i] for i in range(3)]
        while True:
            ticks = []
            for a in axes:
                p = corners @ a
                ticks.append(np.arange(p.min(), p.max() + 1e-9, h))
            npts = int(np.prod([t.size for t in ticks]))
            if npts <= cfg.max_grid_points:
                break
            h *= (npts / cfg.max_grid_points) ** (1 / len(axes)) * 1.01
        mesh = np.meshgrid(*ticks, indexing="ij")
        X = sum(m.reshape(-1, 1) * a for m, a in zip(mesh, axes))
        self.grid_spacing = h
        self._log(f"free grid: {X.shape[0]} points, spacing {h:.3f} m")
        return X

    # ------------------------------------------------------------ correlations
    def _point_corr(self, X, alpha, residual):
        """Windowed matched-filter score for point candidates at X (Nc,3)."""
        B = self.point_basis
        num = {ch: 0 for ch in self.channels}
        den = 0
        for ib, b in enumerate(self.blocks):
            Bb = B.evaluate(b.dirs)                      # (Na,Nb)
            ff = (b.freqs / self.f_ref) ** alpha
            w = b.w * ff[:, None]
            den = den + (np.sum(b.w * ff[:, None] ** 2, 0) @ Bb)
            proj = X @ b.dirs.T                           # (Nc, Na) range d = r.x
            uniform = b.nf > 2 and np.allclose(np.diff(b.freqs), b.freqs[1] - b.freqs[0], rtol=1e-6)
            for ch in self.channels:
                Y = w * residual[ib][ch]                  # (Nf,Na)
                if uniform:
                    C = self._fft_backproject(b, Y, proj)
                else:
                    k = b.k
                    C = np.zeros(proj.shape, complex)
                    for i in range(b.nf):
                        C += Y[i][None, :] * np.exp(-2j * k[i] * proj)
                num[ch] = num[ch] + C @ Bb
        den = np.maximum(den, 1e-30)
        return sum(np.sum(np.abs(num[ch]) ** 2 / den[None, :], 1) for ch in self.channels)

    def _fft_backproject(self, b, Y, proj):
        nf = b.nf
        M = int(2 ** np.ceil(np.log2(nf * self.cfg.fft_oversample)))
        df = b.freqs[1] - b.freqs[0]
        dk2 = 4 * np.pi * df / C0
        period = 2 * np.pi / dk2
        P = np.fft.fft(Y, n=M, axis=0)                     # p(d_m), d_m = m*period/M
        P = np.vstack([P, P[:1]])
        t = np.mod(proj, period) / period * M
        i0 = np.minimum(np.floor(t).astype(int), M - 1)
        fr = t - i0
        cols = np.arange(proj.shape[1])[None, :]
        val = P[i0, cols] * (1 - fr) + P[i0 + 1, cols] * fr
        return val * np.exp(-2j * b.k[0] * proj)

    def _atom_score(self, feature, residual):
        at = self._make_atom(feature)
        Nb = at.basis.size
        num = {ch: np.zeros(Nb, complex) for ch in self.channels}
        den = np.zeros(Nb)
        for ib, b in enumerate(self.blocks):
            wK = b.w * np.conj(at.K[ib])
            den += np.sum(b.w * np.abs(at.K[ib]) ** 2, 0) @ at.B[ib]
            for ch in self.channels:
                num[ch] += np.sum(wK * residual[ib][ch], 0) @ at.B[ib]
        den = np.maximum(den, 1e-30)
        return float(sum(np.sum(np.abs(num[ch]) ** 2 / den) for ch in self.channels))

    # ----------------------------------------------------------- least squares
    def _solve(self):
        atoms = self.atoms
        sizes = [a.basis.size for a in atoms]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        n = offs[-1]
        G = np.zeros((n, n), complex)
        rhs = np.zeros((n, len(self.channels)), complex)
        for ib, b in enumerate(self.blocks):
            wK = [b.w * np.conj(a.K[ib]) for a in atoms]
            for i, ai in enumerate(atoms):
                for ci, ch in enumerate(self.channels):
                    rhs[offs[i]:offs[i + 1], ci] += ai.B[ib].T @ np.sum(wK[i] * b.S[ch], 0)
                for j in range(i, len(atoms)):
                    P = np.sum(wK[i] * atoms[j].K[ib], 0)
                    blk = ai.B[ib].T @ (P[:, None] * atoms[j].B[ib])
                    G[offs[i]:offs[i + 1], offs[j]:offs[j + 1]] += blk
                    if j != i:
                        G[offs[j]:offs[j + 1], offs[i]:offs[i + 1]] += blk.conj().T
        Reg = np.zeros((n, n))
        for i, a in enumerate(atoms):
            sl = slice(offs[i], offs[i + 1])
            scale = np.real(np.trace(G[sl, sl])) / sizes[i] + 1e-30
            Reg[sl, sl] = scale * (self.cfg.smoothness * a.D.T @ a.D + self.cfg.ridge * np.eye(sizes[i]))
        A = G + Reg
        try:
            X = cho_solve(cho_factor(A), rhs)
        except np.linalg.LinAlgError:
            X = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.coeffs = [X[offs[i]:offs[i + 1]] for i in range(len(atoms))]
        return self.coeffs

    def _atom_field(self, i, ib):
        a = self.atoms[i]
        return {ch: a.K[ib] * (a.B[ib] @ self.coeffs[i][:, ci])[None, :] for ci, ch in enumerate(self.channels)}

    def _residual(self):
        res = []
        for ib, b in enumerate(self.blocks):
            r = {ch: b.S[ch].copy() for ch in self.channels}
            for i in range(len(self.atoms)):
                f = self._atom_field(i, ib)
                for ch in self.channels:
                    r[ch] -= f[ch]
            res.append(r)
        return res

    def _energy(self, res):
        return sum(np.sum(b.w * np.abs(r[ch]) ** 2) for b, r in zip(self.blocks, res) for ch in self.channels)

    # ------------------------------------------------------------- main loop
    def fit(self) -> FitResult:
        cfg = self.cfg
        if self.total_energy <= 0:
            raise ValueError("Data energy is zero")
        # candidate pool
        part_cands = []
        for p in self.parts:
            for f in p.features:
                if f.physical:
                    part_cands.append(f)
                else:
                    for al in cfg.alphas:
                        part_cands.append(f.with_alpha(al))
        grid = self._grid() if cfg.free_points else np.zeros((0, 3))
        grid_alive = {al: np.ones(grid.shape[0], bool) for al in cfg.alphas}
        used_part = set()
        history = []
        residual = [{ch: b.S[ch].copy() for ch in self.channels} for b in self.blocks]
        E = self._energy(residual)
        self._log(f"data energy {E:.3e}; channels {self.channels}; plane normal {self.normal}")
        if all(nz is not None for nz in self.noise):
            # expected residual energy of pure noise, and per-parameter noise-fit energy
            self.noise_energy = sum(nz * np.sum(b.w) * len(self.channels) for nz, b in zip(self.noise, self.blocks))
            self.noise_per_param = sum(nz * np.mean(b.w) for nz, b in zip(self.noise, self.blocks)) \
                * len(self.channels)
            self._log(f"noise estimate: {10 * np.log10(self.noise_energy / self.total_energy):.1f} dB "
                      "relative to (weighted) data energy")
        else:
            self.noise_energy = self.noise_per_param = None
        for it in range(cfg.max_centres):
            best, best_score, best_info = None, -1.0, None
            for ic, f in enumerate(part_cands):
                if (f.part, f.name, id(f)) in used_part or any(
                        (f.part, f.name) == k[:2] for k in used_part):
                    continue
                s = self._atom_score(f, residual) * (1 + cfg.part_bias)
                if s > best_score:
                    best, best_score, best_info = f, s, ("part", ic)
            for al in cfg.alphas:
                if not grid_alive[al].any():
                    continue
                idx = np.where(grid_alive[al])[0]
                sc = np.empty(idx.size)
                for c0 in range(0, idx.size, 2000):
                    sl = idx[c0:c0 + 2000]
                    sc[c0:c0 + 2000] = self._point_corr(grid[sl], al, residual)
                j = int(np.argmax(sc))
                if sc[j] > best_score:
                    best_score = sc[j]
                    best = PointFeature(grid[idx[j]], "unassigned", f"pt{it}", al)
                    best_info = ("grid", al, idx[j])
            if best is None:
                break
            self.atoms.append(self._make_atom(best, free=best_info[0] == "grid"))
            if cfg.refine and best_info[0] == "grid":
                self.coeffs = None
                self._refine_atom(len(self.atoms) - 1, residual, own_zero=True)
            if best_info[0] == "part":
                used_part.add((best.part, best.name, id(best)))
            else:
                al, j = best_info[1], best_info[2]
                near = np.linalg.norm(grid - grid[j], axis=1) < 0.75 * self.grid_spacing
                grid_alive[al] &= ~near
            self._solve()
            residual = self._residual()
            E_new = self._energy(residual)
            gain = (E - E_new) / self.total_energy
            E = E_new
            rdb = 10 * np.log10(E / self.total_energy + 1e-30)
            history.append({"n": len(self.atoms), "residual_db": rdb, "added": repr(best)})
            self._log(f"[{len(self.atoms):3d}] {best!r:<55} residual {rdb:7.2f} dB")
            if rdb < cfg.target_residual_db:
                break
            noise_stop = (self.noise_per_param is not None and
                          gain * self.total_energy < cfg.noise_gain_factor * self.noise_per_param
                          * self.atoms[-1].basis.size)
            if noise_stop:
                self._log("new centre is not significant above the noise - stopping")
            if noise_stop or gain < 10 ** (cfg.min_gain_db / 10):
                self._log("gain below threshold - stopping")
                self.atoms.pop()
                self._solve()
                residual = self._residual()
                break
        if cfg.refine and self.atoms:
            residual = self._refine(residual)
        residual = self._prune(residual)
        return self._finish(residual, history)

    # ------------------------------------------------------------- refinement
    def _refine_atom(self, i, residual, own_zero=False):
        """Locally re-position atom i against residual + its own contribution."""
        a = self.atoms[i]
        if not a.feature.refinable:
            return False
        max_shift = 1.5 * getattr(self, "grid_spacing", 0.1) if a.free else self.cfg.part_feature_shift
        if max_shift <= 0:
            return False
        if own_zero:
            r_i = residual
        else:
            own = [self._atom_field(i, ib) for ib in range(len(self.blocks))]
            r_i = [{ch: residual[ib][ch] + own[ib][ch] for ch in self.channels} for ib in range(len(self.blocks))]
        f0 = a.feature
        x0 = f0.get_vector()
        n = self.normal
        if n is not None:
            u1 = unit(np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0]))
            axes = np.stack([u1, np.cross(n, u1)])
        else:
            axes = np.eye(3)

        def make(v):
            return f0.with_vector(x0 + v @ axes)

        def obj(v):
            if np.linalg.norm(v) > max_shift:
                return 1e3 * np.linalg.norm(v)
            return -self._atom_score(make(v), r_i) / (self.total_energy + 1e-30)

        h = max_shift / 3
        dim = axes.shape[0]
        simplex = np.vstack([np.zeros(dim), h * np.eye(dim)])
        f_start = obj(np.zeros(dim))
        res = minimize(obj, np.zeros(dim), method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-4 * max_shift, "fatol": 1e-10,
                                "maxiter": 60 * dim})
        if res.fun < f_start:
            name = f0.name
            self.atoms[i] = self._make_atom(make(res.x), a.free)
            self.atoms[i].feature.name = name
            return True
        return False

    def _refine(self, residual):
        cfg = self.cfg
        for p in range(cfg.refine_passes):
            moved = 0
            for i in range(len(self.atoms)):
                moved += self._refine_atom(i, residual)
            self._solve()
            residual = self._merge(self._residual())
            self._log(f"refine pass {p + 1}: moved {moved}, residual "
                      f"{10 * np.log10(self._energy(residual) / self.total_energy):.2f} dB")
        return residual

    def _merge(self, residual):
        """Merge free point centres that converged onto the same location."""
        tol = 0.3 * getattr(self, "grid_spacing", 0.0)
        drop = set()
        for i, a in enumerate(self.atoms):
            if i in drop or not isinstance(a.feature, PointFeature):
                continue
            for j in range(i + 1, len(self.atoms)):
                b = self.atoms[j]
                if j in drop or not b.free or not isinstance(b.feature, PointFeature) \
                        or b.feature.alpha != a.feature.alpha:
                    continue
                if np.linalg.norm(a.feature.position - b.feature.position) < tol:
                    drop.add(j)
        if drop:
            self.atoms = [a for k, a in enumerate(self.atoms) if k not in drop]
            self._solve()
            residual = self._residual()
            self._log(f"merged {len(drop)} coincident centres")
        return residual

    def _prune(self, residual):
        if not self.atoms:
            return residual
        thr = self.total_energy * 10 ** (self.cfg.prune_db / 10)
        keep = []
        for i in range(len(self.atoms)):
            e = sum(np.sum(b.w * np.abs(self._atom_field(i, ib)[ch]) ** 2)
                    for ib, b in enumerate(self.blocks) for ch in self.channels)
            keep.append(e >= thr)
        if not all(keep):
            self._log(f"pruning {keep.count(False)} centres")
            self.atoms = [a for a, k in zip(self.atoms, keep) if k]
            self._solve()
            residual = self._residual()
        return residual

    def _finish(self, residual, history):
        centres = []
        for i, a in enumerate(self.atoms):
            f = a.feature
            if a.free:
                f.part = assign_part(f.reference_point if self.normal is None else f.reference_point,
                                     self.parts, self.cfg.assign_radius)
                f.name = f"{f.part}:pt{i}"
            centres.append(ScatteringCentre(f, a.basis, {ch: self.coeffs[i][:, ci].copy()
                                                         for ci, ch in enumerate(self.channels)}))
        cov = self.all_dirs
        if cov.shape[0] > 4000:
            cov = cov[np.linspace(0, cov.shape[0] - 1, 4000).astype(int)]
        allf = np.concatenate([b.freqs for b in self.blocks])
        model = ScatteringCentreModel(centres, self.channels, self.f_ref, "fitted", cov.copy(),
                                      (float(allf.min()), float(allf.max())))
        # energy share per part
        share = {}
        for p in model.parts:
            sub = model.subset(p)
            e = sum(np.sum(b.w * np.abs(v) ** 2) for b in self.blocks
                    for v in sub.evaluate(b.freqs, b.dirs, self.channels).values())
            share[p] = float(10 * np.log10(e / self.total_energy + 1e-30))
        model.meta["energy_share_db"] = share
        model.meta["plane_normal"] = None if self.normal is None else self.normal.tolist()
        rdb = float(10 * np.log10(self._energy(residual) / self.total_energy + 1e-30))
        model.meta["residual_db"] = rdb
        if self.noise_energy is not None:
            model.meta["noise_db"] = float(10 * np.log10(self.noise_energy / self.total_energy))
        raw = sum(b.energy(self.channels) for b in self.raw_blocks)
        res_raw = sum(np.sum(rb.w * np.abs(r[ch]) ** 2) for rb, r in zip(self.raw_blocks, residual)
                      for ch in self.channels)
        model.meta["residual_db_unweighted"] = float(10 * np.log10(res_raw / raw + 1e-30))
        res_blocks = [b.copy(S=r, name=f"{b.name}:residual") for b, r in zip(self.raw_blocks, residual)]
        return FitResult(model, rdb, history, res_blocks, self.cfg)


def fit_scattering_centres(data, parts=None, **kwargs) -> FitResult:
    """Convenience wrapper: ``fit_scattering_centres(data, parts, max_centres=40, ...)``."""
    cfg = kwargs.pop("config", None) or FitConfig(**kwargs)
    return SCFitter(data, parts, cfg).fit()

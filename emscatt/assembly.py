"""Coherent assembly of component scattering-centre models into a full target.

Each component model is fitted in its *own* measurement frame (e.g. on the
turntable/compact range).  A :class:`Pose` places it in the global (vehicle)
frame.  For every global observation direction r_g:

    r_l   = R^T r_g                                   (local aspect)
    S_l   = component model evaluated at (f, r_l)      (2x2 per channel)
    S_g   = T S_l T^T * exp(j 2k r_g . t)             (polarisation + translation)

and the total is the coherent sum over components, plus an optional
*interaction* model (global frame) that captures everything that is not a
superposition of isolated components - multiple bounces between components,
blockage/shadowing, creeping waves and installation effects.  The interaction
model is calibrated once against a full-target measurement or simulation and
kept fixed while individual components are iterated, which is the key
assumption to verify (see docs/METHODOLOGY.md).
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np

from .conventions import CHANNELS, Pose, canonical_channel, polarisation_transform, wavenumber
from .data import ScatteringData, as_blocks
from .model import ScatteringCentreModel


@dataclass
class Component:
    name: str
    model: ScatteringCentreModel
    pose: Pose = field(default_factory=Pose)
    enabled: bool = True


def _full_matrix(Sd: dict, shape):
    """Assemble tt/tp/pt/pp arrays, using reciprocity for a missing cross term."""
    z = None
    out = {}
    for c in CHANNELS:
        if c in Sd:
            out[c] = Sd[c]
    if "tp" in out and "pt" not in out:
        out["pt"] = out["tp"]
    if "pt" in out and "tp" not in out:
        out["tp"] = out["pt"]
    for c in CHANNELS:
        if c not in out:
            if z is None:
                z = np.zeros(shape, complex)
            out[c] = z
    return out


class Assembly:
    def __init__(self, components=(), interaction: ScatteringCentreModel | None = None, name="assembly"):
        self.components = list(components)
        self.interaction = interaction
        self.name = name

    # ----------------------------------------------------------- management
    def add(self, name, model, pose=None):
        self.components.append(Component(name, model, pose or Pose()))
        return self

    def __getitem__(self, name) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    def replace(self, name, model=None, pose=None) -> "Assembly":
        """Copy of the assembly with one component's model and/or pose swapped
        (design iteration).  Interaction terms are retained."""
        new = Assembly([copy.copy(c) for c in self.components], self.interaction, self.name)
        c = new[name]
        if model is not None:
            c.model = model
        if pose is not None:
            c.pose = pose
        return new

    def without(self, *names) -> "Assembly":
        new = Assembly([copy.copy(c) for c in self.components], self.interaction, self.name)
        for c in new.components:
            if c.name in names:
                c.enabled = False
        return new

    # ----------------------------------------------------------- evaluation
    def component_field(self, comp: Component, freqs, dirs, channels) -> dict:
        freqs = np.atleast_1d(freqs)
        dirs = np.atleast_2d(dirs)
        pose = comp.pose
        dl = pose.to_local_dirs(dirs)
        k = wavenumber(freqs)
        trans = np.exp(2j * k[:, None] * (dirs @ pose.translation)[None, :])
        T = polarisation_transform(pose, dirs)
        trivial = np.allclose(T, np.eye(2)[None], atol=1e-9)
        if trivial:
            have = [c for c in channels if c in comp.model.channels]
            loc = comp.model.evaluate(freqs, dl, have)
            out = {c: (loc[c] * trans if c in loc else np.zeros((freqs.size, dirs.shape[0]), complex))
                   for c in channels}
            return out
        loc = comp.model.evaluate(freqs, dl)
        mixing = np.max(np.abs(T[..., 0, 1]))
        missing = [c for c in ("tt", "pp") if c not in comp.model.channels]
        if missing and mixing > 0.05:
            warnings.warn(f"Component '{comp.name}': pose rotates the polarisation basis by up to "
                          f"{np.rad2deg(np.arcsin(min(mixing, 1))):.1f} deg but channels {missing} were not "
                          "measured; treating them as zero. Supply full-polarisation data for this pose.")
        Sm = _full_matrix(loc, (freqs.size, dirs.shape[0]))
        S = np.empty((freqs.size, dirs.shape[0], 2, 2), complex)
        S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1] = Sm["tt"], Sm["tp"], Sm["pt"], Sm["pp"]
        Sg = np.einsum("aij,fajk,alk->fail", T, S, T)
        idx = {"tt": (0, 0), "tp": (0, 1), "pt": (1, 0), "pp": (1, 1)}
        return {c: Sg[..., idx[c][0], idx[c][1]] * trans for c in channels}

    def evaluate(self, freqs, dirs, channels=("tt",), incoherent=False, include_interaction=True,
                 per_component=False):
        """Total scattering amplitude {channel: (Nf, Na)}.

        ``incoherent=True`` returns sqrt(sum |S_i|^2) (zero-phase), the classical
        power-sum estimate usable when only RCS magnitudes are known.
        ``per_component=True`` returns {component_name: {channel: array}} instead.
        """
        chs = tuple(canonical_channel(c) for c in channels)
        freqs = np.atleast_1d(freqs)
        dirs = np.atleast_2d(dirs)
        parts = {}
        for comp in self.components:
            if comp.enabled:
                parts[comp.name] = self.component_field(comp, freqs, dirs, chs)
        if include_interaction and self.interaction is not None:
            have = [c for c in chs if c in self.interaction.channels]
            ev = self.interaction.evaluate(freqs, dirs, have)
            parts["interaction"] = {c: ev.get(c, np.zeros((freqs.size, dirs.shape[0]), complex)) for c in chs}
        if per_component:
            return parts
        out = {c: np.zeros((freqs.size, dirs.shape[0]), complex if not incoherent else float) for c in chs}
        for fld in parts.values():
            for c in chs:
                out[c] = out[c] + (np.abs(fld[c]) ** 2 if incoherent else fld[c])
        if incoherent:
            out = {c: np.sqrt(v).astype(complex) for c, v in out.items()}
        return out

    def predict(self, data: ScatteringData, **kw) -> ScatteringData:
        return data.copy(S=self.evaluate(data.freqs, data.dirs, data.channels, **kw),
                         name=f"{self.name}:prediction")

    # ---------------------------------------------------------- diagnostics
    def coverage(self, dirs, tol_deg: float = 3.0, freqs=None) -> dict:
        """For each component, boolean mask of global directions whose local aspect
        lies outside the component's training data (extrapolation) by > tol_deg,
        or (if ``freqs`` given) frequencies outside the training band."""
        dirs = np.atleast_2d(dirs)
        out = {}
        for comp in self.components:
            cov = comp.model.coverage_dirs
            if cov is None:
                out[comp.name] = np.zeros(dirs.shape[0], bool)
                continue
            dl = comp.pose.to_local_dirs(dirs)
            best = np.full(dirs.shape[0], -1.0)
            for i0 in range(0, cov.shape[0], 2000):
                best = np.maximum(best, np.max(dl @ cov[i0:i0 + 2000].T, axis=1))
            mask = np.rad2deg(np.arccos(np.clip(best, -1, 1))) > tol_deg
            if freqs is not None and comp.model.freq_range is not None:
                lo, hi = comp.model.freq_range
                if np.any((np.asarray(freqs) < lo * 0.999) | (np.asarray(freqs) > hi * 1.001)):
                    warnings.warn(f"Component '{comp.name}': frequencies outside the fitted band "
                                  f"[{lo / 1e9:.3f}, {hi / 1e9:.3f}] GHz (alpha extrapolation)")
            out[comp.name] = mask
        return out

    def calibrate_interactions(self, full_data, parts=None, config=None, **kw):
        """Fit the residual (full measurement - coherent component sum) with a
        scattering-centre model in the global frame and store it as the
        interaction term.  Returns the FitResult."""
        from .fitting import FitConfig, SCFitter
        blocks = as_blocks(full_data)
        resid = []
        for b in blocks:
            pred = self.evaluate(b.freqs, b.dirs, b.channels, include_interaction=False)
            resid.append(b.copy(S={c: b.S[c] - pred[c] for c in b.channels}, name=f"{b.name}:interaction"))
        cfg = config or FitConfig(**kw)
        res = SCFitter(resid, parts, cfg, reference=blocks).fit()
        res.model.name = "interaction"
        self.interaction = res.model
        return res

    def summary(self) -> str:
        lines = [f"Assembly '{self.name}':"]
        for c in self.components:
            t = c.pose.translation
            lines.append(f"  {'+' if c.enabled else '-'} {c.name:<16} {len(c.model.centres):3d} centres  "
                         f"t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  {'mirror' if not c.pose.is_proper else ''}")
        if self.interaction is not None:
            lines.append(f"  + interaction     {len(self.interaction.centres):3d} centres")
        return "\n".join(lines)

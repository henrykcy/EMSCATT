"""Scattering-centre model: a list of (feature, angular amplitude) centres."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .basis import AngularBasis
from .conventions import canonical_channel, Pose
from .data import ScatteringData
from .features import Feature, feature_from_dict


@dataclass
class ScatteringCentre:
    feature: Feature
    basis: AngularBasis
    coeffs: dict                      # channel -> complex (Nb,)

    @property
    def part(self):
        return self.feature.part

    def amplitude(self, dirs, channel) -> np.ndarray:
        c = self.coeffs.get(channel)
        if c is None:
            return np.zeros(np.atleast_2d(dirs).shape[0], complex)
        return self.basis.evaluate(dirs) @ c

    def field(self, freqs, dirs, f_ref, channels) -> dict:
        K = self.feature.kernel(freqs, dirs, f_ref)
        B = self.basis.evaluate(dirs)
        return {ch: K * (B @ self.coeffs[ch])[None, :] if ch in self.coeffs else np.zeros_like(K)
                for ch in channels}

    def peak_rcs_dbsm(self, freqs, dirs, f_ref) -> float:
        K = self.feature.kernel(freqs, dirs, f_ref)
        B = self.basis.evaluate(dirs)
        p = max(np.max(np.abs(K * (B @ c)[None, :]) ** 2) for c in self.coeffs.values())
        return float(10 * np.log10(p + 1e-30))

    def to_dict(self):
        return {"feature": self.feature.to_dict(), "basis": self.basis.to_dict(),
                "coeffs": {c: [v.real.tolist(), v.imag.tolist()] for c, v in self.coeffs.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(feature_from_dict(d["feature"]), AngularBasis.from_dict(d["basis"]),
                   {c: np.array(v[0]) + 1j * np.array(v[1]) for c, v in d["coeffs"].items()})


@dataclass
class ScatteringCentreModel:
    centres: list
    channels: tuple
    f_ref: float
    name: str = "model"
    coverage_dirs: np.ndarray | None = None      # training directions (model frame)
    freq_range: tuple | None = None
    meta: dict = field(default_factory=dict)

    # --- evaluation ----------------------------------------------------------
    def evaluate(self, freqs, dirs, channels=None, parts=None) -> dict:
        """Complex scattering amplitude {channel: (Nf, Na)} in the model frame."""
        chs = self.channels if channels is None else tuple(canonical_channel(c) for c in channels)
        freqs = np.atleast_1d(freqs)
        dirs = np.atleast_2d(dirs)
        out = {c: np.zeros((freqs.size, dirs.shape[0]), complex) for c in chs}
        for sc in self.centres:
            if parts is not None and sc.part not in parts:
                continue
            fld = sc.field(freqs, dirs, self.f_ref, chs)
            for c in chs:
                out[c] += fld[c]
        return out

    def predict(self, data: ScatteringData, parts=None) -> ScatteringData:
        chs = [c for c in data.channels if c in self.channels]
        return data.copy(S=self.evaluate(data.freqs, data.dirs, chs, parts), name=f"{self.name}:prediction")

    # --- organisation -------------------------------------------------------
    @property
    def parts(self) -> list:
        seen = []
        for sc in self.centres:
            if sc.part not in seen:
                seen.append(sc.part)
        return seen

    def subset(self, parts) -> "ScatteringCentreModel":
        parts = [parts] if isinstance(parts, str) else list(parts)
        return ScatteringCentreModel([c for c in self.centres if c.part in parts], self.channels, self.f_ref,
                                     f"{self.name}[{','.join(parts)}]", self.coverage_dirs, self.freq_range,
                                     dict(self.meta))

    def part_contributions(self, freqs, dirs, channel=None) -> dict:
        ch = self.channels[0] if channel is None else canonical_channel(channel)
        return {p: self.evaluate(freqs, dirs, [ch], [p])[ch] for p in self.parts}

    def transformed(self, pose: Pose, name=None) -> "ScatteringCentreModel":
        """Geometric transform into a parent frame.

        Exact for co-polar channels when the pose does not rotate the
        polarisation basis (e.g. yaw + translation measured in an azimuth cut).
        In general use :class:`emscatt.assembly.Assembly`, which applies the
        full polarisation rotation.
        """
        cs = [ScatteringCentre(sc.feature.transformed(pose), sc.basis.transformed(pose.rotation), dict(sc.coeffs))
              for sc in self.centres]
        cov = None if self.coverage_dirs is None else pose.apply_vectors(self.coverage_dirs)
        return ScatteringCentreModel(cs, self.channels, self.f_ref, name or self.name, cov, self.freq_range,
                                     dict(self.meta))

    def summary(self, freqs=None, dirs=None) -> str:
        freqs = np.array([self.f_ref]) if freqs is None else np.atleast_1d(freqs)
        if dirs is None:
            dirs = self.coverage_dirs if self.coverage_dirs is not None else np.eye(3)
        lines = [f"Scattering-centre model '{self.name}': {len(self.centres)} centres, channels {self.channels}, "
                 f"f_ref = {self.f_ref / 1e9:.3f} GHz",
                 f"{'#':>3} {'part':<18}{'kind':<10}{'alpha':>6} {'location (m)':>28} {'peak dBsm':>10}"]
        for i, sc in enumerate(self.centres):
            loc = sc.feature.reference_point
            lines.append(f"{i:>3} {sc.part[:17]:<18}{sc.feature.kind:<10}{sc.feature.alpha:>6.2g} "
                         f"({loc[0]:8.3f},{loc[1]:8.3f},{loc[2]:8.3f}) {sc.peak_rcs_dbsm(freqs, dirs, self.f_ref):>10.1f}")
        return "\n".join(lines)

    # --- persistence --------------------------------------------------------
    def to_dict(self):
        return {"name": self.name, "channels": list(self.channels), "f_ref": self.f_ref,
                "freq_range": None if self.freq_range is None else list(self.freq_range),
                "coverage_dirs": None if self.coverage_dirs is None else self.coverage_dirs.tolist(),
                "meta": self.meta, "centres": [c.to_dict() for c in self.centres]}

    @classmethod
    def from_dict(cls, d):
        return cls([ScatteringCentre.from_dict(c) for c in d["centres"]], tuple(d["channels"]), d["f_ref"],
                   d.get("name", "model"),
                   None if d.get("coverage_dirs") is None else np.array(d["coverage_dirs"]),
                   None if d.get("freq_range") is None else tuple(d["freq_range"]), d.get("meta", {}))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

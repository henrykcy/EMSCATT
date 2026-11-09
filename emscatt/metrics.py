"""RCS comparison metrics used by RCS engineers."""
from __future__ import annotations

import numpy as np


def to_dbsm(S, floor_db=-100.0):
    return np.maximum(10 * np.log10(np.abs(S) ** 2 + 1e-30), floor_db)


def relative_error_db(ref, test) -> float:
    """10 log10(||test - ref||^2 / ||ref||^2) - coherent (field) error."""
    ref, test = np.asarray(ref), np.asarray(test)
    return float(10 * np.log10(np.sum(np.abs(test - ref) ** 2) / np.sum(np.abs(ref) ** 2) + 1e-30))


def rms_db_error(ref, test, dynamic_range_db=40.0) -> float:
    """RMS difference of RCS in dB, both clipped to (peak(ref) - dynamic range)."""
    r, t = to_dbsm(ref), to_dbsm(test)
    floor = r.max() - dynamic_range_db
    return float(np.sqrt(np.mean((np.maximum(r, floor) - np.maximum(t, floor)) ** 2)))


def sector_statistics(angles_deg, S, sector_deg=10.0, axis=-1):
    """Sector-averaged RCS (linear mean -> dBsm), median and 90th percentile per sector.

    ``S`` may be (Nf, Na) (frequencies pooled) or (Na,).  Returns dict of arrays.
    """
    a = np.mod(np.asarray(angles_deg), 360)
    sig = np.abs(np.asarray(S)) ** 2
    if sig.ndim == 1:
        sig = sig[None, :]
    edges = np.arange(0, 360 + 1e-9, sector_deg)
    centres, mean, med, p90 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a < hi)
        if not m.any():
            continue
        v = sig[:, m].ravel()
        centres.append((lo + hi) / 2)
        mean.append(10 * np.log10(v.mean() + 1e-30))
        med.append(10 * np.log10(np.median(v) + 1e-30))
        p90.append(10 * np.log10(np.percentile(v, 90) + 1e-30))
    return {"centre_deg": np.array(centres), "mean_dbsm": np.array(mean), "median_dbsm": np.array(med),
            "p90_dbsm": np.array(p90)}


def compare(ref, test, angles_deg=None, sector_deg=10.0) -> dict:
    out = {"field_error_db": relative_error_db(ref, test), "rms_db_error": rms_db_error(ref, test)}
    if angles_deg is not None:
        a = sector_statistics(angles_deg, ref, sector_deg)
        b = sector_statistics(angles_deg, test, sector_deg)
        d = b["mean_dbsm"] - a["mean_dbsm"]
        out["sector_mean_error_db_rms"] = float(np.sqrt(np.mean(d ** 2)))
        out["sector_mean_error_db_max"] = float(np.max(np.abs(d)))
    return out

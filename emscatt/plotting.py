"""Matplotlib helpers (optional dependency)."""
from __future__ import annotations

import numpy as np

from .metrics import sector_statistics, to_dbsm

PART_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf", "#393b79", "#ad494a", "#637939", "#7b4173", "#3182bd", "#e6550d"]


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def plot_rcs_cut(angles_deg, curves: dict, ax=None, freq_index=None, title=None, ylim=None, lw=1.0):
    """Plot RCS (dBsm) vs angle. ``curves``: label -> S (Na,) or (Nf, Na).
    For (Nf, Na) arrays, ``freq_index`` selects a frequency (default: centre)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    for i, (lab, S) in enumerate(curves.items()):
        S = np.asarray(S)
        if S.ndim == 2:
            S = S[S.shape[0] // 2 if freq_index is None else freq_index]
        ax.plot(angles_deg, to_dbsm(S), lw=lw, label=lab, alpha=0.9 if i == 0 else 0.8)
    if ylim is None:
        top = max(np.max(to_dbsm(np.asarray(S))) for S in curves.values())
        ylim = (top - 70, top + 5)
    ax.set_xlabel("aspect angle (deg)")
    ax.set_ylabel("RCS (dBsm)")
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=3)
    if title:
        ax.set_title(title)
    return ax


def plot_sector_means(angles_deg, curves: dict, sector_deg=10.0, ax=None, title=None):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    top = -np.inf
    for lab, S in curves.items():
        st = sector_statistics(angles_deg, S, sector_deg)
        ax.step(st["centre_deg"], st["mean_dbsm"], where="mid", label=lab)
        top = max(top, st["mean_dbsm"].max())
    ax.set_ylim(top - 70, top + 5)
    ax.set_xlabel("aspect angle (deg)")
    ax.set_ylabel(f"{sector_deg:g}-deg sector mean RCS (dBsm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_sc_map(model, ax=None, image=None, image_extent=None, dirs=None, plane=(0, 1), title=None,
                dynamic_range_db=40.0, sizes=(10, 250), outline=None):
    """Top-view map of scattering centres coloured by part and sized by peak RCS.

    ``image`` - optional |back-projection| (Nu, Nv) in dB for the background.
    ``outline`` - optional list of (N,3) polylines (geometry) to draw.
    Sliding centres are drawn as their specular-point trajectory over ``dirs``.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    i, j = plane
    if image is not None:
        im = 20 * np.log10(np.abs(image) / np.abs(image).max() + 1e-12)
        ax.imshow(im.T, origin="lower", extent=image_extent, cmap="Greys", vmin=-dynamic_range_db, vmax=0,
                  aspect="equal")
    if outline is not None:
        for poly in outline:
            P = np.asarray(poly)
            ax.plot(P[:, i], P[:, j], color="0.55", lw=0.8)
    dirs = model.coverage_dirs if dirs is None else dirs
    peaks = np.array([sc.peak_rcs_dbsm(np.array([model.f_ref]), dirs, model.f_ref) for sc in model.centres])
    if peaks.size == 0:
        return ax
    pmax = peaks.max()
    parts = model.parts
    col = {p: PART_COLOURS[k % len(PART_COLOURS)] for k, p in enumerate(parts)}
    labelled = set()
    for sc, pk in zip(model.centres, peaks):
        s = sizes[0] + (sizes[1] - sizes[0]) * np.clip((pk - pmax + dynamic_range_db) / dynamic_range_db, 0, 1)
        lab = sc.part if sc.part not in labelled else None
        labelled.add(sc.part)
        f = sc.feature
        if f.kind == "ellipsoid":
            loc = f.location(dirs)
            ax.plot(loc[:, i], loc[:, j], "-", color=col[sc.part], lw=2.5, label=lab)
        elif f.kind == "cylinder":
            p1, p2 = f.centre - f.length / 2 * f.axis, f.centre + f.length / 2 * f.axis
            ax.plot([p1[i], p2[i]], [p1[j], p2[j]], "-", color=col[sc.part], lw=4, alpha=0.8, label=lab)
        elif f.kind == "edge":
            p1, p2 = f.endpoints
            ax.plot([p1[i], p2[i]], [p1[j], p2[j]], "-", color=col[sc.part], lw=3, label=lab)
        elif f.kind == "plate":
            V = np.vstack([f.vertices, f.vertices[:1]])
            ax.fill(V[:, i], V[:, j], color=col[sc.part], alpha=0.35)
            ax.plot(V[:, i], V[:, j], color=col[sc.part], lw=2.5, label=lab)
        else:
            ax.scatter(f.position[i], f.position[j], s=s, color=col[sc.part], edgecolor="k", lw=0.5,
                       label=lab, zorder=5)
    ax.set_xlabel("xyz"[i] + " (m)")
    ax.set_ylabel("xyz"[j] + " (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    if title:
        ax.set_title(title)
    return ax

"""JSON project files: fit component models and assemble them without writing code.

See ``docs/PROJECT_FILE.md`` for the schema and ``examples/project/`` for a
complete example.  Entry point: ``python -m emscatt {fit,assemble,run} project.json``.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .assembly import Assembly
from .conventions import Pose, dir_from_az_el, dir_from_theta_phi
from .data import ScatteringData
from .features import Part, feature_from_dict
from .fitting import FitConfig, SCFitter
from .metrics import compare, sector_statistics, to_dbsm
from .model import ScatteringCentreModel


def load_project(path) -> dict:
    with open(path) as fh:
        proj = json.load(fh)
    proj["_root"] = os.path.dirname(os.path.abspath(path))
    proj.setdefault("output", "output")
    return proj


def _path(proj, p):
    return p if os.path.isabs(p) else os.path.join(proj["_root"], p)


def _out(proj, *names):
    d = _path(proj, proj["output"])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *names)


def load_data_spec(proj, spec) -> ScatteringData:
    """One data block: {"file": ..., "format": "csv"|"npz", loader options...,
    "phase_reference": [x,y,z], "weight": 1.0, "noise_power": optional}."""
    spec = dict(spec)
    f = _path(proj, spec.pop("file"))
    fmt = spec.pop("format", "npz" if f.endswith(".npz") else "csv")
    pref = spec.pop("phase_reference", None)
    weight = spec.pop("weight", None)
    noise = spec.pop("noise_power", None)
    if fmt == "npz":
        d = ScatteringData.load_npz(f)
        if spec.pop("time_convention", "+j") == "-i":
            d = d.conjugate()
    else:
        if "angle_cols" in spec:
            spec["angle_cols"] = tuple(spec["angle_cols"])
        d = ScatteringData.load_csv(f, **spec)
    if pref is not None:
        d = d.with_phase_reference(pref)
    if weight is not None:
        d = d.copy(weights=d.w * float(weight))
    if noise is not None:
        d.meta["noise_power"] = float(noise)
    return d


def parts_from_spec(specs) -> list:
    out = []
    for p in specs or []:
        feats = [feature_from_dict(dict(f)) for f in p.get("features", [])]
        reg = p.get("region")
        out.append(Part(p["name"], feats, None if reg is None else tuple(np.asarray(r, float) for r in reg)))
    return out


def pose_from_spec(spec) -> Pose:
    if spec is None:
        return Pose()
    if "rotation" in spec:
        pose = Pose(np.asarray(spec["rotation"]), spec.get("translation", [0, 0, 0]))
    else:
        ypr = spec.get("yaw_pitch_roll_deg", [0, 0, 0])
        pose = Pose.from_yaw_pitch_roll(*ypr, translation=spec.get("translation", [0, 0, 0]))
    if "mirror_normal" in spec:   # mirror in the component frame first, then rotate/translate
        pose = pose.compose(Pose.mirror(spec["mirror_normal"]))
    return pose


def _fit_config(proj, overrides=None) -> FitConfig:
    kw = dict(proj.get("fit", {}))
    kw.update(overrides or {})
    for k in ("alphas", "channels"):
        if k in kw and kw[k] is not None:
            kw[k] = tuple(kw[k])
    if "grid_bounds" in kw and kw["grid_bounds"] is not None:
        kw["grid_bounds"] = tuple(np.asarray(b, float) for b in kw["grid_bounds"])
    return FitConfig(**kw)


def eval_grid(spec):
    """{"freqs_hz": [start, stop, n] | list, "az_deg": [start, stop, step], "el_deg": 0}
    or {"theta_deg": ..., "phi_deg": [start, stop, step]}."""
    fr = spec["freqs_hz"]
    freqs = np.linspace(*fr[:2], int(fr[2])) if len(fr) == 3 and spec.get("freqs_is_range", True) else np.array(fr)
    if "az_deg" in spec:
        a = spec["az_deg"]
        ang = np.arange(a[0], a[1], a[2])
        dirs = dir_from_az_el(ang, spec.get("el_deg", 0.0))
    else:
        p = spec["phi_deg"]
        ang = np.arange(p[0], p[1], p[2])
        dirs = dir_from_theta_phi(spec.get("theta_deg", 90.0), ang)
    return freqs, dirs, ang


# --------------------------------------------------------------------------- commands
def fit_components(proj, only=None, verbose=False) -> dict:
    models = {}
    for name, comp in proj["components"].items():
        if only and name not in only:
            continue
        blocks = [load_data_spec(proj, s) for s in comp["data"]]
        parts = parts_from_spec(comp.get("parts"))
        cfg = _fit_config(proj, comp.get("fit"))
        cfg.verbose = verbose or cfg.verbose
        res = SCFitter(blocks, parts, cfg).fit()
        res.model.name = name
        res.model.save(_out(proj, f"model_{name}.json"))
        with open(_out(proj, f"model_{name}.txt"), "w") as fh:
            fh.write(res.report() + "\n\n" + res.model.summary(blocks[0].freqs, blocks[0].dirs) + "\n")
        print(f"[fit] {name}: {len(res.model.centres)} centres, residual {res.residual_db:.1f} dB")
        _plot_fit(proj, name, blocks[0], res.model)
        models[name] = res.model
    return models


def build_assembly(proj, models=None) -> Assembly:
    asm_spec = proj["assembly"]
    asm = Assembly(name=proj.get("name", "assembly"))
    for inst in asm_spec["instances"]:
        cname = inst.get("component", inst["name"])
        m = (models or {}).get(cname)
        if m is None:
            m = ScatteringCentreModel.load(_out(proj, f"model_{cname}.json"))
        asm.add(inst["name"], m, pose_from_spec(inst.get("pose")))
    return asm


def assemble(proj, models=None, verbose=False) -> Assembly:
    asm = build_assembly(proj, models)
    spec = proj["assembly"]
    print(asm.summary())
    full = [load_data_spec(proj, s) for s in spec.get("full_data", [])]
    if full and spec.get("calibrate_interactions", True):
        cfg = _fit_config(proj, spec.get("interaction_fit"))
        cfg.verbose = verbose or cfg.verbose
        res = asm.calibrate_interactions(full, None, cfg)
        res.model.save(_out(proj, "model_interaction.json"))
        print(f"[assemble] interaction model: {len(res.model.centres)} centres")
    elif os.path.exists(_out(proj, "model_interaction.json")) and spec.get("use_saved_interaction", True):
        asm.interaction = ScatteringCentreModel.load(_out(proj, "model_interaction.json"))
    freqs, dirs, ang = eval_grid(spec["evaluate"])
    cov = asm.coverage(dirs, spec.get("coverage_tol_deg", 3.0), freqs)
    for n, mask in cov.items():
        if mask.any():
            warnings.warn(f"component '{n}': {100 * mask.mean():.0f}% of evaluation aspects outside measured coverage")
    chans = tuple(spec["evaluate"].get("channels", ["tt"]))
    out = asm.evaluate(freqs, dirs, chans)
    per = asm.evaluate(freqs, dirs, chans, per_component=True)
    import csv
    with open(_out(proj, "assembly_prediction.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_hz", "angle_deg"] + [f"{c}_{k}" for c in chans for k in ("re", "im", "dbsm")])
        for i, f in enumerate(freqs):
            for j, a in enumerate(ang):
                w.writerow([f, a] + [x for c in chans for x in (out[c][i, j].real, out[c][i, j].imag,
                                                                  to_dbsm(out[c][i, j]))])
    _plot_assembly(proj, ang, out[chans[0]], {k: v[chans[0]] for k, v in per.items()},
                   full[0].S.get(chans[0]) if full and full[0].na == len(ang) and full[0].nf == len(freqs) else None)
    if full and full[0].nf == len(freqs) and full[0].na == len(ang):
        print("[assemble] vs full data:", {k: round(v, 2) for k, v in
                                           compare(full[0].S[chans[0]], out[chans[0]], ang).items()})
    return asm


def _plot_fit(proj, name, data, model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import plot_rcs_cut, plot_sc_map
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    plot_sc_map(model, ax=ax[0], dirs=data.dirs, title=f"{name}: scattering centres")
    pred = model.evaluate(data.freqs, data.dirs)
    ch = model.channels[0]
    ang = np.rad2deg(np.unwrap(np.arctan2(data.dirs[:, 1], data.dirs[:, 0])))
    plot_rcs_cut(ang, {"data": data.S[ch], "model": pred[ch]}, ax=ax[1], title=f"{name}: data vs model ({ch})")
    fig.tight_layout()
    fig.savefig(_out(proj, f"fit_{name}.png"), dpi=120)
    plt.close(fig)


def _plot_assembly(proj, ang, total, per, full):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import plot_rcs_cut, plot_sector_means
    except ImportError:
        return
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    curves = {"assembly": total}
    if full is not None:
        curves = {"full data": full, **curves}
    plot_rcs_cut(ang, curves, ax=ax[0], title="Assembled RCS (centre frequency)")
    plot_sector_means(ang, {**curves, **per}, ax=ax[1])
    fig.tight_layout()
    fig.savefig(_out(proj, "assembly.png"), dpi=120)
    plt.close(fig)

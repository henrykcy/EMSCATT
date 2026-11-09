"""End-to-end validation of the EMSCATT workflow on a generic aircraft.

Synthetic "truth" data come from an independent physical-optics solver
(faceted meshes) plus a synthetic installation/interaction term, so the
scattering-centre fitter is *not* tested against its own kernels.

Steps
  A. Whole-target extraction: part-labelled SC model from full-aircraft data.
  B. Component route: fit each component in its own measurement frame,
     assemble coherently (poses incl. a mirrored wing), calibrate the
     interaction terms against one full-target dataset.
  C. Design iteration: re-measure only a redesigned wing (35 -> 45 deg sweep),
     swap it into the assembly and predict the full-aircraft RCS; compare with
     the true full-aircraft RCS of the modified design.
  D. Guard rails: coverage check when a component pose needs aspects that were
     never measured (canted fin from azimuth-cut data).

Run:  python examples/aircraft_workflow.py   (about 3-6 minutes; data cached)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generic_aircraft as ga
from emscatt import (Assembly, FitConfig, Pose, ScatteringData, SCFitter, backproject, dir_from_az_el, metrics,
                     plane_grid, po_monostatic)
from emscatt.plotting import plot_rcs_cut, plot_sc_map, plot_sector_means

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
FREQS = np.linspace(2.8e9, 3.2e9, 32)          # S-band, 400 MHz -> 0.375 m range res., 11.6 m unambiguous
AZ = np.arange(0, 360, 0.5)
DIRS = dir_from_az_el(AZ, 0.0)                 # horizontal azimuth cut (VV == 'tt')
SNR_DB = 45.0
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def synth(name, mesh, extra=None, seed=0):
    """PO 'measurement' of a mesh (cached), optional extra field, plus noise."""
    path = os.path.join(OUT, f"data_{name}.npz")
    if os.path.exists(path):
        return ScatteringData.load_npz(path)
    t = time.time()
    S = po_monostatic(mesh, FREQS, DIRS)
    if extra is not None:
        S = S + extra
    d = ScatteringData(FREQS, DIRS, {"VV": S}, name=name).add_noise(SNR_DB, seed)
    d.save_npz(path)
    log(f"   synthesised {name:<14} ({len(mesh.faces)} facets) in {time.time() - t:.1f}s")
    return d


def cfg(**kw):
    base = dict(basis_spacing_deg=15.0, physical_basis_spacing_deg=30.0, max_centres=45,
                target_residual_db=-30.0, min_gain_db=-45.0, grid_margin=0.3, alphas=(0.0, 0.5, 1.0),
                verbose=False)
    base.update(kw)
    return FitConfig(**base)


def fit(data, parts, label, **kw):
    t = time.time()
    r = SCFitter(data, parts, cfg(**kw)).fit()
    r.model.name = label
    m = r.model.meta
    log(f"   {label:<16}: {len(r.model.centres):3d} centres, residual {r.residual_db:6.1f} dB aspect-balanced "
        f"/ {m['residual_db_unweighted']:6.1f} dB unweighted, noise floor {m.get('noise_db', np.nan):6.1f} dB "
        f"({time.time() - t:.0f}s)")
    return r


def main():
    t0 = time.time()
    comps = ga.components()
    log("== Synthesising data (independent PO solver + interaction term) ==")
    data_comp = {
        "fuselage": synth("fuselage", comps["fuselage"][0], seed=1),
        "wing": synth("wing35", comps["wing_R"][0], seed=2),
        "fin": synth("fin", comps["fin"][0], seed=3),
        "intake": synth("intake", comps["intake"][0], seed=4),
    }
    inter = ga.interaction_truth(FREQS, DIRS)
    full = synth("full35", ga.global_mesh(comps), inter, seed=5)
    comps45 = ga.components({"sweep_deg": 45.0})
    wing45 = synth("wing45", comps45["wing_R"][0], seed=6)
    full45 = synth("full45", ga.global_mesh(comps45), inter, seed=7)
    S_full = full.S["tt"]

    # ------------------------------------------------------------------ A
    log("\n== A. Whole-target scattering-centre extraction (part-labelled) ==")
    gparts = ga.global_parts(comps)
    rA = fit(full, gparts, "full-target", max_centres=60)
    log(rA.report())
    mA = rA.model
    mA.save(os.path.join(OUT, "model_full_target.json"))
    predA = mA.evaluate(FREQS, DIRS)["tt"]
    log("   accuracy vs data:", fmt(metrics.compare(S_full, predA, AZ)))
    with open(os.path.join(OUT, "model_full_target.txt"), "w") as fh:
        fh.write(mA.summary(FREQS, DIRS))

    X, u, v, _ = plane_grid((-3.0, 2.6, -2.6, 2.6), 0.02)
    img = np.abs(backproject(full, X)).reshape(u.size, v.size)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    plot_sc_map(mA, ax=ax[0], image=img, image_extent=(u[0], u[-1], v[0], v[-1]), outline=ga.outline(comps),
                title="A. Full-target SC model over back-projection image")
    contrib = mA.part_contributions(FREQS, DIRS)
    order = sorted(contrib, key=lambda p: -np.sum(np.abs(contrib[p]) ** 2))[:6]
    plot_sector_means(AZ, {"data": S_full, "model": predA, **{p: contrib[p] for p in order}}, ax=ax[1],
                      title="Sector-mean RCS and the largest part contributions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "A_full_target_model.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ B
    log("\n== B. Component models (each fitted in its own measurement frame) ==")
    models = {
        "fuselage": fit(data_comp["fuselage"], comps["fuselage"][1], "fuselage").model,
        "wing": fit(data_comp["wing"], comps["wing_R"][1], "wing35").model,
        "fin": fit(data_comp["fin"], comps["fin"][1], "fin").model,
        "intake": fit(data_comp["intake"], comps["intake"][1], "intake").model,
    }
    for n, m in models.items():
        m.save(os.path.join(OUT, f"model_component_{n}.json"))
    P = ga.poses()
    asm = Assembly(name="airframe")
    asm.add("fuselage", models["fuselage"], P["fuselage"])
    asm.add("wing_R", models["wing"], P["wing_R"])
    asm.add("wing_L", models["wing"], P["wing_L"])          # mirror image of the same measured wing
    asm.add("fin", models["fin"], P["fin"])
    asm.add("intake", models["intake"], P["intake"])
    log(asm.summary())
    pred_noint = asm.evaluate(FREQS, DIRS)["tt"]
    log("   assembly (no interaction)  vs full data:", fmt(metrics.compare(S_full, pred_noint, AZ)))
    log("   incoherent power sum       vs full data:",
        fmt(metrics.compare(S_full, asm.evaluate(FREQS, DIRS, incoherent=True)["tt"], AZ)))
    rI = asm.calibrate_interactions(full, None, cfg(max_centres=20, alphas=(0.0,), basis_spacing_deg=10.0,
                                                    grid_bounds=([-2.6, -2.3, 0], [2.3, 2.3, 0])))
    log(f"   interaction model: {len(rI.model.centres)} centres "
        "(true synthetic terms: two junction scatterers at (-0.50, +/-0.25))")
    for sc in rI.model.centres:
        p = sc.feature.reference_point
        log(f"      interaction centre at ({p[0]:6.2f},{p[1]:6.2f}) peak "
            f"{sc.peak_rcs_dbsm(FREQS, DIRS, rI.model.f_ref):6.1f} dBsm")
    pred_int = asm.evaluate(FREQS, DIRS)["tt"]
    log("   assembly + interaction     vs full data:", fmt(metrics.compare(S_full, pred_int, AZ)))
    S_true35 = po_monostatic(ga.global_mesh(comps), FREQS, DIRS) + inter
    log("   assembly + interaction     vs noise-free truth:", fmt(metrics.compare(S_true35, pred_int, AZ)))

    fig, ax = plt.subplots(3, 1, figsize=(12, 11))
    plot_rcs_cut(AZ, {"full-aircraft data": S_full, "assembled components": pred_noint}, ax=ax[0],
                 title="B. Coherent assembly of component models (centre frequency)", ylim=(-40, 35))
    plot_rcs_cut(AZ, {"full-aircraft data": S_full, "assembly + interaction": pred_int}, ax=ax[1],
                 ylim=(-40, 35))
    plot_sector_means(AZ, {"full-aircraft data": S_full, "assembly": pred_noint,
                           "assembly + interaction": pred_int,
                           "incoherent power sum (no interaction)":
                               asm.evaluate(FREQS, DIRS, incoherent=True, include_interaction=False)["tt"]}, ax=ax[2])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "B_component_assembly.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ C
    log("\n== C. Design iteration: re-measure only the wing (sweep 35 -> 45 deg) ==")
    m45 = fit(wing45, comps45["wing_R"][1], "wing45").model
    asm45 = asm.replace("wing_R", model=m45).replace("wing_L", model=m45)
    asm45 = asm45.replace("wing_L", model=m45)
    pred45 = asm45.evaluate(FREQS, DIRS)["tt"]
    S_true45 = po_monostatic(ga.global_mesh(comps45), FREQS, DIRS) + inter
    log("   predicted new aircraft vs true new aircraft:", fmt(metrics.compare(S_true45, pred45, AZ)))
    st = {k: metrics.sector_statistics(AZ, v, 10.0)["mean_dbsm"]
          for k, v in {"t35": S_true35, "t45": S_true45, "p35": pred_int, "p45": pred45}.items()}
    true_delta, pred_delta = st["t45"] - st["t35"], st["p45"] - st["p35"]
    log(f"   10-deg sector-mean RCS change: true range [{true_delta.min():.1f}, {true_delta.max():.1f}] dB; "
        f"prediction error rms {np.sqrt(np.mean((pred_delta - true_delta) ** 2)):.2f} dB, "
        f"max {np.max(np.abs(pred_delta - true_delta)):.2f} dB")
    centres = metrics.sector_statistics(AZ, S_true35, 10.0)["centre_deg"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_rcs_cut(AZ, {"true (45 deg wing)": S_true45, "predicted from component swap": pred45}, ax=ax[0],
                 title="C. Design iteration: full-aircraft RCS predicted from a re-measured wing only",
                 ylim=(-40, 35))
    ax[1].step(centres, true_delta, where="mid", label="true change")
    ax[1].step(centres, pred_delta, where="mid", label="predicted change", ls="--")
    ax[1].set_xlabel("aspect angle (deg)")
    ax[1].set_ylabel("change of 10-deg sector-mean RCS (dB)")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "C_design_iteration.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ D
    log("\n== D. Guard rails ==")
    canted = asm.replace("fin", pose=Pose.from_yaw_pitch_roll(roll=20, translation=[-1.0, 0, 0.15]))
    cov = canted.coverage(DIRS, tol_deg=2.0)
    for n, mask in cov.items():
        if mask.any():
            log(f"   component '{n}': {mask.mean() * 100:.0f}% of requested aspects lie outside its "
                "measured coverage -> prediction would be extrapolated (re-measure in 3-D)")
    log(f"\nTotal run time {time.time() - t0:.0f}s.  Figures and models written to {OUT}")
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


def fmt(d):
    return ", ".join(f"{k}={v:.2f}" for k, v in d.items())


if __name__ == "__main__":
    main()

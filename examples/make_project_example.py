"""Write a small CLI example project (CSV data + project.json) to examples/project/.

The data are synthetic PO 'measurements' of the generic aircraft components,
exported in the long CSV format (freq in GHz, az/el in degrees, RCS in dBsm
and phase in degrees) that many measurement ranges and solvers can produce.
Then run:   python -m emscatt run examples/project/project.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import generic_aircraft as ga
from emscatt import ScatteringData, dir_from_az_el, po_monostatic
from emscatt.metrics import to_dbsm

OUT = os.path.join(os.path.dirname(__file__), "project")
os.makedirs(OUT, exist_ok=True)
FREQS = np.linspace(2.8e9, 3.2e9, 24)
AZ = np.arange(0, 360, 0.5)
DIRS = dir_from_az_el(AZ, 0)


def write_csv(path, S):
    with open(path, "w") as fh:
        fh.write("freq_ghz,az,el,vv_db,vv_deg\n")
        for i, f in enumerate(FREQS):
            for j, a in enumerate(AZ):
                fh.write(f"{f / 1e9:.6f},{a:.3f},0,{to_dbsm(S[i, j], -300):.4f},{np.rad2deg(np.angle(S[i, j])):.4f}\n")


def feat(f):
    d = f.to_dict()
    return {k: v for k, v in d.items() if k not in ("part",)}


comps = ga.components()
spec_components = {}
rng = np.random.default_rng(0)
for cname, key in (("fuselage", "fuselage"), ("wing", "wing_R"), ("fin", "fin"), ("intake", "intake")):
    mesh, parts, _ = comps[key]
    d = ScatteringData(FREQS, DIRS, {"VV": po_monostatic(mesh, FREQS, DIRS)}).add_noise(45, rng)
    write_csv(os.path.join(OUT, f"{cname}.csv"), d.S["tt"])
    spec_components[cname] = {
        "data": [{"file": f"{cname}.csv", "format": "csv", "freq_col": "freq_ghz", "freq_scale": 1e9,
                  "angle_cols": ["az", "el"], "angle_type": "az_el", "channels": {"VV": "vv"},
                  "value_format": "db_deg"}],
        "parts": [{"name": p.name, "features": [feat(f) for f in p.features],
                   **({"region": [r.tolist() for r in p.region[:2]]} if p.region is not None else {})}
                  for p in parts],
    }
full = ScatteringData(FREQS, DIRS, {"VV": po_monostatic(ga.global_mesh(comps), FREQS, DIRS)
                                    + ga.interaction_truth(FREQS, DIRS)}).add_noise(45, rng)
write_csv(os.path.join(OUT, "full_aircraft.csv"), full.S["tt"])
P = ga.poses()
proj = {
    "name": "generic airframe",
    "output": "output",
    "fit": {"basis_spacing_deg": 15, "physical_basis_spacing_deg": 30, "max_centres": 45,
            "target_residual_db": -30, "alphas": [0, 0.5, 1]},
    "components": spec_components,
    "assembly": {
        "instances": [
            {"name": "fuselage"},
            {"name": "wing_R", "component": "wing", "pose": {"translation": P["wing_R"].translation.tolist()}},
            {"name": "wing_L", "component": "wing",
             "pose": {"translation": P["wing_L"].translation.tolist(), "mirror_normal": [0, 1, 0]}},
            {"name": "fin", "pose": {"translation": P["fin"].translation.tolist()}},
            {"name": "intake", "pose": {"translation": P["intake"].translation.tolist()}},
        ],
        "full_data": [{"file": "full_aircraft.csv", "format": "csv", "freq_col": "freq_ghz", "freq_scale": 1e9,
                       "angle_cols": ["az", "el"], "angle_type": "az_el", "channels": {"VV": "vv"},
                       "value_format": "db_deg"}],
        "interaction_fit": {"max_centres": 20, "alphas": [0], "basis_spacing_deg": 10,
                            "grid_bounds": [[-2.6, -2.3, 0], [2.3, 2.3, 0]]},
        "evaluate": {"freqs_hz": [2.8e9, 3.2e9, 24], "az_deg": [0, 360, 0.5], "el_deg": 0, "channels": ["tt"]},
    },
}
with open(os.path.join(OUT, "project.json"), "w") as fh:
    json.dump(proj, fh, indent=2)
print("wrote", OUT)

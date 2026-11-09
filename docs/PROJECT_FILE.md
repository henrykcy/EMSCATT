# Project file (`python -m emscatt run project.json`)

Use a project file to run the whole workflow without writing Python. To generate a complete example, run `python examples/make_project_example.py`, which writes `examples/project/`.

```jsonc
{
  "name": "my airframe",
  "output": "output",                       // relative to the project file
  "fit": {                                  // defaults for every component (FitConfig fields)
    "basis_spacing_deg": 15, "physical_basis_spacing_deg": 30,
    "max_centres": 45, "target_residual_db": -30, "alphas": [0, 0.5, 1]
  },
  "components": {
    "wing": {
      "data": [                             // one or more blocks (measured and/or simulated)
        {"file": "wing.csv", "format": "csv",
         "freq_col": "freq_ghz", "freq_scale": 1e9,
         "angle_cols": ["az", "el"], "angle_type": "az_el",       // or ["theta","phi"], "theta_phi"
         "channels": {"VV": "vv"},                                // columns vv_db / vv_deg ...
         "value_format": "db_deg",                                // or "re_im", "lin_deg"
         "time_convention": "+j",                                 // "-i" -> conjugated on load
         "phase_reference": [0.0, 0.0, 0.0],                      // turntable centre in component frame
         "weight": 1.0,                                           // relative weight of this block
         "noise_power": null}                                     // optional; otherwise estimated
      ],
      "parts": [
        {"name": "leading_edge", "features": [
            {"kind": "cylinder", "p1": [0,0,0], "p2": [-1.26,-1.8,0], "radius": 0.025, "name": "LE_round"},
            {"kind": "edge",     "p1": [0,0,0], "p2": [-1.26,-1.8,0], "name": "LE_sharp"}]},
        {"name": "root", "region": [[-0.55,0,0], [0.6,0.1,0.1]]}  // centre, half-sizes
      ],
      "fit": {"max_centres": 30}            // per-component overrides
    }
  },
  "assembly": {
    "instances": [
      {"name": "wing_R", "component": "wing", "pose": {"translation": [0.55,-0.18,0]}},
      {"name": "wing_L", "component": "wing",
       "pose": {"translation": [0.55,0.18,0], "mirror_normal": [0,1,0]}},
      {"name": "fin", "pose": {"yaw_pitch_roll_deg": [0,0,0], "translation": [-1,0,0.15]}}
    ],
    "full_data": [ { ...same block format... } ],        // optional: calibrates interactions
    "interaction_fit": {"max_centres": 20, "basis_spacing_deg": 10},
    "evaluate": {"freqs_hz": [2.8e9, 3.2e9, 24], "az_deg": [0, 360, 0.5], "el_deg": 0,
                 "channels": ["tt"]}
  }
}
```

**Feature kinds and their parameters:**

| `kind` | Parameters |
|---|---|
| `point` | `position` |
| `edge` | `p1`, `p2` (or `centre`, `axis`, `length`) |
| `cylinder` | `p1`, `p2`, `radius` |
| `ellipsoid` | `centre`, `semi_axes`, optional `rotation` (3×3) |
| `plate` | `vertices` (convex, planar), optional `normal`, `two_sided` |

Every kind also accepts an optional `alpha` and `name`.

**Pose:** one of the following, applied after an optional `mirror_normal`:

* `{"rotation": 3x3, "translation": [...]}`
* `{"yaw_pitch_roll_deg": [...], "translation": [...]}`

**Commands:**

* `fit`: fits every component (or the subset given with `--only`). It writes these files to the output directory:
  * `model_<name>.json` and `model_<name>.txt`
  * `fit_<name>.png`
* `assemble`: builds the assembly and calibrates interactions if `full_data` is given. It writes:
  * `assembly_prediction.csv` and `assembly.png`
  * `model_interaction.json`
* `run`: runs `fit`, then `assemble`.

# EMSCATT methodology

This note covers the physics, the algorithms, and the assumptions behind each step, so you can judge when the component → system prediction is trustworthy.

References that motivate the design:

* K.-Y. Guo, Q.-F. Li, X.-Q. Sheng, M. Gashinova, *Sliding scattering center model for extended streamlined targets*, PIER 139, 2013. This paper motivates the sliding centres, whose position depends on aspect, and the flexible amplitude functions.
* J. Lu, Y. Zhang, H. Yan, L. Zhang, H. Yin, *Global scattering center representation of target wide-angle single reflection/diffraction mechanisms based on the multiple manifold concept*, Electronics 11, 4209, 2022. This paper motivates the taxonomy of fixed, sliding and distributed centres, and the mapping from geometry to centre type.
* Gerry, Potter et al. describe the attributed scattering centre (ASC) model, which gives the sinc length term and the α frequency exponents. Li & Stoica describe RELAX.

---

## 1. Model

Conventions: `exp(+jωt)` time dependence; r̂ points from the target to the radar (monostatic); σ = |S|².

```
S_ch(f, r̂) = Σ_m  K_m(f, r̂; geometry_m) · a_m,ch(r̂)
a_m,ch(r̂)  = Σ_j c_m,j,ch · B_m,j(r̂)          (smooth angular amplitude)
```

* **Kernel K_m.** The kernel holds everything that geometry fixes:
  - phase centre, including **sliding** of specular points;
  - coherent length (sinc lobes of distributed centres);
  - the GO/PO magnitude;
  - the GTD frequency exponent (f/f_ref)^α.
* **Amplitude a_m(r̂).** The amplitude holds what geometry does not fix: material or RAM reflectivity, diffraction-coefficient angle dependence, visibility and shadowing, and polarisation.
  - It is expanded in normalised Gaussian RBFs: anchors every `basis_spacing_deg` over the arc or sphere covered by the data.
  - The expansion is a partition of unity, so a constant pattern is represented exactly.
  - It generalises both the ASC `exp(-2πfγ sin φ)` term and the rational amplitude function of Guo et al., without fixing a functional form.
* **Channels.** All polarisation channels share the geometry; only the amplitudes are per-channel. This is a multiple-measurement-vector sparse problem.

### Feature (kernel) library

| Feature | Mechanism | GSC type (Lu 2022) | Kernel | Fitted amplitude means |
|---|---|---|---|---|
| `PointFeature` | tip, corner, cavity, junction, unknown | FF fixed | `e^{j2k r̂·x}(f/f_ref)^α` | √σ in m |
| `EdgeFeature` | straight sharp edge (diffraction) | FD fixed-distributed | `sinc(kL r̂·t̂)e^{j2k r̂·x_c}(f/f_ref)^α` | √σ at broadside, m |
| `CylinderFeature` | rounded edge, tube, nacelle | SD sliding-distributed | PO: `−j e^{−jπ/4} L √(kρ cosβ) sinc(kL r̂·t̂) e^{j2k r̂·(c+ρn̂⊥)}` | reflectivity (1 = PEC) |
| `EllipsoidFeature` | doubly curved surface (fuselage, radome, sphere) | SS sliding | GO: `−√π abc/(uᵀD²u) e^{j2k r̂·x_s(r̂)}` | reflectivity (1 = PEC) |
| `PlateFeature` | flat polygonal panel | DD distributed | exact PO integral of the polygon | reflectivity (1 = PEC) |

The physical kernels were checked against the independent faceted-PO solver:

* ellipsoid: ≈1 % error;
* cylinder: ≈4–7 % (faceting);
* plate: exact.

**Why sliding features matter.** The demo makes this concrete. The rounded wing leading edge (radius 2.5 cm) reflects from a specular line offset ρ·n̂⊥ from the axis, and that offset changes side with aspect. The result is an aspect-dependent range shift: about 0.4 rad of phase across a 400 MHz band. No fixed-position edge can absorb that with an angle-only amplitude.

* Edge-only fit of that leading edge: −19 dB.
* `CylinderFeature` fit: −34 dB.
* Effect on the whole wing: the model shrinks from 41 centres at −26 dB to **4 centres at the noise floor**.

## 2. Extraction algorithm (`SCFitter`)

1. **Dictionary.** Candidates come from two sources:
   - the features of each named `Part`, which you supply from CAD, each tried with every `alpha` in `alphas` (for point and edge features);
   - optionally, a free grid of point candidates covering the parts.

   In a planar cut the grid is 2-D in the data plane, because the out-of-plane coordinate is unobservable.
2. **Greedy group selection.** This is group-OMP or MMV-OMP. Each step adds the candidate with the largest windowed matched-filter energy against the residual:

   ```
   score = Σ_ch Σ_j |Σ_r̂ B_j(r̂) Σ_f w K* R|² / Σ_r̂ B_j(r̂) Σ_f w |K|²
   ```

   This is the sub-aperture ISAR image energy per anchor. For point candidates on uniform frequency grids it is computed by FFT range-profile back-projection, so thousands of candidates are cheap. Part features get a `part_bias` preference so that physically named mechanisms win ties.
3. **Joint regularised least squares.** All amplitudes are re-solved with the normal equations, using a smoothness penalty on neighbouring anchors plus a small ridge term.
4. **RELAX refinement.** Each free point is first re-positioned immediately with Nelder–Mead on its own matched filter, which avoids grid-quantisation duplicates. All refinable centres are then cycled `refine_passes` times. Coincident centres are merged.
5. **Pruning and attribution.**
   - Centres below `prune_db` of the energy are dropped.
   - Free centres are attributed to the part whose explicit `region` contains them; otherwise to the nearest part within `assign_radius`; otherwise they are labelled `unassigned`.
   - `model.meta["energy_share_db"]` gives each part's share of the energy.

### Weighting and stopping (important for RCS work)

* **Aspect balancing** (`balance_aspects=True`). RCS requirements are set per aspect sector in dB. With unweighted least squares, a 27 dBsm fin flash would make every −30 dBsm aspect irrelevant. The fit therefore re-weights each aspect by 1/(sector-smoothed power + floor), with the floor at `balance_dynamic_range_db` below the strongest sector, so relative accuracy is roughly uniform over aspect.
  - When fitting a residual (interaction calibration), the weights come from the **full-target** data, not from the residual.
* **Noise-aware stopping.**
  - The noise variance is estimated automatically from range gates outside the object, after a Hann window, with a median estimator. It is accurate to within a few % in the demo. You can also supply it as `noise_power` or `meta["noise_power"]`.
  - A new centre is accepted only if it removes more than `noise_gain_factor × (number of coefficients) × σ²` of energy, which is more than fitting pure noise would remove.
  - This stops the fitter from building dozens of −45 dBsm "centres" out of noise in quiet sectors.

## 3. Coherent assembly (`Assembly`)

Each component is fitted in its own measurement frame. A `Pose` (R, t) places it in the global frame. R may be improper: a mirror image is how one measured wing becomes both wings. Maxwell's equations are reflection-invariant, so this is exact for a geometrically mirrored part.

For each global direction r̂_g:

```
r̂_l = Rᵀ r̂_g
S_g = T(r̂_g) · S_l(f, r̂_l) · T(r̂_g)ᵀ · exp(j2k r̂_g·t)
T_ij = ê_g,i · (R ê_l,j)        (θ̂/φ̂ basis rotation, backscatter alignment)
```

* For yaw plus translation in an azimuth cut, T = I and single-polarisation models are enough.
* For roll or pitch the channels mix. Supply full-pol (tt, tp, pp) component data, or the code warns.

The total is the coherent sum of the components plus an **interaction model**.

### Interaction (installation) terms

A sum of isolated components misses several effects:

* multiple bounces between components (wing–fuselage dihedral, fin–stabiliser corner);
* blockage and shadowing;
* creeping and travelling waves that cross joints;
* mounting artefacts.

`Assembly.calibrate_interactions(full_data)` fits the residual

```
full measurement − coherent component sum
```

with a global-frame scattering-centre model, and keeps it fixed while components are iterated.

* This is the key engineering assumption: an interaction depends weakly on the detailed design of one component.
* It holds well when the redesign does not change the geometry at the junction, for example a coating change, edge treatment, or planform change away from the root.
* It must be **re-validated** when the junction itself changes. Re-measure the full target, or simulate the junction region, and recalibrate.

The model also gives a useful diagnostic. Interaction centres cluster where the physics happens; in the demo that is the wing roots. That clustering tells you where to look.

### Coherent vs incoherent

`evaluate(..., incoherent=True)` returns √Σ|S_i|². This is all you can do with RCS-magnitude-only component data.

* It recovers sector-*average* levels reasonably well.
* It cannot predict the lobing structure, nulls, peak values, glint, ISAR images, or coherent cancellation between components.

The coherent route needs phase-calibrated data with a known phase reference.

### Guard rails

* `Assembly.coverage(dirs)` flags global aspects whose local aspect was never measured for some component, for example a canted fin modelled from azimuth-cut data. It also flags frequencies outside the fitted band, where α extrapolation applies.
* Use `sampling_check(data, max_radius)` to verify Nyquist sampling:
  - angular step < λ/(4R);
  - unambiguous range c/(2Δf) > 2R.

## 4. Measurement and simulation requirements

To be usable for coherent component-to-system prediction, data must satisfy all of the following:

1. **Complex and calibrated.** Amplitude and phase, calibrated to absolute RCS (for example with a sphere or dihedral), background-subtracted, with the same polarisation definition for every component.
2. **Known phase reference.** The phase of each component's data must refer to a known point in that component's frame, typically the turntable centre. Use `data.with_phase_reference(p)` to move it to the component origin. A position error δ produces a phase error of 2k·δ: 1 cm at 10 GHz is about 120°. This is the single most important practical requirement.
3. **Known mounting pose.** You need the pose of each component in the global frame, and you need to know which side was measured. One-sided parts, such as the inner face of an intake, must be measured with the same visibility they have when installed.
4. **Coverage.** Each component's measured aspects must cover the local aspects it will see when installed (`coverage()`). An azimuth cut only supports poses that keep the component's measurement plane in the global evaluation plane.
5. **Bandwidth.** Bandwidth is needed for range resolution and to determine α. Single-frequency data work: the fitter then relies on angular diversity, which is sub-aperture cross-range resolution. α is not identifiable at a single frequency.
6. **Combining sources.** Pass several `ScatteringData` blocks to the fitter. They can have different frequency and angle grids, for example simulated full-sphere data plus a measured azimuth cut. Use `weights` (≈1/noise variance) to balance them. Check for systematic scale or phase offsets between the sources before combining.
7. **Isolation artefacts.** A component measured in isolation has artificial terminations: open roots, cut edges, mounting columns. Gate them out or model them as separate parts, so they are not double-counted. In the demo, the wing "root" part collects exactly these returns.

## 5. Typical workflow

1. **Baseline.** Measure or simulate the full target once, and each component on its own.
2. **Parts.** Define `Part`s from CAD: edges, rounded edges (cylinders), curved hulls (ellipsoids), panels (plates), regions for cavities and joints.
3. **Fit.** Fit each component model, checking residual vs noise floor and part energy shares.
4. **Assemble.** Assemble with poses and compare with the full target. The gap is the interaction level.
5. **Calibrate.** `calibrate_interactions(full)`. Inspect where the interaction centres sit.
6. **Iterate.** Change a component, re-measure only that component, fit, `assembly.replace(...)`, predict, and compare sector statistics.
7. **Re-baseline.** Re-check against a full-target measurement whenever a change touches a junction, or at programme milestones.

## 6. Validation results

These come from `examples/aircraft_workflow.py`. The truth data are a faceted PO solution plus a synthetic interaction term, with noise 45 dB below the mean signal power.

| Test | Result |
|---|---|
| Whole-target extraction (15 part-labelled centres) | field error −35.5 dB; 10° sector-mean error 0.26 dB rms |
| Components alone, assembled coherently | sector-mean error 4.4 dB rms (13.6 dB max), all at the junction-lobe aspects |
| Plus interaction calibration | field error −47 dB vs noise-free truth; sector-mean error 0.04 dB rms (0.09 dB max) |
| Wing redesign 35°→45° sweep; only the wing re-measured | full-aircraft sector-mean change predicted to 0.03 dB rms (0.16 dB max), over true changes of −23 to +18 dB |
| Canted fin from azimuth-cut data | flagged: 94 % of aspects outside coverage |

The synthetic test isolates the modelling and fitting error, plus noise. Real projects add calibration, alignment, mounting and interaction-stationarity errors. Section 4 is the checklist for keeping those under control.

## 7. Limitations and extensions

* **Multiple bounces inside a component**, such as cavities and dihedrals, are fitted as point or edge centres with angular amplitudes. These are valid within the measured aspects, but they do not extrapolate physically. A dedicated dihedral/trihedral feature could be added the same way.
* **Interaction stationarity** is an assumption. Verify it whenever a junction changes.
* **Near-field or compact-range data** must already be in far-field form.
* **3-D full-sphere fits** need larger grids. Use `max_grid_points`, a coarser `basis_spacing_deg`, or part features, which are far more efficient than free points.
* **Sliding features for arbitrary curved surfaces.** Ellipsoids and cylinders cover most hulls, nacelles and rounded edges. A general NURBS or mesh specular-point feature is a natural extension: the kernel interface is just `kernel(freqs, dirs, f_ref)`.

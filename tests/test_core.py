import numpy as np
import pytest

from emscatt import (AngularBasis, Assembly, CylinderFeature, EdgeFeature, EllipsoidFeature, FitConfig, Mesh, Part,
                     PlateFeature, PointFeature, Pose, ScatteringCentreModel, ScatteringData, SCFitter, C0,
                     dir_from_az_el, dir_from_theta_phi, fibonacci_sphere, po_monostatic, rotation_about_axis,
                     wavenumber, fit_scattering_centres)
from emscatt.conventions import polarisation_transform
from emscatt.po import tri_unit_integral


def brute_tri(a, b, n=1500):
    u = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, u, indexing="ij")
    m = U + V <= 1
    return np.sum(np.exp(1j * (a * U + b * V)) * m) / n ** 2


@pytest.mark.parametrize("a,b", [(0, 0), (1e-7, 4), (6, 6 + 1e-8), (12, -7), (0.3, -0.2), (-25, 1e-10), (0.6, 0)])
def test_triangle_integral(a, b):
    assert abs(tri_unit_integral(a, b) - brute_tri(a, b)) < 2e-3


def test_po_plate_and_sphere():
    f = 3e9
    lam = C0 / f
    m = Mesh.polygon([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]], normal=[0, 0, 1])
    S = po_monostatic(m, [f], [[0, 0, 1]])
    assert np.isclose(abs(S[0, 0]) ** 2, 4 * np.pi * 0.5 ** 2 / lam ** 2, rtol=1e-9)
    sp = Mesh.ellipsoid([0.5, 0.5, 0.5], n_u=120, n_v=60)
    S = po_monostatic(sp, [f], dir_from_az_el([0, 40], [0, 25]))
    assert np.allclose(10 * np.log10(abs(S) ** 2), 10 * np.log10(np.pi * 0.25), atol=0.1)


def test_physical_kernels_match_po():
    f = np.linspace(2.8e9, 3.2e9, 4)
    d = dir_from_az_el(np.arange(0, 360, 9.0), 12)
    R = rotation_about_axis([0, 0, 1], 25)
    e = EllipsoidFeature([0.1, 0.2, 0], [0.8, 0.3, 0.25], rotation=R)
    m = Mesh.ellipsoid([0.8, 0.3, 0.25], n_u=160, n_v=80).transformed(Pose(R, [0.1, 0.2, 0]))
    K, P = e.kernel(f, d, 3e9), po_monostatic(m, f, d)
    assert np.linalg.norm(K - P) / np.linalg.norm(P) < 0.03
    poly = [[0, 0, 0], [0.6, 0, 0], [0.5, 0.4, 0], [0, 0.3, 0]]
    pl = PlateFeature(poly, two_sided=True)
    assert np.allclose(pl.kernel(f, d, 3e9), po_monostatic(Mesh.polygon(poly, two_sided=True), f, d))
    c = CylinderFeature([0, -0.5, 0], [0.2, 0.5, 0.0], radius=0.06)
    mc = Mesh.cylinder([0, -0.5, 0], [0.2, 0.5, 0.0], 0.06, n_circ=128)
    d2 = dir_from_az_el(np.arange(0, 360, 1.0), 0)
    K, P = c.kernel(f, d2, 3e9), po_monostatic(mc, f, d2)
    assert np.linalg.norm(K - P) / np.linalg.norm(P) < 0.1


def test_feature_transform_covariance():
    pose = Pose.from_yaw_pitch_roll(30, 10, -20, translation=[0.3, -0.2, 0.5])
    f = np.array([3e9, 3.1e9])
    dg = fibonacci_sphere(50)
    dl = pose.to_local_dirs(dg)
    k = wavenumber(f)
    tr = np.exp(2j * k[:, None] * (dg @ pose.translation)[None, :])
    feats = [PointFeature([0.1, 0.2, 0.3]), EdgeFeature([0, 0, 0], [0.5, 0.1, 0]),
             EllipsoidFeature([0, 0, 0.1], [0.5, 0.2, 0.1]), CylinderFeature([0, 0, 0], [0.4, 0, 0], 0.05),
             PlateFeature([[0, 0, 0], [0.3, 0, 0], [0, 0.2, 0]], two_sided=True)]
    for p in (pose, Pose.mirror([0, 1, 0], [0.2, 0.3, 0])):
        dl = p.to_local_dirs(dg)
        tr = np.exp(2j * k[:, None] * (dg @ p.translation)[None, :])
        for ft in feats:
            a = ft.transformed(p).kernel(f, dg, 3e9)
            b = ft.kernel(f, dl, 3e9) * tr
            assert np.allclose(a, b, atol=1e-9), ft


def test_basis_partition_of_unity():
    d = dir_from_az_el(np.arange(0, 360, 1.0), 0)
    B = AngularBasis.for_directions(d, 15).evaluate(d)
    assert np.allclose(B.sum(1), 1)
    d3 = fibonacci_sphere(500)
    B3 = AngularBasis.for_directions(d3, 30)
    assert B3.size > 20 and np.allclose(B3.evaluate(d3).sum(1), 1)
    arc = dir_from_az_el(np.arange(-30, 31, 1.0), 0)
    Ba = AngularBasis.for_directions(arc, 15)
    assert 4 <= Ba.size <= 6


def _points_data(pts, amps, f, d, extra_channels=False):
    k = wavenumber(f)
    S = sum(a * np.exp(2j * k[:, None] * (d @ np.array(p))[None, :]) for a, p in zip(amps, pts))
    ch = {"VV": S}
    if extra_channels:
        ch["HH"] = 0.5 * S
    return ScatteringData(f, d, ch)


def test_fit_recovers_points_planar():
    pts = [[1.0, 0.2, 0], [-0.5, 0.7, 0], [0.3, -0.8, 0]]
    amps = [1.0, 0.5, 0.8]
    f = np.linspace(9e9, 10e9, 16)
    d = dir_from_az_el(np.arange(0, 360, 0.5), 0)
    data = _points_data(pts, amps, f, d, True).add_noise(40, 0)
    r = fit_scattering_centres(data, grid_bounds=([-1.5, -1.5, 0], [1.5, 1.5, 0]), max_centres=8)
    assert r.residual_db < -35
    pos = np.array([sc.feature.position for sc in r.model.centres])
    for p in pts:
        assert np.min(np.linalg.norm(pos - p, axis=1)) < 0.01
    assert set(r.model.channels) == {"tt", "pp"}


def test_fit_recovers_points_3d_single_frequency():
    pts = [[0.3, 0.1, -0.2], [-0.2, -0.25, 0.15]]
    amps = [1.0, 0.7j]
    d = fibonacci_sphere(1500)
    data = _points_data(pts, amps, [3e9], d)
    r = fit_scattering_centres(data, grid_bounds=([-0.5] * 3, [0.5] * 3), basis_spacing_deg=40,
                               max_centres=6, max_grid_points=6000)
    assert r.residual_db < -30
    pos = np.array([sc.feature.position for sc in r.model.centres])
    for p in pts:
        assert np.min(np.linalg.norm(pos - p, axis=1)) < 0.02


def test_part_features_are_preferred_and_labelled():
    f = np.linspace(2.8e9, 3.2e9, 11)
    d = dir_from_az_el(np.arange(0, 360, 0.5), 0)
    ell = EllipsoidFeature([0, 0, 0], [1.0, 0.2, 0.2])
    edge = EdgeFeature([0.3, -0.2, 0], [-0.5, -1.2, 0])
    S = 0.8 * ell.kernel(f, d, 3e9) + 0.3 * edge.kernel(f, d, 3e9) \
        + 0.2 * np.exp(2j * wavenumber(f)[:, None] * (d @ [0.6, 0.1, 0])[None, :])
    data = ScatteringData(f, d, {"VV": S})
    parts = [Part("hull", [ell]), Part("wing_LE", [edge]), Part("canopy", region=([0.6, 0.1, 0], [0.1, 0.1, 0.1]))]
    r = SCFitter(data, parts, FitConfig(max_centres=10, target_residual_db=-35)).fit()
    assert r.residual_db < -30
    by_part = {sc.part for sc in r.model.centres}
    assert {"hull", "wing_LE", "canopy"} <= by_part
    hull = [sc for sc in r.model.centres if sc.part == "hull"][0]
    assert abs(np.mean(hull.coeffs["tt"]) - 0.8) < 0.05      # reflectivity recovered


def test_assembly_matches_direct_evaluation_and_mirror():
    f = np.array([3e9, 3.2e9])
    d = dir_from_az_el(np.arange(0, 360, 2.0), 0)
    data = _points_data([[0.2, -0.5, 0], [-0.1, -0.2, 0]], [1, 0.4], f, d)
    m = fit_scattering_centres(data, grid_bounds=([-1, -1, 0], [1, 1, 0]), max_centres=4).model
    pR = Pose.from_translation([0.5, -0.1, 0])
    pL = Pose.mirror([0, 1, 0], [0.5, 0.1, 0])
    asm = Assembly().add("R", m, pR).add("L", m, pL)
    tot = asm.evaluate(f, d)["tt"]
    direct = m.transformed(pR).evaluate(f, d)["tt"] + m.transformed(pL).evaluate(f, d)["tt"]
    assert np.allclose(tot, direct)
    # truth: mirrored points
    k = wavenumber(f)
    truth = sum(a * np.exp(2j * k[:, None] * (d @ (pose.apply_points(p)))[None, :])
                for pose in (pR, pL) for a, p in zip([1, 0.4], [[0.2, -0.5, 0], [-0.1, -0.2, 0]]))
    assert np.linalg.norm(tot - truth) / np.linalg.norm(truth) < 0.02
    # mirrored model is symmetric: RCS(az) == RCS(-az) for the pair sum about y
    asm2 = Assembly().add("L", m, pL)
    az = np.arange(0, 360, 2.0)
    d_m = dir_from_az_el(-az, 0)
    a1 = Assembly().add("R", m, Pose.from_translation([0, 0, 0])).evaluate(f, d)["tt"]
    a2 = Assembly().add("L", m, Pose.mirror([0, 1, 0])).evaluate(f, d_m)["tt"]
    assert np.allclose(a1, a2)


def test_polarisation_rotation_about_line_of_sight():
    # a horizontal dipole-like scatterer (only HH) rolled 90 deg about x, seen from +x, becomes VV
    d = np.array([[1.0, 0, 0]])
    model = ScatteringCentreModel([], ("tt", "pp"), 3e9)
    from emscatt.model import ScatteringCentre
    model.centres.append(ScatteringCentre(PointFeature([0, 0, 0]), AngularBasis.constant(),
                                          {"tt": np.array([0j]), "pp": np.array([1 + 0j])}))
    asm = Assembly().add("dip", model, Pose.from_yaw_pitch_roll(roll=90))
    out = asm.evaluate([3e9], d, channels=("tt", "pp", "tp"))
    assert abs(abs(out["tt"][0, 0]) - 1) < 1e-9 and abs(out["pp"][0, 0]) < 1e-9
    asm45 = Assembly().add("dip", model, Pose.from_yaw_pitch_roll(roll=45))
    o = asm45.evaluate([3e9], d, channels=("tt", "pp", "tp", "pt"))
    assert np.allclose([abs(o[c][0, 0]) for c in ("tt", "pp", "tp", "pt")], 0.5)
    T = polarisation_transform(Pose.from_yaw_pitch_roll(yaw=40), dir_from_az_el(np.arange(0, 360, 10.0), 0))
    assert np.allclose(T, np.eye(2))


def test_coverage_flags_out_of_plane_poses():
    f = np.array([3e9])
    d = dir_from_az_el(np.arange(0, 360, 1.0), 0)
    data = _points_data([[0.2, 0, 0]], [1], f, d)
    m = fit_scattering_centres(data, grid_bounds=([-1, -1, 0], [1, 1, 0]), max_centres=2).model
    asm = Assembly().add("a", m, Pose.from_yaw_pitch_roll(yaw=33))
    assert not asm.coverage(d)["a"].any()
    asm = Assembly().add("a", m, Pose.from_yaw_pitch_roll(roll=20))
    assert asm.coverage(d)["a"].mean() > 0.5


def test_interaction_calibration_recovers_missing_term():
    f = np.linspace(9e9, 10e9, 11)
    d = dir_from_az_el(np.arange(0, 360, 0.5), 0)
    comp = _points_data([[0.3, 0.0, 0]], [1.0], f, d)
    m = fit_scattering_centres(comp, grid_bounds=([-1, -1, 0], [1, 1, 0]), max_centres=3).model
    full = _points_data([[0.3, 0.0, 0], [-0.6, 0.4, 0]], [1.0, 0.3], f, d)
    asm = Assembly().add("c", m)
    r = asm.calibrate_interactions(full, grid_bounds=([-1, -1, 0], [1, 1, 0]), max_centres=3)
    p = r.model.centres[0].feature.position
    assert np.linalg.norm(p - [-0.6, 0.4, 0]) < 0.01
    pred = asm.evaluate(f, d)["tt"]
    assert np.linalg.norm(pred - full.S["tt"]) / np.linalg.norm(full.S["tt"]) < 0.01


def test_model_and_data_roundtrip(tmp_path):
    f = np.array([3e9, 3.1e9])
    d = dir_from_theta_phi(90, np.arange(0, 360, 5.0))
    data = _points_data([[0.2, 0.1, 0]], [1], f, d)
    data.save_npz(tmp_path / "d.npz")
    data.save_csv(tmp_path / "d.csv")
    d2 = ScatteringData.load_npz(tmp_path / "d.npz")
    d3 = ScatteringData.load_csv(tmp_path / "d.csv", channels={"tt": "tt"})
    assert np.allclose(d2.S["tt"], data.S["tt"])
    order = np.argsort(np.round(np.arctan2(data.dirs[:, 1], data.dirs[:, 0]), 9))
    assert np.allclose(np.sort_complex(d3.S["tt"].ravel()), np.sort_complex(data.S["tt"].ravel()))
    m = fit_scattering_centres(data, grid_bounds=([-1, -1, 0], [1, 1, 0]), max_centres=2).model
    m.save(tmp_path / "m.json")
    m2 = ScatteringCentreModel.load(tmp_path / "m.json")
    assert np.allclose(m.evaluate(f, d)["tt"], m2.evaluate(f, d)["tt"])


def test_phase_reference_shift():
    f = np.array([3e9, 3.3e9])
    d = dir_from_az_el(np.arange(0, 360, 3.0), 0)
    p0 = np.array([0.4, -0.1, 0])
    measured = _points_data([np.array([0.5, 0.2, 0]) - p0], [1], f, d)   # referenced to p0
    fixed = measured.with_phase_reference(p0)
    ref = _points_data([[0.5, 0.2, 0]], [1], f, d)
    assert np.allclose(fixed.S["tt"], ref.S["tt"])

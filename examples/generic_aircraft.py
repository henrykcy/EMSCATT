"""Geometry of a *generic* (non-specific) small aircraft used for validation.

Each component is defined in its own local "measurement" frame, exactly as it
would be mounted on a turntable for a component RCS measurement, together
with (a) a triangulated mesh (used only to synthesise PO "measurement" data)
and (b) a list of :class:`Part` objects with candidate scattering features
derived from the CAD description (used by the fitter).

Global frame: x forward (nose), y to port (left), z up.
"""
import numpy as np

from emscatt import (CylinderFeature, EdgeFeature, EllipsoidFeature, Mesh, Part, PlateFeature, Pose,
                     rotation_about_axis)
from emscatt.conventions import unit

T_WING = 0.025      # wing half thickness (m)


# ---------------------------------------------------------------- fuselage
def fuselage():
    R = rotation_about_axis([0, 1, 0], 90)            # mesh poles along x
    mesh = Mesh.ellipsoid([0.22, 0.22, 2.0], n_u=40, n_v=120, name="fuselage").transformed(Pose(R))
    parts = [Part("fuselage", [EllipsoidFeature([0, 0, 0], [2.0, 0.22, 0.22], name="hull")])]
    return mesh, parts


# ------------------------------------------------------------------ wing
def wing(sweep_deg=35.0, span=1.8, root_chord=1.1, tip_chord=0.45):
    """Starboard wing (span along -y), root leading edge at the origin."""
    rle = np.array([0.0, 0.0, 0.0])
    tle = np.array([-span * np.tan(np.deg2rad(sweep_deg)), -span, 0.0])
    rte = rle + [-root_chord, 0, 0]
    tte = tle + [-tip_chord, 0, 0]
    t = np.array([0, 0, T_WING])
    top = Mesh.polygon([rle + t, rte + t, tte + t, tle + t], normal=[0, 0, 1])
    bot = Mesh.polygon([rle - t, rte - t, tte - t, tle - t], normal=[0, 0, -1])
    le = Mesh.cylinder(rle, tle, T_WING, n_circ=24)
    te_n = unit(np.cross(tte - rte, [0, 0, 1]))
    te_n = te_n if te_n[0] < 0 else -te_n
    te = Mesh.polygon([rte + t, rte - t, tte - t, tte + t], normal=te_n)
    tip = Mesh.polygon([tle + t, tle - t, tte - t, tte + t], normal=[0, -1, 0])
    mesh = Mesh.merge([top, bot, le, te, tip])
    mesh.name = "wing"
    parts = [
        # rounded leading edge: offer both a sliding cylinder and a sharp edge, the fit decides
        Part("leading_edge", [CylinderFeature(rle, tle, T_WING, name="LE_round"),
                              EdgeFeature(rle, tle, name="LE_sharp")]),
        Part("trailing_edge", [EdgeFeature(rte, tte, name="TE")]),
        Part("tip", [EdgeFeature(tle, tte, name="tip")]),
        Part("root", region=((rle + rte) / 2, [root_chord / 2 + 0.05, 0.1, 0.1])),
    ]
    return mesh, parts


# ------------------------------------------------------------------- fin
def fin(sweep_deg=45.0, height=0.9, root_chord=0.9, tip_chord=0.4):
    """Vertical fin in the xz-plane, root leading edge at the origin."""
    rle = np.array([0.0, 0, 0])
    tle = np.array([-height * np.tan(np.deg2rad(sweep_deg)), 0, height])
    rte, tte = rle + [-root_chord, 0, 0], tle + [-tip_chord, 0, 0]
    poly = [rle, rte, tte, tle]
    panel = Mesh.polygon(poly, two_sided=True)
    le = Mesh.cylinder(rle, tle, 0.012, n_circ=16)
    mesh = panel + le
    mesh.name = "fin"
    parts = [
        Part("panel", [PlateFeature(poly, two_sided=True, name="panel")]),
        Part("leading_edge", [CylinderFeature(rle, tle, 0.012, name="LE_round"),
                              EdgeFeature(rle, tle, name="LE_sharp")]),
        Part("trailing_edge", [EdgeFeature(rte, tte, name="TE")]),
    ]
    return mesh, parts


# ---------------------------------------------------------------- intake
def intake(radius=0.15, length=0.8):
    """Ventral intake pod along -x with its (blocked) compressor face at the origin."""
    face = Mesh.disk([-0.05, 0, 0], [1, 0, 0], radius * 0.95, n=32)
    pod = Mesh.cylinder([0, 0, 0], [-length, 0, 0], radius, n_circ=48)
    mesh = face + pod
    mesh.name = "intake"
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    disk_poly = np.stack([np.full_like(ang, -0.05), 0.95 * radius * np.cos(ang), 0.95 * radius * np.sin(ang)], 1)
    parts = [
        Part("face", [PlateFeature(disk_poly, normal=[1, 0, 0], name="face")]),
        Part("pod", [CylinderFeature([0, 0, 0], [-length, 0, 0], radius, name="pod")]),
    ]
    return mesh, parts


# ------------------------------------------------------------ the airframe
def poses():
    return {
        "fuselage": Pose(),
        "wing_R": Pose.from_translation([0.55, -0.18, 0.0]),
        "wing_L": Pose.mirror([0, 1, 0], translation=[0.55, 0.18, 0.0]),
        "fin": Pose.from_translation([-1.0, 0.0, 0.15]),
        "intake": Pose.from_translation([1.2, 0.0, -0.32]),
    }


def components(wing_kwargs=None):
    """{name: (local_mesh, local_parts, pose)}; the port wing re-uses the starboard wing."""
    w = wing(**(wing_kwargs or {}))
    P = poses()
    return {
        "fuselage": (*fuselage(), P["fuselage"]),
        "wing_R": (*w, P["wing_R"]),
        "wing_L": (*w, P["wing_L"]),
        "fin": (*fin(), P["fin"]),
        "intake": (*intake(), P["intake"]),
    }


def global_mesh(comps):
    return Mesh.merge([m.transformed(p) for m, _, p in comps.values()])


def global_parts(comps):
    out = []
    for cname, (_, parts, pose) in comps.items():
        for p in parts:
            out.append(p.transformed(pose, name=f"{cname}/{p.name}"))
    return out


def interaction_truth(freqs, dirs):
    """Synthetic stand-in for installation effects that isolated components do not
    contain (wing-root/fuselage multi-bounce): two junction scatterers with
    rear-quarter lobes, mirror-symmetric."""
    from emscatt.conventions import az_el_from_dir, wavenumber
    k = wavenumber(freqs)
    az, _ = az_el_from_dir(dirs)
    S = np.zeros((len(freqs), len(dirs)), complex)
    for y, az0 in ((-0.25, -125.0), (0.25, 125.0)):
        x = np.array([-0.5, y, 0.0])
        d_az = (az - az0 + 180) % 360 - 180
        lobe = 0.8 * np.exp(-0.5 * (d_az / 10.0) ** 2)
        S += lobe[None, :] * np.exp(2j * k[:, None] * (dirs @ x)[None, :]) * np.exp(0.7j)
    return S


def outline(comps):
    """Polylines for plotting the planform."""
    lines = []
    for cname, (mesh, parts, pose) in comps.items():
        for p in parts:
            for f in p.features:
                g = f.transformed(pose)
                if hasattr(g, "endpoints"):
                    lines.append(np.array(g.endpoints))
                elif g.kind == "plate":
                    lines.append(np.vstack([g.vertices, g.vertices[:1]]))
                elif g.kind == "ellipsoid":
                    t = np.linspace(0, 2 * np.pi, 200)
                    lines.append(g.centre + np.stack([g.semi_axes[0] * np.cos(t), g.semi_axes[1] * np.sin(t),
                                                      0 * t], 1) @ g.rotation.T)
                elif g.kind == "cylinder":
                    a = g.centre - g.length / 2 * g.axis
                    b = g.centre + g.length / 2 * g.axis
                    n = unit(np.cross(g.axis, [0, 0, 1]))
                    lines.append(np.array([a + g.radius * n, b + g.radius * n, b - g.radius * n,
                                           a - g.radius * n, a + g.radius * n]))
    return lines

"""EMSCATT - scattering-centre modelling and coherent component assembly for RCS.

Workflow
--------
1. Load complex scattering data (measured and/or simulated)    -> :mod:`emscatt.data`
2. Describe the object's physical parts with candidate features -> :mod:`emscatt.features`
3. Extract a part-labelled scattering-centre model              -> :mod:`emscatt.fitting`
4. Place component models with poses and sum them coherently    -> :mod:`emscatt.assembly`
5. Calibrate interaction terms against a full-target dataset, then iterate
   components and track the change of the total RCS              -> :mod:`emscatt.metrics`
"""
from .conventions import (C0, CHANNELS, Pose, angle_between, az_el_from_dir, canonical_channel, dir_from_az_el,
                          dir_from_theta_phi, fibonacci_sphere, rotation_about_axis, theta_phi_from_dir, wavenumber)
from .data import ScatteringData, plane_normal, sampling_check
from .basis import AngularBasis
from .features import (CylinderFeature, EdgeFeature, EllipsoidFeature, Feature, Part, PlateFeature, PointFeature,
                       assign_part)
from .model import ScatteringCentre, ScatteringCentreModel
from .fitting import FitConfig, FitResult, SCFitter, fit_scattering_centres
from .assembly import Assembly, Component
from .po import Mesh, po_monostatic
from .imaging import backproject, plane_grid
from . import metrics

__version__ = "0.1.0"

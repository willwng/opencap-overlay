import opensim as osim
from dataclasses import dataclass, field
from typing import Any

import pyvista as pv
import os
import numpy as np


@dataclass
class MeshMotion:
    """One mesh and its per-frame world pose over the motion"""
    name: str
    mesh_file: str
    scale: list  # [sx, sy, sz] baked into the glTF node scale
    frame: osim.PhysicalFrame
    translation: Any = field(default_factory=list)  # (T, 3) (xyz)
    rotation: Any = field(default_factory=list)  # (T, 4) (xyzw quat)


def load_geometry(mesh_file, geometry_dir):
    """Read an OpenSim mesh file (.vtp/.stl/...) to vertices + triangle indices."""
    mesh = pv.read(os.path.join(geometry_dir, mesh_file)).triangulate()
    return np.asarray(mesh.points, dtype=np.float32), mesh.regular_faces.astype(np.uint32)

from dataclasses import dataclass

import numpy as np
import opensim as osim


@dataclass(eq=False)
class MeshMotion:
    """One mesh and its per-frame world pose over the motion"""
    name: str
    mesh_file: str
    scale: np.array
    frame: osim.PhysicalFrame
    translation: np.ndarray  # (T, 3)
    rotation: np.ndarray  # (T, 4)


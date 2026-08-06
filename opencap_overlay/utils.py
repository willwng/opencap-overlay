import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import opensim as osim
import pyvista as pv

CAMERA_KEYS = ('intrinsicMat', 'rotation', 'translation', 'imageSize')


@dataclass
class MeshMotion:
    """One mesh and its per-frame world pose over the motion"""
    name: str
    mesh_file: str
    scale: list
    frame: osim.PhysicalFrame
    translation: Any = field(default_factory=list)  # (T, 3) (xyz)
    rotation: Any = field(default_factory=list)  # (T, 4) (xyzw quat)


def load_geometry(mesh_file, geometry_dir):
    """Read an OpenSim mesh file (.vtp/.stl/...) to vertices + triangle indices."""
    mesh = pv.read(os.path.join(geometry_dir, mesh_file)).triangulate()
    return np.asarray(mesh.points, dtype=np.float32), mesh.regular_faces.astype(np.uint32)


def load_camera(pickle_path):
    with open(pickle_path, 'rb') as fh:
        cal = pickle.load(fh)
    return {k: np.asarray(cal[k]).tolist() for k in CAMERA_KEYS}


def frames_to_video(frames_dir, out_path, fps, pattern='frame_%04d.png', start_number=1):
    """Stitch rendered PNG frames into an mp4 with ffmpeg."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg not found on PATH')
    subprocess.run([
        ffmpeg, '-y',
        '-framerate', str(fps),
        '-start_number', str(start_number),
        '-i', os.path.join(frames_dir, pattern),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        os.path.abspath(out_path),
    ], check=True)

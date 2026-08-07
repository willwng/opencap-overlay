import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import opensim as osim
import pyvista as pv


@dataclass
class MeshMotion:
    """One mesh and its per-frame world pose over the motion"""
    name: str
    mesh_file: str
    scale: list
    frame: osim.PhysicalFrame
    translation: Any = field(default_factory=list)  # (T, 3) (xyz)
    rotation: Any = field(default_factory=list)  # (T, 4) (xyzw quat)


def apply_custom_geometry_map(
        mesh_file: str,
        custom_geometry_map: dict[str, str]
) -> str:
    ext = Path(mesh_file).suffix
    base_name = Path(mesh_file).stem
    mapped_name = custom_geometry_map.get(base_name, base_name)
    return f"{mapped_name}{ext}"


def load_geometry(mesh_file, geometry_dir):
    """Read an OpenSim mesh file (.vtp/.stl/...) to vertices + triangle indices."""
    mesh = pv.read(os.path.join(geometry_dir, mesh_file)).triangulate()
    return np.asarray(mesh.points, dtype=np.float32), mesh.regular_faces.astype(np.uint32)


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


def rm_file_or_folder(path):
    if not os.path.exists(path):
        return

    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def as_rotation(R):
    R = np.array(R, dtype=float)
    if abs(np.linalg.det(R) - 1.0) < 1e-4:
        return R
    raise ValueError(f"Invalid rotation matrix with det={np.linalg.det(R)}:\n{R}")

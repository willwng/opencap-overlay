import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv


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


def find_ffmpeg():
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg not found on PATH')
    return ffmpeg


def frames_to_video(frames_dir, out_path, fps, pattern='frame_%04d.png', start_number=1):
    """Stitch rendered PNG frames into a mp4"""
    ffmpeg = find_ffmpeg()
    subprocess.run([
        ffmpeg, '-y',
        '-framerate', str(fps),
        '-start_number', str(start_number),
        '-i', os.path.join(frames_dir, pattern),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        os.path.abspath(out_path),
    ], check=True)


def stack_videos_horizontal(video_paths, out_path):
    """Stitch videos side by side into one mp4"""
    ffmpeg = find_ffmpeg()

    n = len(video_paths)
    labels = ''.join(f'[{i}:v]' for i in range(n))
    cmd = [ffmpeg, '-y']
    for v in video_paths:
        cmd += ['-i', os.path.abspath(v)]
    cmd += ['-filter_complex', f'{labels}hstack=inputs={n}',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.abspath(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def rm_file_or_folder(path):
    """ Remove file or folder at given path """
    if not os.path.exists(path):
        return

    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
    return

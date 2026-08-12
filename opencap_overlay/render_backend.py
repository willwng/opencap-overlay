""" pyrender backend """
import os
from typing import Optional

import numpy as np
import cv2
import pyrender
import trimesh
from tqdm import tqdm
from PIL import Image
from contextlib import contextmanager

from .camera import Camera
from .motion import MeshMotion
from .utils import load_geometry


@contextmanager
def managed_renderer(renderer):
    try:
        yield renderer
    finally:
        renderer.delete()


def _quat_to_mat(q):
    """3x3 rotation from a xyzw quaternion."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]])


def _camera_pose(camera: Camera):
    """Camera-to-world 4x4 (OpenGL convention) from the camera extrinsics.

    Standard OpenCV -> OpenGL: camera centre C = -R^T t, camera-to-world rotation
    R^T with the camera Y/Z axes flipped (OpenGL looks down -Z). The extrinsics are
    expected to already map the model's world to the camera (see
    OpenCapOverlayTool.update_camera), so no extra world rotation is applied.
    """
    R = camera.rotation
    t = camera.translation
    pose = np.eye(4)
    pose[:3, :3] = R.T @ np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = -R.T @ t
    return pose


def _direction_pose(direction):
    """4x4 whose local -Z axis points along `direction` (a DirectionalLight shines
    down its -Z). `direction` is where the light travels, in OpenSim world (Y up)."""
    z = -np.asarray(direction, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
    return pose


def render_pyrender(
        mesh_motions: list[MeshMotion],
        geometry_dir: str,
        camera: Camera,
        frames_dir: str,
        num_frames: int,
        motion_times: np.ndarray,
        background_video: Optional[str] = None,
        opacity: float = 1.0,
        markers: Optional[np.ndarray] = None
):
    """Render the motion to frames_dir/frame_XXXX.png.

    markers, if given, is a (num_frames, N, 3) array of experimental marker
    positions (OpenSim world, metres) rendered as spheres; NaN = occluded.
    """

    # Camera intrinsics
    # ----------
    K = camera.intrinsicMat
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = (int(round(v)) for v in camera.imageSize)

    # Background: Transparent when overlaying
    # ----------
    bg = [0.0, 0.0, 0.0, 0.0] if background_video else [0.05, 0.05, 0.06, 1.0]
    scene = pyrender.Scene(bg_color=bg, ambient_light=[0.2, 0.2, 0.2])
    cam_pose = _camera_pose(camera)
    scene.add(
        pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            znear=0.01, zfar=100.0),
        pose=cam_pose
    )

    # Lighting
    # ----------
    for intensity, direction in [(4.0, (-0.5, -1.0, -0.6)),
                                 (1.5, (0.7, -0.2, -0.5)),
                                 (2.0, (0.2, 0.6, 0.8))]:
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity),
                  pose=_direction_pose(direction))

    # Build meshes
    # ----------
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.83, 0.78, 1.0], metallicFactor=0.0, roughnessFactor=0.6)
    geom = {}
    for m in mesh_motions:
        if m.mesh_file not in geom:
            verts, faces = load_geometry(m.mesh_file, geometry_dir)
            tm = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                                 faces=np.asarray(faces), process=False)
            geom[m.mesh_file] = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)

    # Build a node per each MeshMotion
    # ----------
    nodes = [(scene.add(geom[m.mesh_file]), m.scale, m.translation, m.rotation) for m in mesh_motions]

    # Marker spheres (optional)
    # ----------
    marker_nodes = []
    if markers is not None:
        sphere = trimesh.creation.uv_sphere(radius=0.02)
        marker_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.9, 0.1, 0.1, 1.0], emissiveFactor=[0.5, 0.0, 0.0],
            metallicFactor=0.0, roughnessFactor=0.5)
        marker_mesh = pyrender.Mesh.from_trimesh(sphere, material=marker_mat)
        marker_nodes = [scene.add(marker_mesh) for _ in range(markers.shape[1])]

    # Preload the reference video (converted to the render's RGB and size).
    video_frames = []
    video_fps = 0.0
    if background_video:
        cap = cv2.VideoCapture(background_video)
        if not cap.isOpened():
            raise ValueError(f'could not open background video: {background_video}')
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            video_frames.append(frame)
        cap.release()

    # Motion times/fps
    t0, t1 = float(motion_times[0]), float(motion_times[-1])
    motion_fps = (num_frames - 1) / (t1 - t0)

    # Output runs at the video's fps (real-time) and covers the whole video plus
    # any motion beyond it; without a video it's just the motion at its own rate.
    if video_frames:
        out_fps = video_fps
        n_out = max(len(video_frames), int(np.ceil(t1 * out_fps)) + 1)
    else:
        out_fps = motion_fps
        n_out = num_frames

    # Start Render
    # ----------
    os.makedirs(frames_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    flags = pyrender.RenderFlags.RGBA if video_frames else pyrender.RenderFlags.NONE
    with managed_renderer(renderer):
        for i in tqdm(range(n_out), desc='Rendering frames', unit='frame'):
            # Determine which frame of motion to use
            if video_frames:
                t = i / out_fps  # wall-clock time of this frame
                mf = int(round((t - t0) * motion_fps))
                mf = None if (mf < 0 or mf >= num_frames) else min(mf, num_frames - 1)  # None = IK not captured
            else:
                mf = i

            # Render the scene with the meshes at their current pose
            color = None
            if mf is not None:
                for node, scale, trans, rot in nodes:
                    # world vertex = R (scale * v) + p, in OpenSim world coordinates.
                    M = np.eye(4)
                    M[:3, :3] = _quat_to_mat(rot[mf]) @ np.diag(scale)
                    M[:3, 3] = trans[mf]
                    scene.set_pose(node, M)
                for j, mnode in enumerate(marker_nodes):
                    p = markers[mf, j]
                    M = np.eye(4)
                    # Occluded (NaN) markers get parked far away so they clip out.
                    M[:3, 3] = 1e4 if np.isnan(p).any() else p
                    scene.set_pose(mnode, M)
                color, _ = renderer.render(scene, flags=flags)

            # Blend with the video frame if available
            if video_frames:
                frame = video_frames[min(i, len(video_frames) - 1)]  # hold last frame
                if mf is None:
                    out = frame  # before the motion starts: video only
                else:
                    a = color[..., 3:4].astype(float) / 255.0 * opacity
                    out = (a * color[..., :3] + (1 - a) * frame).astype(np.uint8)
            else:
                out = color
            Image.fromarray(out).save(os.path.join(frames_dir, f'frame_{i + 1:04d}.png'))
    return out_fps

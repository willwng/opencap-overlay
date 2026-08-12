""" pyrender backend """
import os
from contextlib import contextmanager
from typing import Optional

import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image
from tqdm import tqdm

from .camera import Camera
from .motion import MeshMotion
from .utils import load_geometry, get_video_times


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
    """Render the motion to frames_dir/frame_XXXX.png. """

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

    # Preload reference video and get actual per-frame timestamps
    # ----------
    video_frames = []
    video_times = np.array([], dtype=float)
    if background_video:
        video_times = get_video_times(background_video)
        cap = cv2.VideoCapture(background_video)
        if not cap.isOpened():
            raise ValueError(f'could not open background video: {background_video}')

        # Reshape and convert video frames
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            video_frames.append(frame)
        cap.release()

    # Motion timestamps (prefer the video timestamps if available)
    # ----------
    motion_times = np.asarray(motion_times, dtype=float)
    output_times = video_times if video_frames else motion_times

    # Start Render
    # ----------
    os.makedirs(frames_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    flags = pyrender.RenderFlags.RGBA if video_frames else pyrender.RenderFlags.NONE

    with managed_renderer(renderer):
        for i, t in enumerate(tqdm(output_times, desc="Rendering frames", unit="frame")):
            # Find the motion frame corresponding to this timestamp
            if video_frames:
                mf = np.searchsorted(motion_times, t)

                if mf == 0:  # Possibly before the first motion frame
                    mf = None if t < motion_times[0] else 0
                elif mf >= num_frames:  # After the last motion frame
                    mf = None
                else:
                    # Choose whichever neighboring motion frame is closer
                    before = mf - 1
                    after = mf
                    if abs(motion_times[before] - t) <= abs(motion_times[after] - t):
                        mf = before
                    else:
                        mf = after
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

                color, _ = renderer.render(
                    scene,
                    flags=flags,
                )

            # Blend with video frame if available
            if video_frames:
                frame = video_frames[i]
                if mf is None:
                    # Video only before/after motion.
                    out = frame
                else:
                    a = color[..., 3:4].astype(float) / 255.0 * opacity
                    out = (a * color[..., :3] + (1 - a) * frame).astype(np.uint8)
            else:
                out = color
            Image.fromarray(out).save(os.path.join(frames_dir, f'frame_{i + 1:04d}.png'))

    # effective average frame rate
    out_fps = (len(output_times) - 1) / (output_times[-1] - output_times[0])
    return out_fps

""" pyrender backend """
import os
from typing import Optional

import numpy as np
import cv2
import pyrender
import trimesh
from PIL import Image

from .camera import Camera
from .utils import load_geometry, MeshMotion


def _quat_to_mat(q):
    """3x3 rotation from an xyzw quaternion."""
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
    """Camera-to-world 4x4 (OpenGL convention) from an calibration dict """
    R = np.asarray(camera.rotation, dtype=float)  # world -> cam
    t = np.asarray(camera.translation, dtype=float).reshape(3) / 1000.0  # mm -> m
    pose = np.eye(4)
    pose[:3, :3] = R.T @ np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = -R.T @ t
    W = np.diag([1.0, -1.0, -1.0, 1.0])  # calibration world -> OpenSim world
    return W @ pose


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
        motion_times,
        background_video: Optional[str] = None,
        opacity: float = 1.0
):
    """Render the motion to frames_dir/frame_XXXX.png with pyrender.

    Returns (frames_dir, out_fps). motion_times are the motion's absolute
    timestamps (seconds), shared with the video clock. When compositing over a
    video, the model is placed at each frame's wall-clock time: video-only before
    the motion's first timestamp, the last pose held after its last, so both play
    at true speed. Without a video the motion plays start-to-end at its own rate.
    """
    if camera is None:
        raise ValueError('pyrender backend needs camera intrinsics/extrinsics')

    K = np.asarray(camera.intrinsicMat, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = (int(round(v)) for v in np.asarray(camera.imageSize, dtype=float).ravel()[:2])

    # Transparent background when compositing so the model's alpha is clean.
    bg = [0.0, 0.0, 0.0, 0.0] if background_video else [0.05, 0.05, 0.06, 1.0]
    scene = pyrender.Scene(bg_color=bg, ambient_light=[0.2, 0.2, 0.2])
    cam_pose = _camera_pose(camera)
    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                        znear=0.01, zfar=100.0), pose=cam_pose)

    # Lights come from off the camera axis so curvature shades from light to dark
    # -> depth perception. Directions are where each light travels (OpenSim Y up):
    # key from upper front-left, softer fill from the front-right, rim from behind
    # above for edge separation.
    for intensity, direction in [(4.0, (-0.5, -1.0, -0.6)),
                                 (1.5, (0.7, -0.2, -0.5)),
                                 (2.0, (0.2, 0.6, 0.8))]:
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity),
                  pose=_direction_pose(direction))

    # Matte bone-ish material; pyrender's default is fully metallic and reads flat.
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.83, 0.78, 1.0], metallicFactor=0.0, roughnessFactor=0.6)

    # One pyrender mesh per unique geometry file; nodes reuse (instance) them.
    geom = {}
    for m in mesh_motions:
        if m.mesh_file not in geom:
            verts, faces = load_geometry(m.mesh_file, geometry_dir)
            tm = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                                 faces=np.asarray(faces), process=False)
            geom[m.mesh_file] = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)

    # One node per MeshMotion, updated in place each frame.
    nodes = [(scene.add(geom[m.mesh_file]), np.asarray(m.scale, dtype=float),
              m.translation, m.rotation) for m in mesh_motions]

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
        if video_fps <= 0:
            video_fps = motion_fps  # container lacked fps metadata

    # Absolute motion timeline (shared clock with the video).
    motion_times = np.asarray(motion_times, dtype=float)
    t0, t1 = float(motion_times[0]), float(motion_times[-1])
    motion_fps = (num_frames - 1) / (t1 - t0) if t1 > t0 else 30.0

    # Output runs at the video's fps (real-time) and covers the whole video plus
    # any motion beyond it; without a video it's just the motion at its own rate.
    if video_frames:
        out_fps = video_fps
        n_out = max(len(video_frames), int(np.ceil(t1 * out_fps)) + 1)
    else:
        out_fps = motion_fps
        n_out = num_frames

    os.makedirs(frames_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    flags = pyrender.RenderFlags.RGBA if video_frames else pyrender.RenderFlags.NONE
    try:
        for i in range(n_out):
            if video_frames:
                t = i / out_fps  # wall-clock time (video clock) of this output frame
                mf = int(round((t - t0) * motion_fps))
                mf = None if mf < 0 else min(mf, num_frames - 1)  # None = pre-motion
            else:
                mf = i

            color = None
            if mf is not None:
                for node, scale, trans, rot in nodes:
                    # world vertex = R (scale * v) + p, in OpenSim world coordinates.
                    M = np.eye(4)
                    M[:3, :3] = _quat_to_mat(rot[mf]) @ np.diag(scale)
                    M[:3, 3] = trans[mf]
                    scene.set_pose(node, M)
                color, _ = renderer.render(scene, flags=flags)

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
    finally:
        renderer.delete()
    return frames_dir, out_fps

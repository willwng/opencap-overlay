""" pyrender backend """
import os

import numpy as np


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


def _camera_pose(cal):
    """Camera-to-world 4x4 (OpenGL convention) from an calibration dict """
    R = np.asarray(cal['rotation'], dtype=float)  # world -> cam
    t = np.asarray(cal['translation'], dtype=float).reshape(3) / 1000.0  # mm -> m
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
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
    return pose


def render_pyrender(mesh_motions, geometry_dir, camera, frames_dir, num_frames):
    """Render every frame of the motion to frames_dir/frame_XXXX.png with pyrender."""
    import pyrender
    import trimesh
    from PIL import Image

    from .utils import load_geometry

    if camera is None:
        raise ValueError('pyrender backend needs camera intrinsics/extrinsics')

    K = np.asarray(camera['intrinsicMat'], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = (int(round(v)) for v in np.asarray(camera['imageSize'], dtype=float).ravel()[:2])

    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.06, 1.0],
                           ambient_light=[0.2, 0.2, 0.2])
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

    os.makedirs(frames_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        for f in range(num_frames):
            for node, scale, trans, rot in nodes:
                # world vertex = R (scale * v) + p, in OpenSim world coordinates.
                M = np.eye(4)
                M[:3, :3] = _quat_to_mat(rot[f]) @ np.diag(scale)
                M[:3, 3] = trans[f]
                scene.set_pose(node, M)
            color, _ = renderer.render(scene)
            Image.fromarray(color).save(os.path.join(frames_dir, f'frame_{f + 1:04d}.png'))
    finally:
        renderer.delete()
    return frames_dir

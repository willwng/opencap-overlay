"""
Blender side of the pipeline: turn MeshMotion poses into a ready-to-open .blend.

Two roles:
  * Imported by the user-facing side:
    - find_blender() finds the Blender executable
    - build_blend() bundles the scene (geometry + per-frame poses + camera) to a
      pickle and launches Blender headless to run this script
  * Executed inside Blender (blender -b -P backend_blender.py -- <bundle.pkl>
    <blend>): builds one object per MeshMotion, bakes per-frame world keyframes
    straight at integer frames, sets up the camera, and saves the .blend.
"""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile


def find_blender():
    """Locate the Blender executable (explicit path, PATH, or macOS app)."""
    mac = '/Applications/Blender.app/Contents/MacOS/Blender'
    for cand in (shutil.which('blender'), mac):
        if cand and os.path.exists(cand):
            return cand

    raise ValueError("Blender executable not found")


def build_blend(mesh_motions, geometry_dir, blend_path, blender_exe, camera=None):
    """Build a ready-to-open .blend directly from MeshMotion poses """
    from .utils import load_geometry

    geometries = {}
    for m in mesh_motions:
        if m.mesh_file not in geometries:
            geometries[m.mesh_file] = load_geometry(m.mesh_file, geometry_dir)
    nodes = [{'name': m.name, 'mesh_file': m.mesh_file, 'scale': list(m.scale),
              'translation': m.translation, 'rotation': m.rotation}
             for m in mesh_motions]
    bundle = {'num_frames': int(len(mesh_motions[0].translation)),
              'geometries': geometries, 'nodes': nodes, 'camera': camera}

    with tempfile.NamedTemporaryFile('wb', suffix='.pkl', delete=False) as fh:
        pickle.dump(bundle, fh, protocol=4)
        bundle_path = fh.name
    try:
        subprocess.run([blender_exe, '-b', '-P', os.path.abspath(__file__), '--',
                        bundle_path, os.path.abspath(blend_path)],
                       check=True, stdout=subprocess.DEVNULL)
    finally:
        os.remove(bundle_path)


def render_blend(blend_path, out_path, blender_exe, engine='BLENDER_WORKBENCH'):
    """Render a .blend to PNG(s) in a single headless Blender launch.

    frame=None renders the whole animation (the .blend's frame range) into the
    directory out_path as frame_0001.png, frame_0002.png, ...; otherwise renders
    that single frame to the file out_path. Uses the camera stored in the .blend.
    engine defaults to Workbench (fast, solid shading); pass 'BLENDER_EEVEE_NEXT'
    for a lit render.
    """
    setup = (
        'import bpy;'
        'sc=bpy.context.scene;'
        f'sc.render.engine={engine!r};'
        "sc.render.image_settings.file_format='PNG';"
    )
    os.makedirs(out_path, exist_ok=True)
    prefix = os.path.join(os.path.abspath(out_path), 'frame_')
    expr = setup + f'sc.render.filepath={prefix!r};bpy.ops.render.render(animation=True)'
    subprocess.run([blender_exe, '-b', os.path.abspath(blend_path), '--python-expr', expr],
                   check=True)


def _make_calibrated_camera(sc, cal):
    """Create a perspective camera matching an unpacked calibration dict
    """
    import bpy
    import mathutils
    import numpy as np

    K = np.asarray(cal['intrinsicMat'], dtype=float)
    R = np.asarray(cal['rotation'], dtype=float)  # world -> cam
    t = np.asarray(cal['translation'], dtype=float).reshape(3)  # world -> cam, mm
    h, w = (int(round(v)) for v in np.asarray(cal['imageSize'], dtype=float).ravel()[:2])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    cam_d = bpy.data.cameras.new('Cam0')
    cam = bpy.data.objects.new('Cam0', cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam

    # Intrinsics: match render resolution and pinhole to K
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.pixel_aspect_x, sc.render.pixel_aspect_y = 1.0, fx / fy
    cam_d.type = 'PERSP'
    cam_d.sensor_fit = 'HORIZONTAL'
    cam_d.sensor_width = 36.0
    cam_d.lens = fx / w * cam_d.sensor_width
    cam_d.shift_x = -(cx - w / 2.0) / w
    cam_d.shift_y = (cy - h / 2.0) / w
    cam_d.clip_start, cam_d.clip_end = 0.01, 1000.0

    # Extrinsics: camera centre in the calibration world (metres) is C = -R^T t.
    C = (-R.T @ t) / 1000.0
    # cam->world rotation for OpenCV axes, then flip to Blender camera axes.
    R_bcam2calib = R.T @ np.diag([1.0, -1.0, -1.0])
    # Calibration world -> scene world. The model arrives from OpenSim (Y up) via
    # glTF (Blender Z up) this lines the camera up with it. Adjust this rotation
    # if the model comes out mis-oriented in frame.
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])
    loc, rot = A @ C, A @ R_bcam2calib

    M = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = rot[i][j]
        M[i][3] = loc[i]
    cam.matrix_world = M
    return cam


def _run_in_blender():
    import math

    import bpy
    import mathutils

    argv = sys.argv[sys.argv.index('--') + 1:]
    bundle_path, blend = argv[:2]
    with open(bundle_path, 'rb') as fh:
        bundle = pickle.load(fh)
    num_frames = bundle['num_frames']
    camera = bundle['camera']

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, num_frames

    # OpenSim ground (Y up, metres) -> Blender (Z up), the same axis convention the
    # glTF importer used to apply, so the calibrated camera still lines up.
    S = mathutils.Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))

    # One mesh datablock per unique geometry file; objects reuse (instance) them.
    mesh_db = {}
    for mesh_file, (verts, faces) in bundle['geometries'].items():
        me = bpy.data.meshes.new(mesh_file)
        me.from_pydata(verts.tolist(), [], faces.tolist())
        me.update()
        mesh_db[mesh_file] = me

    # One object per MeshMotion, with per-frame world keyframes baked straight at
    # integer frames from the OpenSim poses (no glTF import, no retiming).
    objs = []
    for node in bundle['nodes']:
        obj = bpy.data.objects.new(node['name'], mesh_db[node['mesh_file']])
        sc.collection.objects.link(obj)
        obj.rotation_mode = 'QUATERNION'
        obj.scale = node['scale']  # constant over the motion
        trans, rot = node['translation'], node['rotation']
        for f in range(num_frames):
            x, y, z, w = rot[f]
            obj.location = S @ mathutils.Vector((float(trans[f][0]),
                                                 float(trans[f][1]),
                                                 float(trans[f][2])))
            obj.rotation_quaternion = (
                    S @ mathutils.Quaternion((w, x, y, z)).to_matrix()).to_quaternion()
            obj.keyframe_insert('location', frame=f + 1)
            obj.keyframe_insert('rotation_quaternion', frame=f + 1)
        objs.append(obj)
    sc.frame_set(1)

    if camera:
        # Perspective camera matching a calibrated OpenCV camera.
        _make_calibrated_camera(sc, camera)
    else:
        # Orthographic camera framing the run (Blender up = Z), looking along +Y.
        lo = mathutils.Vector((1e9, 1e9, 1e9))
        hi = -lo
        for f in range(1, num_frames + 1):
            sc.frame_set(f)
            bpy.context.view_layer.update()
            for o in objs:
                for c in o.bound_box:
                    wv = o.matrix_world @ mathutils.Vector(c)
                    lo = mathutils.Vector((min(lo.x, wv.x), min(lo.y, wv.y), min(lo.z, wv.z)))
                    hi = mathutils.Vector((max(hi.x, wv.x), max(hi.y, wv.y), max(hi.z, wv.z)))
        center, ext = (lo + hi) / 2, (hi - lo)
        cam_d = bpy.data.cameras.new('Camera')
        cam = bpy.data.objects.new('Camera', cam_d)
        sc.collection.objects.link(cam)
        cam_d.type = 'ORTHO'
        cam_d.clip_start = 0.1
        cam_d.clip_end = 100.0
        aspect = sc.render.resolution_x / sc.render.resolution_y
        cam_d.ortho_scale = max(ext.x, ext.z * aspect) * 1.15
        cam.location = (center.x, lo.y - 5.0, center.z)
        cam.rotation_euler = (math.radians(90), 0, 0)
        sc.camera = cam
        sc.frame_set(1)

    sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN'))
    sc.collection.objects.link(sun)
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(55), 0, math.radians(30))
    w = bpy.data.worlds.new('World')
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.05, 0.06, 1)
    bg.inputs[1].default_value = 0.7

    for space in (s for scr in bpy.data.screens for a in scr.areas
                  if a.type == 'VIEW_3D' for s in a.spaces if s.type == 'VIEW_3D'):
        space.shading.type = 'SOLID'
        space.shading.color_type = 'OBJECT'

    # Don't leave a .blend1 backup of the previous build behind on overwrite.
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=blend)


if __name__ == '__main__':
    _run_in_blender()

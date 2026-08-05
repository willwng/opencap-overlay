"""
Blender side of the pipeline: turn an animated .glb into a ready-to-open .blend.

Two roles:
  * Imported by mot_to_gltf.py (opensim env): use find_blender() and
    build_blend() to invoke Blender headlessly. Does NOT import bpy on import.
  * Executed inside Blender
    (blender -b -P blender_build.py -- <glb> <blend> <num_frames>): imports the
    .glb, retimes it to num_frames integer frames (no downsampling), bakes plain
    per-object world keyframes, frames the run with an ortho camera, and saves.
"""
import os
import shutil
import subprocess
import sys


def find_blender():
    """Locate the Blender executable (explicit path, PATH, or macOS app)."""
    mac = '/Applications/Blender.app/Contents/MacOS/Blender'
    for cand in (shutil.which('blender'), mac):
        if cand and os.path.exists(cand):
            return cand
    return None


def build_blend(glb_path, blend_path, blender_exe, num_frames):
    """Run Blender headlessly to build a ready-to-open .blend from the .glb."""
    subprocess.run([blender_exe, '-b', '-P', os.path.abspath(__file__), '--',
                    os.path.abspath(glb_path), os.path.abspath(blend_path), str(num_frames)],
                   check=True, stdout=subprocess.DEVNULL)


def _run_in_blender():
    import math

    import bpy
    import mathutils

    glb, blend, num_frames = sys.argv[sys.argv.index('--') + 1:][:3]
    num_frames = int(num_frames)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    sc = bpy.context.scene
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']

    def object_fcurves(o):
        ad = o.animation_data
        if not (ad and ad.action):
            return []
        act = ad.action
        slot = getattr(ad, 'action_slot', None)
        if slot is not None and getattr(act, 'layers', None):
            for layer in act.layers:
                for strip in layer.strips:
                    cb = strip.channelbag(slot) if hasattr(strip, 'channelbag') else None
                    if cb:
                        return cb.fcurves
        return act.fcurves

    # The importer keeps every motion sample but spreads them over fractional
    # frames at the scene fps. Retime each keyframe to one integer frame so the
    # .blend has exactly as many frames as the motion (no downsampling).
    for o in meshes:
        for fc in object_fcurves(o):
            for i, kp in enumerate(fc.keyframe_points):
                kp.co.x = kp.handle_left.x = kp.handle_right.x = i + 1
            fc.update()
    sc.frame_start, sc.frame_end = 1, num_frames

    # Sample world transforms (and bounds) at each integer frame.
    samples = {o.name: [] for o in meshes}
    lo = mathutils.Vector((1e9, 1e9, 1e9)); hi = -lo
    for f in range(1, num_frames + 1):
        sc.frame_set(f); bpy.context.view_layer.update()
        for o in meshes:
            mw = o.matrix_world.copy()
            samples[o.name].append(mw)
            for c in o.bound_box:
                w = mw @ mathutils.Vector(c)
                lo = mathutils.Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
                hi = mathutils.Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
    center, ext = (lo + hi) / 2, (hi - lo)

    # The glTF importer parents meshes to an Empty and animates local transforms,
    # which renders but doesn't refresh in the viewport on scrub. Replace it with
    # plain baked world-space keyframes on unparented objects.
    for empty in [o for o in bpy.data.objects if o.type == 'EMPTY']:
        bpy.data.objects.remove(empty, do_unlink=True)
    for o in meshes:
        o.animation_data_clear()
        o.parent = None
        o.matrix_parent_inverse.identity()
        o.rotation_mode = 'QUATERNION'
        for i, f in enumerate(range(1, num_frames + 1)):
            o.location, o.rotation_quaternion, o.scale = samples[o.name][i].decompose()
            o.keyframe_insert('location', frame=f)
            o.keyframe_insert('rotation_quaternion', frame=f)
            o.keyframe_insert('scale', frame=f)
    sc.frame_set(1)

    # Orthographic camera looking along +Y (Blender up = Z), down the run (X).
    cam_d = bpy.data.cameras.new('Camera'); cam = bpy.data.objects.new('Camera', cam_d)
    sc.collection.objects.link(cam)
    cam_d.type = 'ORTHO'; cam_d.clip_start = 0.1; cam_d.clip_end = 100.0
    aspect = sc.render.resolution_x / sc.render.resolution_y
    cam_d.ortho_scale = max(ext.x, ext.z * aspect) * 1.15
    cam.location = (center.x, lo.y - 5.0, center.z)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.camera = cam

    sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN'))
    sc.collection.objects.link(sun)
    sun.data.energy = 4.0; sun.rotation_euler = (math.radians(55), 0, math.radians(30))
    w = bpy.data.worlds.new('World'); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.05, 0.06, 1); bg.inputs[1].default_value = 0.7

    for space in (s for scr in bpy.data.screens for a in scr.areas
                  if a.type == 'VIEW_3D' for s in a.spaces if s.type == 'VIEW_3D'):
        space.shading.type = 'SOLID'; space.shading.color_type = 'OBJECT'

    bpy.ops.wm.save_as_mainfile(filepath=blend)


if __name__ == '__main__':
    _run_in_blender()

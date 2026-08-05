import argparse
import os

import opensim as osim

from blender_build import build_blend, find_blender
from mot_to_gltf import build_glb
from opensim_helper import process_motion

parser = argparse.ArgumentParser(
    description='Export an OpenSim model + IK motion to an animated glTF (.glb) '
                'and a ready-to-open Blender (.blend) scene.')
parser.add_argument('--model', help='Path to the .osim model.')
parser.add_argument('--mot', help='Path to the IK .mot motion file.')
args = parser.parse_args()


def get_out_path(mot_path):
    """<mot basename, minus a trailing _ik>.glb next to the motion file."""
    stem = os.path.splitext(os.path.basename(mot_path))[0]
    if stem.endswith('_ik'):
        stem = stem[:-len('_ik')]
    return os.path.join(os.path.dirname(os.path.abspath(mot_path)), stem + '.glb')


def main():
    geometry_dir = os.path.join(os.path.dirname(os.path.abspath(args.model)), 'Geometry')
    out_path = get_out_path(args.mot)
    print(f'Exporting {args.mot} -> {out_path} ...')

    # Process the mesh motions
    model = osim.Model(args.model)
    times, mesh_motions = process_motion(model, args.mot)

    # Build GLB
    build_glb(times, mesh_motions, geometry_dir, out_path)
    print(f'\t{len(times)} frames, {len(mesh_motions)} meshes')

    # Create the blender script
    blender = find_blender()
    if not blender:
        print('Blender not found')
        return
    blend_path = os.path.splitext(out_path)[0] + '.blend'
    print(f'\tBuilding {blend_path} ...')
    build_blend(out_path, blend_path, blender, num_frames=len(times))


if __name__ == '__main__':
    main()

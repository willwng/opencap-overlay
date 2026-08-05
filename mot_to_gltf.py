"""
Export an OpenSim model + IK motion (.osim + .mot) to an animated glTF (.glb).
"""

import numpy as np
import pygltflib as gltf
from utils import load_geometry

# glTF component types / buffer targets.
FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


class GLBBuilder:
    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []

    def _add_view(self, arr, target):
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        raw = np.ascontiguousarray(arr).tobytes()
        self.blob += raw
        self.views.append(gltf.BufferView(buffer=0, byteOffset=offset,
                                          byteLength=len(raw), target=target))
        return len(self.views) - 1

    def add_accessor(self, arr, comp_type, type_, target=None, minmax=False):
        view = self._add_view(arr, target)
        acc = gltf.Accessor(bufferView=view, componentType=comp_type,
                            count=int(arr.shape[0]), type=type_)
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc.min = flat.min(axis=0).tolist()
            acc.max = flat.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def build_glb(times, metas, geometry_dir, out_path):
    b = GLBBuilder()

    # One glTF mesh per unique geometry file (nodes reuse them).
    geom_mesh = {}
    meshes = []
    for meta in metas:
        mf = meta.mesh_file
        if mf in geom_mesh:
            continue
        verts, faces = load_geometry(mf, geometry_dir)
        pos = b.add_accessor(verts, FLOAT, 'VEC3', ARRAY_BUFFER, minmax=True)
        idx = b.add_accessor(faces.reshape(-1), UNSIGNED_INT, 'SCALAR', ELEMENT_ARRAY_BUFFER)
        meshes.append(gltf.Mesh(primitives=[gltf.Primitive(
            attributes=gltf.Attributes(POSITION=pos), indices=idx, mode=4)]))
        geom_mesh[mf] = len(meshes) - 1

    time_acc = b.add_accessor(times, FLOAT, 'SCALAR', minmax=True)

    # Root node parents every mesh node so the model is one object in Blender.
    nodes = [gltf.Node(name='model', children=[])]
    samplers, channels = [], []
    for meta in metas:
        node_i = len(nodes)
        # Bake frame 0 into the base transform so viewers that don't autoplay
        # the animation still show a correct pose (not a pile at the origin).
        nodes.append(gltf.Node(name=meta.name, mesh=geom_mesh[meta.mesh_file],
                               scale=meta.scale,
                               translation=meta.translation[0].tolist(),
                               rotation=meta.rotation[0].tolist()))
        nodes[0].children.append(node_i)

        t_acc = b.add_accessor(meta.translation, FLOAT, 'VEC3')
        r_acc = b.add_accessor(meta.rotation, FLOAT, 'VEC4')
        samplers.append(gltf.AnimationSampler(input=time_acc, output=t_acc, interpolation='LINEAR'))
        channels.append(gltf.AnimationChannel(
            sampler=len(samplers) - 1,
            target=gltf.AnimationChannelTarget(node=node_i, path='translation')))
        samplers.append(gltf.AnimationSampler(input=time_acc, output=r_acc, interpolation='LINEAR'))
        channels.append(gltf.AnimationChannel(
            sampler=len(samplers) - 1,
            target=gltf.AnimationChannelTarget(node=node_i, path='rotation')))

    while len(b.blob) % 4:
        b.blob.append(0)

    g = gltf.GLTF2(
        asset=gltf.Asset(version='2.0', generator='mot_to_gltf'),
        scene=0, scenes=[gltf.Scene(nodes=[0])],
        nodes=nodes, meshes=meshes,
        accessors=b.accessors, bufferViews=b.views,
        buffers=[gltf.Buffer(byteLength=len(b.blob))],
        animations=[gltf.Animation(name='ik', samplers=samplers, channels=channels)],
    )
    g.set_binary_blob(bytes(b.blob))
    g.save_binary(out_path)

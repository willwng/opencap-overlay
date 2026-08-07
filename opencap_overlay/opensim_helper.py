import opensim as osim
import numpy as np

from .motion import MeshMotion
from .utils import apply_custom_geometry_map


def load_trc(path):
    """ Returns times (T,), positions (T, N, 3) in metres, marker names """
    with open(path) as fh:
        lines = fh.read().splitlines()
    meta = lines[2].split('\t')
    units = meta[4].strip().lower() if len(meta) > 4 else 'm'
    names = [n for n in lines[3].split('\t')[2:] if n.strip()]
    data = np.atleast_2d(np.genfromtxt(lines[5:], delimiter='\t'))
    times = data[:, 1].astype(float)
    markers = data[:, 2:2 + 3 * len(names)].reshape(len(data), len(names), 3)
    if units == 'mm':
        markers = markers / 1000.0
    return times, markers, names


def process_motion(
        model: osim.Model,
        mot_path: str,
        custom_geometry_map: dict[str, str]
) -> tuple[np.ndarray, list[MeshMotion]]:
    state = model.initSystem()
    storage = osim.Storage(mot_path)
    if storage.isInDegrees():
        model.getSimbodyEngine().convertDegreesToRadians(storage)

    labels = storage.getColumnLabels()
    label_idx = {labels.get(i): i - 1 for i in range(1, labels.getSize())}

    def column_for(name):
        """ Helps with absolute vs relative coordinates in the motion file """
        if name in label_idx:
            return label_idx[name]
        suffix = f'/{name}/value'
        for label, idx in label_idx.items():
            if label.endswith(suffix):
                return idx
        return None

    coords = model.getCoordinateSet()
    coord_cols = [(coords.get(i), column_for(coords.get(i).getName()))
                  for i in range(coords.getSize())]
    coord_cols = [(c, col) for c, col in coord_cols if col is not None]

    # Cache mesh metadata once
    metas = []
    for comp in model.getComponentsList():
        m = osim.Mesh.safeDownCast(comp)
        if m is None:
            continue
        sf = m.get_scale_factors()

        # Use custom geometry if provided
        mesh_file = m.get_mesh_file()
        mesh_file = apply_custom_geometry_map(mesh_file, custom_geometry_map)

        metas.append(MeshMotion(
            name=m.getName(),
            mesh_file=mesh_file,
            scale=np.array([sf.get(0), sf.get(1), sf.get(2)]),
            frame=m.getFrame(),
            # These will be updated after processing the motion
            translation=np.empty(0),
            rotation=np.empty(0)
        ))

    # Traverse through the motion
    times = []
    translations = {meta: [] for meta in metas}
    rotations = {meta: [] for meta in metas}
    for r in range(storage.getSize()):
        sv = storage.getStateVector(r)
        data = sv.getData()
        times.append(sv.getTime())
        for coord, col in coord_cols:
            coord.setValue(state, data.get(col), False)
        model.realizePosition(state)
        for meta in metas:
            X = meta.frame.getTransformInGround(state)
            p = X.p()
            q = X.R().convertRotationToQuaternion()  # w, x, y, z
            translations[meta].append([p.get(0), p.get(1), p.get(2)])
            rotations[meta].append([q.get(1), q.get(2), q.get(3), q.get(0)])  # x, y, z, w

    # Update translations and rotations
    for meta in metas:
        meta.translation = np.stack(translations[meta], axis=0)  # (T, 3)
        meta.rotation = np.stack(rotations[meta], axis=0)  # (T, 4)
    return np.array(times, dtype=np.float32), metas

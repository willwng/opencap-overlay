import opensim as osim
import numpy as np

from utils import MeshMotion


def process_motion(
        model: osim.Model,
        mot_path: str
) -> tuple[np.ndarray, list[MeshMotion]]:
    state = model.initSystem()
    storage = osim.Storage(mot_path)
    if storage.isInDegrees():
        storage.convertToRadians()  # rotational columns -> radians

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
        metas.append(MeshMotion(
            name=m.getName(),
            mesh_file=m.get_mesh_file(),
            scale=[sf.get(0), sf.get(1), sf.get(2)],
            frame=m.getFrame(),
        ))

    # Traverse through the motion
    times = []
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
            meta.translation.append([p.get(0), p.get(1), p.get(2)])
            meta.rotation.append([q.get(1), q.get(2), q.get(3), q.get(0)])  # x, y, z, w

    for meta in metas:
        meta.translation = np.asarray(meta.translation, dtype=np.float32)
        meta.rotation = np.asarray(meta.rotation, dtype=np.float32)
    return np.asarray(times, dtype=np.float32), metas

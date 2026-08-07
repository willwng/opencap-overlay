import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CheckerboardPlacement(Enum):
    GROUND = 'ground'
    BACK_WALL = 'backWall'


# OpenCap rotates triangulated keypoints into OpenSim's frame (Y up) by fixed axis
# rotations set by the checkerboard placement, then runs IK. To project the model
# back onto video we invert that rotation and fold it into the camera extrinsics,
_OPENSIM_TO_WORLD = {
    # rotation angles: x 90, y 90
    CheckerboardPlacement.GROUND: np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float),
    # rotation angles: y 90, z 180
    CheckerboardPlacement.BACK_WALL: np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float),
}


@dataclass
class Camera:
    intrinsicMat: list
    rotation: list
    translation: list
    imageSize: list

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            intrinsicMat=data['intrinsicMat'],
            rotation=data['rotation'],
            translation=data['translation'],
            imageSize=data['imageSize']
        )

    def correct_extrinsics(self, checkerboard_placement: CheckerboardPlacement):
        """Fold the OpenSim-ground -> OpenCap-world rotation into the extrinsics """
        R = _OPENSIM_TO_WORLD[checkerboard_placement]
        self.rotation = np.asarray(self.rotation, dtype=float) @ R
        return

    @classmethod
    def from_pickle(
            cls,
            pickle_path: str,
            checkerboard_placement: CheckerboardPlacement,
    ) -> 'Camera':
        with open(pickle_path, 'rb') as fh:
            cal = pickle.load(fh)
        camera = Camera.from_dict(cal)
        camera.correct_extrinsics(checkerboard_placement)
        return camera

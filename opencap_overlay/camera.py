import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CheckerboardPlacement(Enum):
    GROUND = 'ground'
    BACK_WALL = 'backWall'


# OpenCap rotates triangulated keypoints into OpenSim's frame (Y up) by fixed axis
# rotations set by the checkerboard placement, then runs IK. To project the model
# back onto video we invert that rotation and fold it into the camera extrinsics
_OPENSIM_TO_WORLD = {
    # rotation angles: x 90, y 90
    CheckerboardPlacement.GROUND: np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float),
    # rotation angles: y 90, z 180
    CheckerboardPlacement.BACK_WALL: np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float),
}


@dataclass
class Camera:
    intrinsicMat: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    imageSize: np.ndarray

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            intrinsicMat=np.array(data['intrinsicMat']).reshape(3, 3),
            rotation=np.array(data['rotation']).reshape(3, 3),
            translation=np.array(data['translation']).ravel() / 1000.0,  # mm -> m
            imageSize=np.array(data['imageSize']).ravel()
        )

    def correct_extrinsics(self, checkerboard_placement: CheckerboardPlacement):
        """Fold the OpenSim-ground -> OpenCap-world rotation into the extrinsics """
        R = _OPENSIM_TO_WORLD[checkerboard_placement]
        self.rotation = self.rotation @ R
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

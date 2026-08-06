from dataclasses import dataclass
import pickle


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

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, 'rb') as fh:
            cal = pickle.load(fh)
        return Camera.from_dict(cal)

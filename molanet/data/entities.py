import os
from enum import Enum

import numpy as np
from bson.binary import Binary
from numpy.core.records import ndarray
from typing import Dict, Any, List

# TODO: This is probably the least idiomatic python code ever
from molanet.data.molanetdir import MolanetDir


def entity(cls, prefix=""):
    def create_dict(self) -> Dict[str, Any]:
        return {name[len(prefix):]: sanitise(value) for name, value in vars(self).items() if name.startswith(prefix)}

    def sanitise(value):
        if isinstance(value, Enum):
            return value.name
        elif isinstance(value, List):
            return [create_dict(item) for item in value]
        else:
            return value

    cls.dict = create_dict
    return cls


class SkillLevel(Enum):
    NOVICE = 0
    EXPERT = 1


class Diagnosis(Enum):
    UNKNOWN = 0
    MALIGNANT = 1
    BENIGN = 2


class UseCase(Enum):
    UNSPECIFIED = 0
    TRAINING = 1
    TEST = 2
    VALIDATION = 3
    IGNORE = 4


@entity
class Segmentation(object):
    def __init__(
            self,
            segmentation_id: str,
            mask: Binary,
            skill_level: SkillLevel,
            dimensions: (int, int) = None,
            use_case: UseCase = UseCase.UNSPECIFIED):
        self.segmentation_id = segmentation_id
        self.mask = mask
        self.skill_level = skill_level
        self.dimensions = dimensions
        self.use_case = use_case

    def save_original_segmentation(self, image: bytes, uuid: str, dir: MolanetDir):
        with open(os.path.join(dir.segmentations, uuid + self.segmentation_id + '.jpg'), 'wb') as f:
            f.write(image)

    def save_np_segmentation(self, np_image: ndarray, uuid: str, dir: MolanetDir):
        np.save(os.path.join(dir.segmentations_numpy, uuid + self.segmentation_id), np_image)

    def load_np_segmentation(self, uuid: str, dir: MolanetDir):
        return np.load(os.path.join(dir.segmentations_numpy, uuid + self.segmentation_id))


@entity
class MoleSample(object):
    def __init__(
            self,
            uuid: str,
            data_source: str,
            data_set: str,
            source_id: str,
            name: str,
            dimensions: (int, int),
            diagnosis: Diagnosis,
            image: Binary,
            segmentations: [Segmentation],
            use_case: UseCase = UseCase.UNSPECIFIED):
        self.uuid = uuid
        self.data_source = data_source
        self.data_set = data_set
        self.source_id = source_id
        self.name = name
        self.dimensions = dimensions
        self.diagnosis = diagnosis
        self.use_case = use_case
        self.image = image
        self.segmentations = segmentations

    def save_original_jpg(self, image: bytes, dir: MolanetDir):
        with open(os.path.join(dir.images, self.uuid + '.jpg'), 'wb') as f:
            f.write(image)

    def save_np_image(self, np_image: ndarray, dir: MolanetDir):
        np.save(os.path.join(dir.images_numpy, self.uuid), np_image)

    def load_np_image(self, dir: MolanetDir):
        return np.load(os.path.join(dir.images_numpy, self.uuid))

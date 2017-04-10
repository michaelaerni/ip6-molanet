from enum import Enum

from numpy.core.records import ndarray
from typing import Dict, Any, List

# TODO: This is probably the least idiomatic python code ever


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
            mask: ndarray,
            skill_level: SkillLevel,
            dimensions: (int, int) = None,
            gridfs_id=None,
            use_case: UseCase = UseCase.UNSPECIFIED):
        self.segmentation_id = segmentation_id
        self.mask = mask
        self.skill_level = skill_level
        self.dimensions = dimensions
        self.use_case = use_case
        self.gridfs_id = gridfs_id


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
            image: ndarray,
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
        self.gridfs_id = None

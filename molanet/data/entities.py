from enum import Enum

from bson.binary import Binary
from typing import Dict, Any

# TODO: This is probably the least idiomatic python code ever


def entity(cls, prefix=""):
    def fun(self) -> Dict[str, Any]:
        return {name[len(prefix):]: value for name, value in vars(self).items() if name.startswith(prefix)}

    cls.dict = fun
    return cls


class Diagnosis(Enum):
    UNKNOWN = 0
    MALIGNANT = 1
    BENIGN = 2


class SkillLevel(Enum):
    NOVICE = 0
    EXPERT = 1


class UseCase(Enum):
    UNSPECIFIED = 0
    TRAINING = 1
    TEST = 2
    VALIDATION = 3,
    IGNORE = 4


@entity
class Segmentation(object):

    def __init__(
            self,
            id: str,
            mask: Binary,
            skill_level: SkillLevel,
            dimensions: (int, int) = None,
            use_case: UseCase = UseCase.UNSPECIFIED):
        self.id = id
        self.mask = mask,
        self.skill_level = skill_level,
        self.dimensions = dimensions,
        self.use_case = use_case


@entity
class MoleSample(object):

    def __init__(
            self,
            uuid: str,
            data_source: str,
            data_set: str,
            source_id: str,
            set_id: str,
            dimensions: (int, int),
            diagnosis: Diagnosis,
            image: Binary,
            segmentations: [Segmentation],
            use_case: UseCase = UseCase.UNSPECIFIED,):
        self.uuid = uuid
        self.data_source = data_source
        self.data_set = data_set
        self.source_id = source_id
        self.set_id = set_id
        self.dimensions = dimensions
        self.diagnosis = diagnosis
        self.use_case = use_case
        self.image = image
        self.segmentations = segmentations

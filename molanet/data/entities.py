from enum import Enum

from numpy import ndarray


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


class Segmentation(object):
    def __init__(
            self,
            source_id: str,
            mask: ndarray,
            skill_level: SkillLevel,
            dimensions: (int, int),
            use_case: UseCase = UseCase.UNSPECIFIED):
        self.source_id = source_id
        self.mask = mask
        self.skill_level = skill_level
        self.dimensions = dimensions
        self.use_case = use_case


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

from enum import Enum
from typing import Union

from numpy import ndarray


class Diagnosis(Enum):
    UNKNOWN = 0
    NEVUS = 1
    MELANOMA = 2
    SEBORRHEIC_KERATOSIS = 3


class Segmentation(object):
    def __init__(
            self,
            source_id: str,
            mask: ndarray,
            dimensions: (int, int)):
        self.source_id = source_id
        self.mask = mask
        self.dimensions = dimensions


class MoleSample(object):
    def __init__(
            self,
            uuid: str,
            data_source: str,
            data_set: str,
            source_id: str,
            name: str,
            dimensions: (int, int),
            diagnosis: Union[Diagnosis, str],
            image: ndarray,
            segmentations: [Segmentation]):
        self.uuid = uuid
        self.data_source = data_source
        self.data_set = data_set
        self.source_id = source_id
        self.name = name
        self.dimensions = dimensions
        self.diagnosis = diagnosis
        self.image = image
        self.segmentations = segmentations

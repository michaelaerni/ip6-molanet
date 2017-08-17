from enum import Enum
from typing import Union

from numpy import ndarray


class Diagnosis(Enum):
    """
    List of known diagnoses.
    There might be other diagnoses present which are not contained in this list.
    """
    UNKNOWN = 0
    NEVUS = 1
    MELANOMA = 2
    SEBORRHEIC_KERATOSIS = 3


class Segmentation(object):
    """
    Entity representing a single lesion segmentation.
    """
    def __init__(
            self,
            source_id: str,
            mask: ndarray,
            dimensions: (int, int)):
        """
        Creates a new segmentation.
        :param source_id: Id of this segmentation in its source
        :param mask: Mask as a two-dimensional array
        :param dimensions: Dimensions of this segmentation as a (height, width) tuple
        """
        self.source_id = source_id
        self.mask = mask
        self.dimensions = dimensions


class MoleSample(object):
    """
    Entity representing a single lesion sample.
    """
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
        """
        Creates a new lesion sample.
        :param uuid: Generated uuid of this sample
        :param data_source: Data source this sample stems from
        :param data_set: Data set which contained this sample in its source
        :param source_id: Id of this sample in its source
        :param name: Friendly name of this sample from its source
        :param dimensions: Dimensions of this sample as a (height, width) tuple
        :param diagnosis: Diagnosis of this sample, either of type Diagnosis if well known or a string otherwise
        :param image: Lesion image as a two-dimensional array
        :param segmentations: List of segmentation entities related to this sample
        """
        self.uuid = uuid
        self.data_source = data_source
        self.data_set = data_set
        self.source_id = source_id
        self.name = name
        self.dimensions = dimensions
        self.diagnosis = diagnosis
        self.image = image
        self.segmentations = segmentations

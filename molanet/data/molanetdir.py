import os

import numpy as np
from numpy.core.records import ndarray

from molanet.data.entities import MoleSample, Segmentation


class MolanetDir(object):
    def __init__(
            self,
            root_path: str):
        self.root_path = root_path
        self.images = os.path.join(root_path, 'images')
        self.segmentations = os.path.join(root_path, 'segmentations')
        self.images_numpy = os.path.join(root_path, 'images_numpy')
        self.segmentations_numpy = os.path.join(root_path, 'segmentations_numpy')

        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if not os.path.exists(self.images):
            os.makedirs(self.images)
        if not os.path.exists(self.segmentations):
            os.makedirs(self.segmentations)
        if not os.path.exists(self.images_numpy):
            os.makedirs(self.images_numpy)
        if not os.path.exists(self.segmentations_numpy):
            os.makedirs(self.segmentations_numpy)

    def save_original_jpg(self, image: bytes, sample: MoleSample):
        with open(os.path.join(self.images, sample.uuid + '.jpg'), 'wb') as f:
            f.write(image)

    def save_np_image(self, np_image: ndarray, sample: MoleSample):
        np.save(os.path.join(self.images_numpy, sample.uuid), np_image)

    def load_np_image(self, sample: MoleSample):
        return np.load(os.path.join(self.images_numpy, sample.uuid))

    def save_original_segmentation(self, image: bytes, uuid: str, sample: MoleSample):
        with open(os.path.join(self.segmentations, uuid + sample.segmentation_id + '.jpg'), 'wb') as f:
            f.write(image)

    def save_np_segmentation(self, np_image: ndarray, sample: MoleSample, segmentation: Segmentation):
        np.save(os.path.join(self.segmentations_numpy, sample.uuid + segmentation.segmentation_id), np_image)

    def load_np_segmentation(self, uuid: str, sample: MoleSample, segmentation: Segmentation):
        return np.load(os.path.join(self.segmentations_numpy, sample.uuid + segmentation.segmentation_id))

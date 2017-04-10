import os


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

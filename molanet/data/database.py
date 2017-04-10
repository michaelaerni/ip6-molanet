import numpy as np
from bson.objectid import ObjectId
from gridfs import GridFS
from pymongo import MongoClient

from molanet.data.entities import MoleSample


class MongoConnection(object):
    def __init__(self, url: str = "mongodb://localhost:27017/", db_name: str = "molanet", gridfs_name="molanetfs",
                 username: str = None, password: str = None):
        self.client = MongoClient(url)
        self.database = self.client[db_name]
        self.gridfs = GridFS(self.client[gridfs_name])

        if username is not None and password is not None:
            self.database.authenticate(username, password)
            self.client[gridfs_name].authenticate(username, password)

    def insert(self, sample: MoleSample) -> ObjectId:
        def apply_id(sample: MoleSample):
            sample_dict = sample.dict()
            sample_dict['_id'] = sample.uuid
            return sample_dict

        sample.gridfs_id = self.gridfs.put(np.ndarray.tobytes(sample.image),
                                           _id=sample.uuid,
                                           name=sample.name)
        sample.image = None
        for idx, segmentation in enumerate(sample.segmentations):
            segmentation.gridfs_id = self.gridfs.put(np.ndarray.tobytes(segmentation.mask),
                                                     _id=sample.uuid + str(idx),
                                                     name=sample.name,
                                                     segmentation_id=segmentation.segmentation_id)
            segmentation.mask = None
        return self.database.samples.insert(apply_id(sample))

    def clear_data(self, data_source: str, data_set: str = None) -> int:
        filter_dict = {"data_source": data_source}
        if data_set is not None:
            filter_dict["data_set"] = data_set

        return self.database.samples.delete_many(filter_dict).deleted_count

from bson.objectid import ObjectId
from pymongo import MongoClient

from molanet.data.entities import MoleSample


class MongoConnection(object):

    def __init__(self, url: str = "mongodb://localhost:27017/", db_name: str = "molanet", username: str = None, password: str = None):
        self.client = MongoClient(url)
        self.database = self.client[db_name]

        if username is not None and password is not None:
            self.database.authenticate(username, password)

    def load_data_set(self, data: [MoleSample]) -> [ObjectId]:
        def apply_id(sample: MoleSample):
            sample_dict = sample.dict()
            sample_dict['_id'] = sample.uuid
            return sample_dict

        dict_list = [apply_id(sample) for sample in data]
        return self.database.samples.insert_many(dict_list).inserted_ids

    def clear_data(self, data_source: str, data_set: str = None) -> int:
        filter_dict = {"data_source": data_source}
        if data_set is not None:
            filter_dict["data_set"] = data_set

        return self.database.samples.delete_many(filter_dict).deleted_count

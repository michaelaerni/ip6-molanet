from bson.objectid import ObjectId
from pymongo import MongoClient

from molanet.data.entities import MoleSample


class MongoConnection(object):
    def __init__(self, user: str, pwd: str, url: str = "mongodb://localhost:27017/", db_name: str = "molanet"):
        self.client = MongoClient(url)
        self.database = self.client[db_name]
        self.database.authenticate(user, pwd, mechanism='SCRAM-SHA-1')

    def load_data_set(self, data: [MoleSample]) -> [ObjectId]:
        def dict_with_id(sample: MoleSample):
            dict = sample.dict()
            dict['_id'] = sample.uuid
            return dict

        dict_list = [dict_with_id(sample) for sample in data]
        return self.database.samples.insert_many(dict_list)

    def clear_data(self, data_source: str, data_set: str = None) -> int:
        filter_dict = {"data_source": data_source}
        if data_set is not None:
            filter_dict["data_set"] = data_set

        return self.database.samples.delete_many(filter_dict).deleted_count

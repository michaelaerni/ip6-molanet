import psycopg2
from typing import Dict

from molanet.data.entities import MoleSample


class DatabaseConnection(object):
    def __init__(self, host: str, database: str, port: int = 5432, username: str = None, password: str = None):
        self._connection_params = {
            "dbname": database,
            "host": host,
            "port": port,
            "user": username,
            "password": password
        }

    def __enter__(self):
        self._connection = psycopg2.connect(**self._connection_params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.close()

    def insert(self, sample: MoleSample) -> str:
        query = "INSERT INTO mole_samples (uuid, data_source, data_set, source_id, name, height, width, diagnosis, use_case, image) " \
                "VALUES (%(uuid)s, %(data_source)s, %(data_set)s, %(source_id)s, %(name)s, %(height)s, %(width)s, %(diagnosis)s, %(use_case)s, %(image)s)"

        with self._connection.cursor() as cur:
            cur.execute(query, self._sample_to_dict(sample))

        return sample.uuid

    def clear_data(self, data_source: str) -> int:
        query = "DELETE FROM mole_samples WHERE data_source = %(data_source)s"

        with self._connection.cursor() as cur:
            cur.execute(query, {"data_source": data_source})
            return cur.rowcount  # Number of deleted rows

    @staticmethod
    def _sample_to_dict(sample: MoleSample) -> Dict:
        return {
            "uuid": sample.uuid,
            "data_source": sample.data_source,
            "data_set": sample.data_set,
            "source_id": sample.source_id,
            "name": sample.name,
            "height": sample.dimensions[0],
            "width": sample.dimensions[1],
            "diagnosis": sample.diagnosis.name,
            "use_case": sample.use_case.name,
            "image": sample.image.tobytes(order="C")
        }

import psycopg2
import numpy as np
from typing import Dict, Iterable

from molanet.data.entities import *


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

    def insert(self, sample: MoleSample) -> None:
        sample_query = "INSERT INTO mole_samples (uuid, data_source, data_set, source_id, name, height, width, diagnosis, use_case, image) " \
            "VALUES (%(uuid)s, %(data_source)s, %(data_set)s, %(source_id)s, %(name)s, %(height)s, %(width)s, %(diagnosis)s, %(use_case)s, %(image)s)"

        segmentation_query = "INSERT INTO segmentations (mole_sample_uuid, source_id, height, width, skill_level, use_case, mask) " \
            "VALUES (%(mole_sample_uuid)s, %(source_id)s, %(height)s, %(width)s, %(skill_level)s, %(use_case)s, %(mask)s)"

        with self._connection.cursor() as cur:
            # Insert sample
            cur.execute(sample_query, self.sample_to_dict(sample))

            # Insert segmentations
            for segmentation in sample.segmentations:
                cur.execute(segmentation_query, self.segmentation_to_dict(sample.uuid, segmentation))

        self._connection.commit()

    def clear_data(self, data_source: str) -> int:
        query = "DELETE FROM mole_samples WHERE data_source = %(data_source)s"

        with self._connection.cursor() as cur:
            # Segmentations are delete cascade, should be deleted too
            cur.execute(query, {"data_source": data_source})
            self._connection.commit()
            return cur.rowcount  # Number of deleted rows

    def get_samples(self, offset=0, batch_size=20) -> Iterable[MoleSample]:
        if offset < 0:
            raise ValueError("offset must be >= 0")

        sample_query = """SELECT uuid, data_source, data_set, source_id, name, height, width, diagnosis, use_case, image
        FROM mole_samples LIMIT %(batch_size)s OFFSET %(offset_count)s"""

        with self._connection.cursor() as cur:
            while True:
                cur.execute(sample_query, {"offset_count": offset, "batch_size": batch_size})

                if cur.rowcount <= 0:
                    break

                offset += cur.rowcount

                for sample_record in cur:
                    uuid = sample_record[0]
                    segmentations = list(self.get_segmentations_for_sample(uuid))
                    dimensions = (int(sample_record[5]), int(sample_record[6]))
                    yield MoleSample(
                        uuid,
                        data_source=sample_record[1],
                        data_set=sample_record[2],
                        source_id=sample_record[3],
                        name=sample_record[4],
                        dimensions=dimensions,
                        diagnosis=Diagnosis[sample_record[7]],
                        use_case=UseCase[sample_record[8]],
                        image=np.frombuffer(sample_record[9], dtype=np.uint8).reshape([dimensions[0], dimensions[1], 3]),
                        segmentations=segmentations
                    )

    def get_segmentations_for_sample(self, sample_uuid: str) -> Iterable[Segmentation]:
        query = """SELECT source_id, height, width, skill_level, use_case, mask
        FROM segmentations
        WHERE mole_sample_uuid = %(sample_uuid)s"""

        with self._connection.cursor() as cur:
            cur.execute(query, {"sample_uuid": sample_uuid})

            for segmentation_record in cur:
                dimensions = (int(segmentation_record[1]), int(segmentation_record[2]))
                yield Segmentation(
                    source_id=segmentation_record[0],
                    dimensions=dimensions,
                    skill_level=SkillLevel[segmentation_record[3]],
                    use_case=UseCase[segmentation_record[4]],
                    mask=np.frombuffer(segmentation_record[5], dtype=np.uint8).reshape([dimensions[0], dimensions[1], 1])
                )

    @staticmethod
    def sample_to_dict(sample: MoleSample, include_image: bool = True) -> Dict:
        result = {
            "uuid": sample.uuid,
            "data_source": sample.data_source,
            "data_set": sample.data_set,
            "source_id": sample.source_id,
            "name": sample.name,
            "height": sample.dimensions[0],
            "width": sample.dimensions[1],
            "diagnosis": sample.diagnosis.name,
            "use_case": sample.use_case.name
        }

        if include_image:
            result["image"] = sample.image.tobytes(order="C")

        return result

    @staticmethod
    def segmentation_to_dict(sample_uuid, segmentation: Segmentation, include_image: bool = True) -> Dict:
        result = {
            "mole_sample_uuid": sample_uuid,
            "source_id": segmentation.source_id,
            "height": segmentation.dimensions[0],
            "width": segmentation.dimensions[1],
            "skill_level": segmentation.skill_level.name,
            "use_case": segmentation.use_case.name
        }

        if include_image:
            result["mask"] = segmentation.mask.tobytes(order="C")

        return result

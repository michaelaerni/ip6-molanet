from typing import Dict, Iterable, Tuple, List

import numpy as np
import psycopg2

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
        sample_query = "INSERT INTO mole_samples (uuid, data_source, data_set, source_id, name, height, width, diagnosis, image) " \
            "VALUES (%(uuid)s, %(data_source)s, %(data_set)s, %(source_id)s, %(name)s, %(height)s, %(width)s, %(diagnosis)s, %(image)s)"

        segmentation_query = "INSERT INTO segmentations (mole_sample_uuid, source_id, height, width, mask) " \
            "VALUES (%(mole_sample_uuid)s, %(source_id)s, %(height)s, %(width)s, %(mask)s)"

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

    def get_data_set_samples(self, data_set: str)\
            -> Iterable[Tuple[MoleSample, List[Segmentation]]]:
        sample_uuid_query = """SELECT DISTINCT mole_sample_uuid FROM set_entries WHERE set_name = %(set_name)s"""
        sample_query = """SELECT uuid, data_source, data_set, source_id, name, height, width, diagnosis, image
                          FROM mole_samples WHERE uuid = %(uuid)s"""
        segmentations_query = """SELECT segmentations.source_id, segmentations.height, segmentations.width, segmentations.mask
                                 FROM segmentations
                                 JOIN set_entries ON (segmentations.source_id = set_entries.segmentation_source_id)
                                 WHERE set_entries.mole_sample_uuid = %(mole_sample_uuid)s AND set_name = %(set_name)s"""

        with self._connection.cursor() as cur:
            # Load all sample ids in the set first so we load them only once
            cur.execute(sample_uuid_query, {"set_name": data_set})
            uuids = [row[0] for row in cur.fetchall()]

            # Now load every sample and then all corresponding masks in the same data set
            for uuid in uuids:
                # Load all segmentations for this sample in the current set
                cur.execute(segmentations_query, {"mole_sample_uuid": uuid, "set_name": data_set})
                segmentations = [self._record_to_segmentation(segmentation_record) for segmentation_record in cur]

                # Finally load the actual sample and yield a new result
                cur.execute(sample_query, {"uuid": uuid})
                sample = self._record_to_sample(cur.fetchone(), segmentations)

                yield sample, segmentations

    def get_data_set_ids(self, data_set: str) -> List[Tuple[str, str]]:
        query = """SELECT mole_sample_uuid, segmentation_source_id FROM set_entries WHERE set_name = %(set_name)s"""

        with self._connection.cursor() as cur:
            cur.execute(query, {"set_name": data_set})
            return cur.fetchall()

    def get_samples(self, offset: int = 0, batch_size: int = 20) -> Iterable[MoleSample]:
        if offset < 0:
            raise ValueError("offset must be >= 0")

        sample_query = """SELECT uuid, data_source, data_set, source_id, name, height, width, diagnosis, image
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
                    yield self._record_to_sample(sample_record, segmentations)

    def get_segmentations_for_sample(self, sample_uuid: str) -> Iterable[Segmentation]:
        query = """SELECT source_id, height, width, mask
        FROM segmentations
        WHERE mole_sample_uuid = %(sample_uuid)s"""

        with self._connection.cursor() as cur:
            cur.execute(query, {"sample_uuid": sample_uuid})

            for segmentation_record in cur:
                yield self._record_to_segmentation(segmentation_record)

    @staticmethod
    def _record_to_segmentation(segmentation_record: Tuple) -> Segmentation:
        dimensions = (int(segmentation_record[1]), int(segmentation_record[2]))
        return Segmentation(
            source_id=segmentation_record[0],
            dimensions=dimensions,
            mask=np.frombuffer(segmentation_record[3], dtype=np.uint8).reshape([dimensions[0], dimensions[1], 1])
        )

    @staticmethod
    def _record_to_sample(sample_record: Tuple, segmentations: List[Segmentation]) -> MoleSample:
        uuid = sample_record[0]
        dimensions = (int(sample_record[5]), int(sample_record[6]))
        raw_diagnosis = str(sample_record[7])
        diagnosis = Diagnosis[raw_diagnosis] if raw_diagnosis in Diagnosis.__members__ else raw_diagnosis
        return MoleSample(
            uuid,
            data_source=sample_record[1],
            data_set=sample_record[2],
            source_id=sample_record[3],
            name=sample_record[4],
            dimensions=dimensions,
            diagnosis=diagnosis,
            image=np.frombuffer(sample_record[8], dtype=np.uint8).reshape([dimensions[0], dimensions[1], 3]),
            segmentations=segmentations
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
            "diagnosis": sample.diagnosis.name if isinstance(sample.diagnosis, Enum) else str(sample.diagnosis)
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
            "width": segmentation.dimensions[1]
        }

        if include_image:
            result["mask"] = segmentation.mask.tobytes(order="C")

        return result

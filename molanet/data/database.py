from typing import Dict, Iterable, Tuple, List

import numpy as np
import psycopg2

from molanet.data.entities import *


class DatabaseConnection(object):
    """
    Connection to a sample data base.
    """
    def __init__(self, host: str, database: str, port: int = 5432, username: str = None, password: str = None):
        """
        Creates a new database connection and connects to the specified database.

        :param host: Database host
        :param database: Name of the database containing the samples
        :param port: Port of the database system
        :param username: Username used to connect
        :param password: Password used to connect
        """

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
        """
        Insert the given sample into the database
        :param sample: Sample to insert
        """

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
        """
        Delete all lesion images and segmentation masks from a single data source.
        :param data_source: Data source whose samples should be deleted
        :return: Number of deleted samples
        """
        query = "DELETE FROM mole_samples WHERE data_source = %(data_source)s"

        with self._connection.cursor() as cur:
            # Segmentations are delete cascade, should be deleted too
            cur.execute(query, {"data_source": data_source})
            self._connection.commit()
            return cur.rowcount  # Number of deleted rows

    def create_data_set_entries(self, data_set_name: str, entries: List[Tuple[str, str]]):
        """
        Create entries for the given list in the given data set.
        :param data_set_name: Name of the target data set
        :param entries: List of (lesion uuid, segmentation id) pairs to insert into the data set
        """
        query = """INSERT INTO set_entries (mole_sample_uuid, segmentation_source_id, set_name)
                   VALUES (%(mole_sample_uuid)s, %(segmentation_source_id)s, %(set_name)s)"""

        with self._connection.cursor() as cur:
            for mole_sample_uuid, segmentation_source_id in entries:
                # Insert entry
                cur.execute(query, {
                    "mole_sample_uuid": mole_sample_uuid,
                    "segmentation_source_id": segmentation_source_id,
                    "set_name": data_set_name
                })

        self._connection.commit()

    def clear_data_set_entries(self, data_set_name: str) -> int:
        """
        Clear all entries in the given data set. This method only deletes references and no actual lesions or
        segmentation masks.
        :param data_set_name: Name of the data set to delete
        :return: Number of deleted data set entries
        """
        query = """DELETE FROM set_entries WHERE set_name = %(set_name)s"""

        with self._connection.cursor() as cur:
            cur.execute(query, {"set_name": data_set_name})
            self._connection.commit()
            return cur.rowcount  # Number of deleted rows

    def get_data_set_samples(self, data_set: str)\
            -> Iterable[Tuple[MoleSample, List[Segmentation]]]:
        """
        Get all samples in the given data set.

        This method loads its results lazy.
        :param data_set: Data set whose samples should be retrieved
        :return: List of (lesion, mask) tuples
        """
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
        """
        Get all ids in the given data set.
        :param data_set: Data set whose ids should be retrieved
        :return: List of (lesion uuid, segmentation id) tuples
        """
        query = """SELECT mole_sample_uuid, segmentation_source_id FROM set_entries WHERE set_name = %(set_name)s"""

        with self._connection.cursor() as cur:
            cur.execute(query, {"set_name": data_set})
            return cur.fetchall()

    def get_samples(self, offset: int = 0, batch_size: int = 20) -> Iterable[MoleSample]:
        """
        Get a list of all samples, starting at the given offset.
        This method performs lazy loading.

        :param offset: Offset index (inclusive) from which samples should be retrieved.
        :param batch_size: Number of samples to fetch each database request. The higher, the better the network
        is utilized. Too high number result in decreased database performance.
        :return: Lazy loaded list of all samples
        """
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
        """
        Get all segmentations for a given sample.
        This method performs lazy loading.
        :param sample_uuid: Uuid of the sample whose segmentation should be retrieved
        :return: Lazy loaded list of all segmentation for the given sample
        """
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
        """
        Serialize a sample into a dictionary.
        :param sample: Sample to serialize
        :param include_image: If True, the lesion image is included in the results, if False it is not
        :return: Serialized sample
        """
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
        """
        Serialize a segmentation into a dictionary.
        :param sample_uuid: Uuid of the sample the segmentation belongs to
        :param segmentation: Segmentation to serialize
        :param include_image: If True, the segmentation mask is included in the results, if False it is not
        :return: Serialized segmentation
        """
        result = {
            "mole_sample_uuid": sample_uuid,
            "source_id": segmentation.source_id,
            "height": segmentation.dimensions[0],
            "width": segmentation.dimensions[1]
        }

        if include_image:
            result["mask"] = segmentation.mask.tobytes(order="C")

        return result

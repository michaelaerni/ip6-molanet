import argparse
import io

import numpy as np
import requests
from PIL import Image
from typing import Iterable

from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample, Diagnosis, Segmentation, SkillLevel


def parse_diagnosis(raw_diagnosis: str) -> Diagnosis:
    return {
        "benign": Diagnosis.BENIGN,
        "malignant": Diagnosis.MALIGNANT
    }.get(raw_diagnosis, Diagnosis.UNKNOWN)  # indeterminate/* or unknown are stored as unknown (not relevant)


def parse_skill_level(raw_level : str) -> SkillLevel:
    return {
        "expert": SkillLevel.EXPERT,
        "novice": SkillLevel.NOVICE
    }.get(raw_level, SkillLevel.UNKNOWN)


class IsicLoader(object):
    def __init__(self, base_url="https://isic-archive.com/api/v1/", data_source="isic", batch_size=100):
        self._base_url = base_url
        self._data_source = data_source
        self._batch_size = batch_size

    def _perform_request(self, relative_url, parameters=None):
        return requests.get(self._base_url + relative_url, params=parameters)

    def fetch_image_ids(self, offset):
        return self._perform_request("image", {"limit": self._batch_size, "offset": offset, "sort": "_id"}).json()

    def fetch_segmentation_ids(self, image_id, offset):
        return self._perform_request("segmentation", {"imageId": image_id, "limit": self._batch_size, "offset": offset, "sort": "_id"}).json()

    def fetch_image_matrix(self, image_id) -> np.ndarray:
        raw_image = Image.open(io.BytesIO(self._perform_request(f"image/{image_id}/download").content))
        return np.array(raw_image)

    def fetch_segmentation_matrix(self, segmentation_id) -> np.ndarray:
        raw_image = Image.open(io.BytesIO(self._perform_request(f"segmentation/{segmentation_id}/mask").content))
        return np.array(raw_image) // 255  # Normalize to [0, 1]

    def fetch_image_metadata(self, image_id):
        return self._perform_request(f"image/{image_id}").json()

    def fetch_segmentations(self, image_id) -> Iterable[Segmentation]:
        batch_offset = 0
        current_batch = self.fetch_segmentation_ids(image_id, batch_offset)
        while len(current_batch) > 0:
            for current_segmentation in current_batch:
                segmentation_id = current_segmentation["_id"]

                if current_segmentation["failed"]:
                    print(f"Failed segmentation {segmentation_id} for image {image_id}, skipped")
                else:
                    mask = self.fetch_segmentation_matrix(segmentation_id)
                    yield Segmentation(
                        segmentation_id,
                        mask,
                        parse_skill_level(current_segmentation["skill"]),
                        (mask.shape[0], mask.shape[1])
                    )

            # Load next batch, might be empty at the end
            batch_offset += len(current_batch)
            current_batch = self.fetch_segmentation_ids(image_id, batch_offset)

    def create_uuid(self, image_id):
        return f"{self._data_source}_{image_id}"

    def load_samples(self, offset=0) -> Iterable[MoleSample]:
        batch_offset = offset

        current_batch = self.fetch_image_ids(batch_offset)
        while len(current_batch) > 0:
            # Process all images in current batch and yield results
            for current_image in current_batch:
                source_id = current_image["_id"]
                friendly_name = current_image["name"]

                # Load required data
                image_values = self.fetch_image_matrix(source_id)
                metadata = self.fetch_image_metadata(source_id)
                segmentations = list(self.fetch_segmentations(source_id))

                # Create new sample
                yield MoleSample(
                    self.create_uuid(source_id),
                    self._data_source,
                    metadata["dataset"]["name"],
                    source_id,
                    friendly_name,
                    (int(metadata["meta"]["acquisition"]["pixelsY"]), int(metadata["meta"]["acquisition"]["pixelsX"])),
                    parse_diagnosis(metadata["meta"]["clinical"]["benign_malignant"]),
                    image_values,
                    segmentations
                )

            # Load next batch, might be empty at the end
            batch_offset += len(current_batch)
            current_batch = self.fetch_image_ids(batch_offset)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Load ISIC images and segmentations into a database")

    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--api-url", type=str, default="https://isic-archive.com/api/v1/", help="Base API url of the data source")
    parser.add_argument("--data-source-name", type=str, default="isic", help="Name of the data source stored in the database")
    parser.add_argument("--fetch-batch-size", type=int, default=200, help="Size of batches which are fetched from the data source")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    data_source = args.data_source_name
    loader = IsicLoader(args.api_url, data_source, args.fetch_batch_size)

    with DatabaseConnection(args.database_host, args.database, username=args.database_username, password=args.database_password) as db:
        if args.offset == 0:
            removed_count = db.clear_data(data_source)
            print(f"Cleared data set, deleted {removed_count} rows")
        else:
            print(f"Starting at offset {args.offset}, existing data will not be cleared")

        sample_count = args.offset
        for sample in loader.load_samples(offset=args.offset):
            sample_count += 1

            db.insert(sample)

            print(f"[{sample_count}]: Saved sample {sample.uuid} from data set {sample.data_set}")

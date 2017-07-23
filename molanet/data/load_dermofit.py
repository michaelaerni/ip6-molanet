import argparse
import uuid
from os import path
from typing import Iterable, Union

import numpy as np
from PIL import Image

from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample, Diagnosis, Segmentation


def parse_diagnosis(raw_diagnosis: str) -> Union[Diagnosis, str]:
    return {
        "AK": "ACTINIC_KERATOSIS",
        "BCC": "BASAL_CELL_CARCINOMA",
        "DF": "DERMATOFIBROMA",
        "IEC": "INTRAEPITHELIAL_CARCINOMA",
        "MEL": Diagnosis.MELANOMA,
        "ML": Diagnosis.NEVUS,
        "PYO": "PYOGENIC_GRANULOMA",
        "SCC": "SQUAMOUS_CELL_CARCINOMA",
        "SK": Diagnosis.SEBORRHEIC_KERATOSIS,
        "VASC": "HAEMANGIOMA"
    }.get(raw_diagnosis, Diagnosis.UNKNOWN)  # Fallback case should never trigger


class DermofitLoader(object):
    def __init__(self, lesion_list_path: str, root_directory: str):
        self._lesion_list_path = lesion_list_path
        self._root_directory = root_directory

    def create_uuid(self) -> str:
        return str(uuid.uuid4())

    def load_samples(self, offset: int) -> Iterable[MoleSample]:
        sample_metadata = [line.strip().split() for line in open(self._lesion_list_path)]

        # Last line of meta data file is empty, ignore it
        if len(sample_metadata[-1]) == 0:
            sample_metadata = sample_metadata[:-1]

        for number, name, lesion in sample_metadata[offset:]:
            yield self.load_sample(number, name, lesion)

    def load_sample(self, number: int, dermofit_id: str, raw_diagnosis: str):
        sample_directory = path.join(self._root_directory, raw_diagnosis.lower(), dermofit_id)
        image_path = path.join(sample_directory, f"{dermofit_id}.png")
        mask_path = path.join(sample_directory, f"{dermofit_id}mask.png")

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        assert image.shape[0] == mask.shape[0]
        assert image.shape[1] == mask.shape[1]
        dimensions = (image.shape[0], image.shape[1])

        # dermofit has exactly one mask per image
        segmentation = Segmentation(f"{dermofit_id}", mask, dimensions)

        sample = MoleSample(
            uuid=self.create_uuid(),
            data_source="dermofit",
            data_set=raw_diagnosis,
            source_id=dermofit_id,
            name=str(number), # Name is the internal dermofit number
            dimensions=dimensions,
            diagnosis=parse_diagnosis(raw_diagnosis.upper()),
            image=image,
            segmentations=[segmentation]
        )
        return sample


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Load Dermofit images and segmentations into a database")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--data-source-name", type=str, default="dermofit",
                        help="Name of the data source stored in the database")
    parser.add_argument("--lesion-list", type=str, default="lesionlist.txt",
                        help="Name of the file containing the lesion meta data")
    parser.add_argument("--root-directory", type=str, default=None,
                        help="Root directory containing the lesion list file and extracted images")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    lesion_list = path.join(args.root_directory, args.lesion_list)
    loader = DermofitLoader(lesion_list, args.root_directory)

    with DatabaseConnection(
            args.database_host, args.database, username=args.database_username, password=args.database_password) as db:

        # TODO: Allow update instead of re-insert (keep uuid)

        if args.offset == 0:
            removed_count = db.clear_data(data_source="dermofit")
            print(f"Cleared data set, deleted {removed_count} rows")
        else:
            print(f"Starting at offset {args.offset}, existing data will not be cleared")

        sample_count = args.offset
        for sample in loader.load_samples(offset=args.offset):
            sample_count += 1

            db.insert(sample)

            print(f"[{sample_count}]: Saved sample {sample.uuid} from data set {sample.data_set}")

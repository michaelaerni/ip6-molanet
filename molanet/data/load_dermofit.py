import argparse
from os import path
from typing import Iterable

import numpy as np
from PIL import Image

from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample, Diagnosis, Segmentation, SkillLevel


class DermofitLoader(object):
    def __init__(self, lesionlist: str, lesiondir: str):
        self.lesionlist = lesionlist
        self.lesiondir = lesiondir

    def create_uuid(self, source_id: str) -> str:
        return source_id

    def parse_diagnosis(self, lesion: str) -> Diagnosis:
        return {
            "ak": Diagnosis.MALIGNANT,  # Actinic Keratosis
            "bcc": Diagnosis.MALIGNANT,  # Basal Cell Carcinoma
            "df": Diagnosis.BENIGN,  # Dermatofibroma
            "iec": Diagnosis.MALIGNANT,  # Intraepithelial Carcinoma  (unsure)
            "mel": Diagnosis.MALIGNANT,  # Malignant Melanoma
            "ml": Diagnosis.BENIGN,  # Melanocytic Nevus
            "pyo": Diagnosis.BENIGN,  # Pyogenic Granuloma
            "scc": Diagnosis.MALIGNANT,  # Squamous Cell Carcinoma
            "sk": Diagnosis.BENIGN,  # Seborrhoeic Keratosis
            "vasc": Diagnosis.BENIGN  # Haemangioma
        }.get(lesion, Diagnosis.UNKNOWN)  # indeterminate/* or unknown are stored as unknown (not relevant)

    def load_sample(self, number: int, name: str, lesion: str):
        dir = path.join(self.lesiondir, lesion.lower(), name)
        imagename = path.join(dir, f"{name}.png")
        maskname = path.join(dir, f"{name}mask.png")
        image = np.array(Image.open(imagename))
        mask = np.array(Image.open(maskname))

        source_id = f"{lesion}_{name}"

        assert image.shape[0] == mask.shape[0]
        assert image.shape[1] == mask.shape[1]

        # dermofit has exactly one mask per image
        segmentation = Segmentation(f"{source_id}mask", mask, SkillLevel.UNKNOWN, (mask.shape[0], mask.shape[1]))

        sample = MoleSample(
            uuid=self.create_uuid(source_id),
            data_source="dermofit",
            data_set=lesion,
            source_id=source_id,
            name=f"dermofit_{number}",
            diagnosis=self.parse_diagnosis(lesion.lower()),
            dimensions=(image.shape[0], image.shape[1]),
            image=image,
            segmentations=[segmentation]
        )
        return sample

    def load_samples(self, offset: int) -> Iterable[MoleSample]:
        splitlines = [line.strip().split() for line in open(self.lesionlist)]

        for number, name, lesion in splitlines[offset:-1]:
            yield self.load_sample(number, name, lesion)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Load ISIC images and segmentations into a database")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--data-source-name", type=str, default="dermofit",
                        help="Name of the data source stored in the database")
    parser.add_argument("--lesionlist", type=str, default="lesionlist.txt",
                        help="name of the file with the lesiondescriptions")
    parser.add_argument("--lesiondir", type=str, default=None,
                        help=" Path to the folder containing the lesion folders and lesionlist ")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    lesionlist = path.join(args.lesiondir, args.lesionlist)
    loader = DermofitLoader(lesionlist, args.lesiondir)

    with DatabaseConnection(args.database_host, args.database, username=args.database_username,
                            password=args.database_password) as db:
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

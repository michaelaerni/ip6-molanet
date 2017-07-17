import argparse
import csv
from typing import Iterable

import molanet.data.data_analysis as data
from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample


def compute_features(sample: MoleSample):
    hair = data.contains_hair(sample.image)
    plaster = data.contains_plaster(sample.image)
    mean, median, stddev = data.calculate_pointwise_statistics(sample.image)

    features = {}
    features['uuid'] = sample.uuid
    features['hair'] = hair
    features['plaster'] = plaster
    features['mean'] = mean
    features['median'] = median
    features['stddev'] = stddev

    for segmentation in sample.segmentations:
        relative_size, absolute_size = data.calculate_mole_sizes(segmentation.mask)
        features['seg_id'] = segmentation.source_id
        features['rel_size'] = relative_size
        features['abs_size'] = absolute_size
        yield features


def extract_features(samples: Iterable[MoleSample],
                     features_csv_path: str,
                     discarded_csv_path: str,
                     fieldnames: [str], delimiter=';'):
    with open(features_csv_path, 'w', newline='') as csvfile:
        with open(discarded_csv_path, 'w', newline='') as discarded_csv:
            writer = csv.DictWriter(csvfile,
                                    delimiter=delimiter,
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL,
                                    fieldnames=fieldnames)
            writer.writeheader()
            discardedWriter = csv.writer(discarded_csv, delimiter=' ',
                                         quotechar='|', quoting=csv.QUOTE_MINIMAL)

            count = offset
            for sample in samples:
                count += 1
                if len(sample.segmentations) == 0:
                    discardedWriter.writerow(sample.uuid)

                for features in compute_features(sample):
                    writer.writerow(features)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Dump all images including metadata from a database into a directory")

    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    offset = 0
    fieldnames = ['uuid', 'seg_id', 'hair', 'plaster', 'mean', 'median', 'stddev', 'rel_size', 'abs_size']
    delimiter = ";"
    features_csv_path = 'features.csv'
    discarded_csv_path = 'discarded.csv'

    with DatabaseConnection(
            args.database_host,
            args.database,
            username=args.database_username,
            password=args.database_password) as db:

        extract_features(db.get_samples(offset=offset),
                         features_csv_path,
                         discarded_csv_path,
                         fieldnames,
                         delimiter=delimiter)

    with open('features.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            print(row['uuid'])

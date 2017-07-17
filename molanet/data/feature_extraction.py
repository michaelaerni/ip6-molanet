import argparse
import csv
import os
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt

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
                     fieldnames: [str],
                     delimiter=';',
                     offset=0):
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
                    continue

                try:
                    for features in compute_features(sample):
                        writer.writerow(features)
                    if (count % 300 == 0):
                        print(f"{count}: computed features for {count-offset} samples")
                except ValueError:
                    print(f"{count}: ValueError in sample {sample.uuid}")
                    pass
                except ZeroDivisionError:
                    print(f"{count}: ZeroDivisionError in sample {sample.uuid}")
                    pass


def count_features(features_csv_path: str, bins=7):
    with open(features_csv_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        print(reader.fieldnames)
        nrows = 0
        plasters = 0
        hair = 0
        mean = []
        stddev = []
        median = []
        rel_size = []
        abs_size = []

        for row in reader:
            nrows += 1
            plasters += 1 if row['plaster'] == 'True' else 0
            hair += 1 if row['hair'] == 'True' else 0
            mean.append(row['mean'])
            stddev.append(row['stddev'])
            median.append(row['median'])
            rel_size.append(row['rel_size'])
            abs_size.append(row['abs_size'])

            print([row[name] for name in fieldnames])

        print(reader.fieldnames)
        print(f"parsed {nrows} rows")
        print(plasters, plasters / nrows)
        print(hair, hair / nrows)

        def plot_hist(list, bins=bins):
            hist, bins = np.histogram(np.array(median, dtype=np.float32), bins=bins)
            width = np.diff(bins)
            center = (bins[:-1] + bins[1:]) / 2

            fig, ax = plt.subplots()
            ax.bar(center, hist, align='center', width=width)
            ax.set_xticks(bins)

        plot_hist(stddev)
        plot_hist(median)
        plot_hist(mean)
        plot_hist(rel_size)
        plot_hist(abs_size)

        plt.show()

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Extract useful features")

    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--features-path", type=str, default="", help="path folder for features csv file")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    offset = 606
    fieldnames = ['uuid', 'seg_id', 'hair', 'plaster', 'mean', 'median', 'stddev', 'rel_size', 'abs_size']
    delimiter = ";"
    features_csv_path = f"features_{offset}.csv"
    discarded_csv_path = f"discarded_{offset}.csv"
    features_csv_path = os.path.join(args.features_path, features_csv_path)
    discarded_csv_path = os.path.join(args.features_path, discarded_csv_path)

    with DatabaseConnection(
            args.database_host,
            args.database,
            username=args.database_username,
            password=args.database_password) as db:
        extract_features(db.get_samples(offset=offset),
                         features_csv_path,
                         discarded_csv_path,
                         fieldnames,
                         delimiter=delimiter,
                         offset=offset)

    bins = 'auto'
    bins = 7
    count_features(features_csv_path, bins)

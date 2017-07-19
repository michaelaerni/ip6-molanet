import csv
from optparse import Option

import numpy as np
from matplotlib import pyplot as plt


def count_features(features_csv_path: str, fieldnames: [str], delimiter: str, bins: Option(int, str) = 7):
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

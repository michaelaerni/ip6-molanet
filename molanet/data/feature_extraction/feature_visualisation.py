import csv
import os
import random
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def read_data(features_csv_path: str, fieldnames: [str], delimiter: str):
    seg_uuid_map = {}
    feature_map = {}

    # plasters are returned as 0,1 (instead of bool)
    # hair is returned as 0,1

    with open(features_csv_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        print(reader.fieldnames)
        nrows = 0
        plasters = []
        hair = []
        mean = []
        stddev = []
        median = []
        rel_size = []
        abs_size = []
        uuids = []
        seg_uuids = []

        for row in reader:
            nrows += 1

            seg_uuids.append(row['seg_id'])
            uuids.append(row['uuid'])
            plasters.append(1 if row['plaster'] == 'True' else 0)
            hair.append(1 if row['hair'] == 'True' else 0)
            mean.append(row['mean'])
            stddev.append(row['stddev'])
            median.append(row['median'])
            rel_size.append(row['rel_size'])
            abs_size.append(row['abs_size'])

        print(f"parsed {nrows} rows")
        return uuids, seg_uuids, plasters, hair, mean, stddev, median, rel_size, abs_size


def create_split_indices(size: int) -> np.ndarray:
    return np.random.permutation(size)


def count_features(data, logdir: str, bins=7, setname: str = "unpsecified"):
    uuids, seg_uuids, plasters, hair, mean, stddev, median, rel_size, abs_size = data
    size = len(seg_uuids)

    def plot_hist(fig, ax, list, bins=bins, title: str = ""):
        hist, bins = np.histogram(np.array(list, dtype=np.float32), bins=bins)
        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        ax.bar(center, hist, align='center', width=width)
        ax.set_xticks(bins)
        ax.set_title(title)

    nplasters = sum(plasters)
    nhair = sum(hair)
    print(f"{setname}_plasters: {nplasters/size*100}% or {nplasters}")
    print(f"{setname}_hair: {nhair/size*100}% or {nhair}")

    fig, ax = plt.subplots(5)
    fig.canvas.set_window_title(setname)
    fig.suptitle(setname, fontsize=12)

    plot_hist(fig, ax[0], stddev, title=setname + '_stddev')
    plot_hist(fig, ax[1], median, title=setname + '_median')
    plot_hist(fig, ax[2], mean, title=setname + '_mean')
    plot_hist(fig, ax[3], rel_size, title=setname + '_rel_size')
    plot_hist(fig, ax[4], abs_size, title=setname + '_abs_size')

    fig.savefig(os.path.join(logdir, f"{setname}_set.png"), bbox_inches='tight', papertype="a4")

    with open(os.path.join(logdir, f"{setname}_set.csv"), 'w') as f:
        f.write(f"uuid;segmentation_id;")
        for i in range(0, len(uuids)):
            s = f"{uuids[i]};{seg_uuids[i]};"
            f.write(s + "\n")


if __name__ == '__main__':
    path = r"C:\Users\pdcwi\Downloads\features.csv"
    fieldnames = ['uuid', 'seg_id', 'hair', 'plaster', 'mean', 'median', 'stddev', 'rel_size', 'abs_size']
    data = read_data(path, fieldnames, ";")
    uuids, seg_uuids, plasters, hair, mean, stddev, median, rel_size, abs_size = data

    seed = random.randrange(sys.maxsize)
    print(seed)
    rng = random.Random(seed)
    combined = list(zip(*data))
    rng.shuffle(combined)

    cv_set_size = int(len(uuids) * 0.15)
    test_set_size = int(len(uuids) * 0.2)

    print(f"training set size={len(uuids)-cv_set_size-test_set_size}")
    print(f"cv set size={cv_set_size}")
    print(f"test set size={test_set_size}")

    test_set = combined[0:test_set_size]
    cv_set = combined[test_set_size:test_set_size + cv_set_size]
    training_set = combined[test_set_size + cv_set_size:]

    now = datetime.now()
    subdirname = f"split_{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}"
    if not os.path.isdir(subdirname): os.mkdir(subdirname)

    count_features(zip(*training_set), subdirname, setname="training")
    count_features(zip(*cv_set), subdirname, setname="cv")
    count_features(zip(*test_set), subdirname, setname="test")

    plt.show()

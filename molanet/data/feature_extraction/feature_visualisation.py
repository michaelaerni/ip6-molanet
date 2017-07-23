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
        plaster = []
        hair = []
        mean = []
        stddev = []
        median = []
        rel_size = []
        abs_size = []
        uuids = []
        seg_uuids = []

        # indices for which the image with <uuid> has more than one segmentation
        multi_seg_indices = []

        lastuuid = ""
        for idx, row in enumerate(reader):
            nrows += 1
            uuid = row['uuid']

            if uuid == lastuuid:
                if len(multi_seg_indices) == 0 or not multi_seg_indices[-1] == idx - 1:
                    multi_seg_indices.append(idx - 1)
                multi_seg_indices.append(idx)

            seg_uuids.append(row['seg_id'])
            uuids.append(row['uuid'])
            plaster.append(1 if row['plaster'] == 'True' else 0)
            hair.append(1 if row['hair'] == 'True' else 0)
            mean.append(row['mean'])
            stddev.append(row['stddev'])
            median.append(row['median'])
            rel_size.append(row['rel_size'])
            abs_size.append(row['abs_size'])
            lastuuid = uuid

        print(f"parsed {nrows} rows")
        print(f"{len(multi_seg_indices)} masks which do not map to a distinct uuid")

        single_segmetation_mask = np.ones(len(uuids), np.bool)
        single_segmetation_mask[multi_seg_indices] = 0

        data_single_mask = (np.array(uuids)[single_segmetation_mask],
                            np.array(seg_uuids)[single_segmetation_mask],
                            np.array(hair, dtype=np.bool)[single_segmetation_mask],
                            np.array(plaster, dtype=np.bool)[single_segmetation_mask],
                            np.array(mean, np.float32)[single_segmetation_mask],
                            np.array(median, np.float32)[single_segmetation_mask],
                            np.array(stddev, np.float32)[single_segmetation_mask],
                            np.array(rel_size, np.float32)[single_segmetation_mask],
                            np.array(abs_size, np.float32)[single_segmetation_mask])

        data_multimask = (np.array(uuids)[multi_seg_indices],
                          np.array(seg_uuids)[multi_seg_indices],
                          np.array(hair, dtype=np.bool)[multi_seg_indices],
                          np.array(plaster, dtype=np.bool)[multi_seg_indices],
                          np.array(mean, np.float32)[multi_seg_indices],
                          np.array(median, np.float32)[multi_seg_indices],
                          np.array(stddev, np.float32)[multi_seg_indices],
                          np.array(rel_size, np.float32)[multi_seg_indices],
                          np.array(abs_size, np.float32)[multi_seg_indices])

        return data_single_mask, data_multimask


def plot_hist(fig, ax, list: np.ndarray, bins, title: str = ""):
    hist, bins = np.histogram(list, bins=bins, density=False)
    # print(np.min(hist[np.nonzero(hist)]),np.max(hist))
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    ax.set_title(title)


def count_features(data, logdir: str, bins=(5, 5, 5, 5, 5), setname: str = "unpsecified"):
    uuids, seg_uuids, hair, plaster, mean, median, stddev, rel_size, abs_size = data
    bins_mean, bins_stddev, bins_median, bins_bins_rel_size, bins_abs_size = bins

    size = len(seg_uuids)

    nplasters = sum(plaster)
    nhair = sum(hair)
    print(f"{setname}_plasters: {nplasters/size*100}% or {nplasters}")
    print(f"{setname}_hair: {nhair/size*100}% or {nhair}")

    fig, ax = plt.subplots(5)
    fig.canvas.set_window_title(setname)
    fig.suptitle(setname, fontsize=12)

    plot_hist(fig, ax[0], stddev, title=setname + '_stddev', bins=bins_stddev)
    plot_hist(fig, ax[1], median, title=setname + '_median', bins=bins_median)
    plot_hist(fig, ax[2], mean, title=setname + '_mean', bins=bins_mean)
    plot_hist(fig, ax[3], rel_size, title=setname + '_rel_size', bins=bins_bins_rel_size)
    plot_hist(fig, ax[4], abs_size, title=setname + '_abs_size', bins=bins_abs_size)

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
    data_single, data_multimask = data
    uuids, seg_uuids, hair, plaster, mean, median, stddev, rel_size, abs_size = data_single

    # bin_arg = 'doane'
    bin_arg = 5
    individual_bin_args = False

    # this could be automated for variable parameter sizes
    if type(bin_arg) == str or individual_bin_args:
        # this is REALLY slow
        _, bins_mean = np.histogram(np.array(mean, dtype=np.float32), bins=bin_arg)
        _, bins_stddev = np.histogram(np.array(stddev, dtype=np.float32), bins=bin_arg)
        _, bins_median = np.histogram(np.array(median, dtype=np.float32), bins=bin_arg)
        _, bins_bins_rel_size = np.histogram(np.array(rel_size, dtype=np.float32), bins=bin_arg)
        _, bins_abs_size = np.histogram(np.array(abs_size, dtype=np.float32), bins=bin_arg)
    else:
        bins_mean, bins_stddev, bins_median, bins_bins_rel_size, bins_abs_size = bin_arg, bin_arg, bin_arg, bin_arg, bin_arg

    bins = bins_mean, bins_stddev, bins_median, bins_bins_rel_size, bins_abs_size

    """
    TODO stratified cv set creation may be desirable (partitioning so that features are uniformely distributed in cv set)

    https://en.wikipedia.org/wiki/Stratified_sampling
    https://stats.stackexchange.com/questions/117643/why-use-stratified-cross-validation-why-does-this-not-damage-variance-related-b
    http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.186.8880

    "This is particularly useful if the responses are dichotomous 
    with an unbalanced representation of the two response values in the data."

    In our case creating an equal partitioning is HARD, not only because some stratums contain very vew example but also
    because a single stratus is defined by a multidimensional feature.
    A sensible approach is to ensure enough data of each stratum is included in the training phase so that the model can
    learn the peculiarities associated with
    """

    # shuffle all array simultaneously by zipping them and shuffling then uzipping later
    seed = random.randrange(sys.maxsize)
    # seed = 247079226373091163
    print(seed)
    rng = random.Random(seed)
    combined = list(zip(*data_single))
    rng.shuffle(combined)

    cv_set_size = int(len(uuids) * 0.15)
    test_set_size = int(len(uuids) * 0.2)
    training_set_size = len(uuids) - cv_set_size - test_set_size + data_multimask[0].size
    print(f"training set size={training_set_size}")
    print(f"cv set size={cv_set_size}")
    print(f"test set size={test_set_size}")

    test_set = combined[0:test_set_size]
    cv_set = combined[test_set_size:test_set_size + cv_set_size]
    training_set = combined[test_set_size + cv_set_size:]
    training_set += list(zip(*data_multimask))
    now = datetime.now()
    subdirname = f"split_{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}"
    if not os.path.isdir(subdirname): os.mkdir(subdirname)

    count_features(zip(*training_set), subdirname, setname="training", bins=bins)
    count_features(zip(*cv_set), subdirname, setname="cv", bins=bins)
    count_features(zip(*test_set), subdirname, setname="test", bins=bins)

    with open(os.path.join(subdirname, "log.txt"), 'w') as log:
        log.write(f"seed={seed}\n")
        log.write(f"training set size={training_set_size}\n")
        log.write(f"cv set size={cv_set_size}\n")
        log.write(f"test set size={test_set_size}\n")
        log.write(f"total size={training_set_size+cv_set_size+test_set_size}\n")

        log.write(f"\nbins\n{bins}")

    plt.show()

import csv
import os
import random
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from molanet.data.entities import Diagnosis


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
        diagnosiseseses = []

        # indices for which the image with <uuid> has more than one segmentation
        multi_seg_indices = []

        # only count rows with a diagnosis that we actually care about
        discarded_diagnosis = {}
        discard_diagnosis_indices = []

        lastuuid = ""
        diagnosis_names = [e.name for e in Diagnosis]

        for idx, row in enumerate(reader):
            nrows += 1
            uuid = row['uuid']
            diagnosis = row['diagnosis']

            if diagnosis not in diagnosis_names:
                if diagnosis in discarded_diagnosis:
                    discarded_diagnosis[diagnosis].append(uuid)
                else:
                    discarded_diagnosis[diagnosis] = [uuid]
                discard_diagnosis_indices.append(idx)

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
            diagnosiseseses.append(diagnosis)
            lastuuid = uuid

        print(f"parsed {nrows} rows")
        print(f"{len(multi_seg_indices)} masks which do not map to a distinct uuid")
        print(
            f"discarded {sum([len(list) for list in discarded_diagnosis])} masks because of diagnosis.\n{discarded_diagnosis}")

        mask_indices = np.unique(multi_seg_indices + discard_diagnosis_indices)

        single_segmetation_mask = np.ones(len(seg_uuids), np.bool)
        single_segmetation_mask[mask_indices] = 0

        multi_segmentation_mask = np.zeros(len(seg_uuids), np.bool)
        multi_segmentation_mask[multi_seg_indices] = 1
        multi_segmentation_mask[discard_diagnosis_indices] = 0

        data_single_mask = (np.array(uuids)[single_segmetation_mask],
                            np.array(seg_uuids)[single_segmetation_mask],
                            np.array(hair, dtype=np.uint8)[single_segmetation_mask],
                            np.array(plaster, dtype=np.uint8)[single_segmetation_mask],
                            np.array(mean, np.float32)[single_segmetation_mask],
                            np.array(median, np.float32)[single_segmetation_mask],
                            np.array(stddev, np.float32)[single_segmetation_mask],
                            np.array(rel_size, np.float32)[single_segmetation_mask],
                            np.array(abs_size, np.float32)[single_segmetation_mask])

        data_multimask = (np.array(uuids)[multi_segmentation_mask],
                          np.array(seg_uuids)[multi_segmentation_mask],
                          np.array(hair, dtype=np.uint8)[multi_segmentation_mask],
                          np.array(plaster, dtype=np.uint8)[multi_segmentation_mask],
                          np.array(mean, np.float32)[multi_segmentation_mask],
                          np.array(median, np.float32)[multi_segmentation_mask],
                          np.array(stddev, np.float32)[multi_segmentation_mask],
                          np.array(rel_size, np.float32)[multi_segmentation_mask],
                          np.array(abs_size, np.float32)[multi_segmentation_mask])

        return data_single_mask, data_multimask


def plot_hist(ax, hist, bins, title: str = ""):
    # print(np.min(hist[np.nonzero(hist)]),np.max(hist))
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    ax.set_title(title)


def count_features(data, featurenames: [str],
                   logdir: str,
                   bins,
                   setname: str = "unpsecified",
                   front_str_cols: int = 2,
                   plot: bool = False):
    if (len(data) == 0): return

    fig, ax = None, None
    if (plot):
        fig, ax = plt.subplots(len(data) - front_str_cols)
        fig.canvas.set_window_title(setname)
        fig.suptitle(setname, fontsize=12)

    for idx, feature in enumerate(data):
        if (idx < front_str_cols): continue
        hist, hist_bins = np.histogram(feature, bins=bins[idx - front_str_cols], density=False)

        # check histogram for missing values

        if (plot):
            plot_hist(ax[idx - front_str_cols], hist, hist_bins, title=f"{setname}_{featurenames[idx]}")

    if (plot):
        fig.savefig(os.path.join(logdir, f"{setname}_set.png"), bbox_inches='tight', papertype="a4")

    # assuming uuid and seg_id are in columns 1 and 2 respectively
    with open(os.path.join(logdir, f"{setname}_set.csv"), 'w') as f:
        f.write(f"uuid;segmentation_id;")
        for i in range(0, len(data[0])):
            s = f"{data[0][i]};{data[1][i]};"
            f.write(s + "\n")

if __name__ == '__main__':
    path = r"C:\Users\pdcwi\Downloads\features.csv"
    fieldnames = ['uuid', 'seg_id', 'hair', 'plaster', 'mean', 'median', 'stddev', 'rel_size', 'abs_size', 'diagnosis']
    data = read_data(path, fieldnames, ";")
    data_single, data_multimask = data
    do_plot = True
    log_directory = ''
    bin_arg = 'auto'
    test_set_rel_size = 0.2
    cv_set_rel_size = 0.15

    # bin_arg = 'doane'
    individual_bin_args = False
    bins = []
    # this could be automated for variable parameter sizes
    if type(bin_arg) == str or individual_bin_args:
        # precalculate hist bins on entire dataset
        for idx, feature in enumerate(data_single):
            if idx < 2: continue
            _, bin_calc = np.histogram(feature, bins=bin_arg)
            bins.append(bin_calc)
    else:
        bins = [bin_arg for _ in range(len(list(data_single)) - 2)]


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
    print(f"seed={seed}")
    rng = random.Random(seed)
    combined = list(zip(*data_single))
    rng.shuffle(combined)

    cv_set_size = int(len(data_single[0]) * cv_set_rel_size)
    test_set_size = int(len(data_single[0]) * test_set_rel_size)
    training_set_size = len(data_single[0]) - cv_set_size - test_set_size + data_multimask[0].size

    test_set = combined[0:test_set_size]
    cv_set = combined[test_set_size:test_set_size + cv_set_size]
    training_set = combined[test_set_size + cv_set_size:]
    training_set += list(zip(*data_multimask))

    print(f"training set size={len(training_set)}")
    print(f"cv set size={len(cv_set)}")
    print(f"test set size={len(test_set)}")

    now = datetime.now()
    subdirname = os.path.join(log_directory, f"split_{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}")
    if not os.path.isdir(subdirname): os.mkdir(subdirname)

    count_features(list(zip(*training_set)), fieldnames, subdirname, setname="training", bins=bins, plot=do_plot)
    count_features(list(zip(*cv_set)), fieldnames, subdirname, setname="cv", bins=bins, plot=do_plot)
    count_features(list(zip(*test_set)), fieldnames, subdirname, setname="test", bins=bins, plot=do_plot)

    with open(os.path.join(subdirname, "log.txt"), 'w') as log:
        log.write(f"seed={seed}\n")
        log.write(f"training set size={len(training_set)}\n")
        log.write(f"cv set size={len(cv_set)}\n")
        log.write(f"test set size={len(test_set)}\n")
        log.write(f"total size={len(training_set)+len(cv_set)+len(test_set)}\n")

        log.write(f"\nbins\n{bins}")

    print('Done')

    if (do_plot):
        plt.show()

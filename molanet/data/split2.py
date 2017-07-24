import csv
import os
import random
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from molanet.data.entities import Diagnosis


def read_data(features_csv_path: str, delimiter: str):
    data = np.loadtxt(features_csv_path,
                      usecols=range(2, 9),
                      skiprows=1,
                      delimiter=delimiter,
                      converters={2: lambda s: 1 if s.decode('utf-8') == 'True' else 0,
                                  3: lambda s: 1 if s.decode('utf-8') == 'True' else 0},
                      dtype=None)

    with open(features_csv_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        print(reader.fieldnames)
        nrows = 0
        uuids = []
        seg_uuids = []
        diagnosiseseses = []

        # indices for which the image with <uuid> has more than one segmentation
        multi_seg_indices = []

        lastuuid = ""
        for idx, row in enumerate(reader):
            nrows += 1
            uuid = row['uuid']
            seg_id = row['seg_id']
            diagnosis = row['diagnosis']

            if uuid == lastuuid:
                if len(multi_seg_indices) == 0 or not multi_seg_indices[-1] == idx - 1:
                    multi_seg_indices.append(idx - 1)
                multi_seg_indices.append(idx)
            seg_uuids.append(seg_id)
            uuids.append(uuid)
            diagnosiseseses.append(diagnosis)
            lastuuid = uuid

        print(f"parsed {nrows} rows")
        print(f"{len(multi_seg_indices)} masks which do not map to a distinct uuid")

    data_text = np.column_stack([np.array(uuids, dtype=np.str_),
                                 np.array(seg_uuids, dtype=np.str_),
                                 np.array(diagnosiseseses, dtype=np.str_)])

    single_seg_mask = np.ones(len(uuids), np.bool_)
    single_seg_mask[multi_seg_indices] = 0

    return data_text, data, single_seg_mask


def random_indices(length: int, seed: int = None) -> (np.ndarray, int):
    if seed is None:
        seed = random.randrange(2194967295)

    indices = np.arange(0, length)
    np.random.seed(seed)
    np.random.shuffle(indices)
    print(f"using seed={seed}")
    return indices, seed


def diagnosis_mask(diagnosis: np.ndarray) -> np.ndarray:
    names = [e.name for e in Diagnosis]
    names.remove(Diagnosis.UNKNOWN.name)

    indices_good_diagnosis = []
    for idx, diag in enumerate(diagnosis):
        if diag in names:
            indices_good_diagnosis.append(idx)

    print(f"found {len(indices_good_diagnosis)} masks with diagnosis of either {names}, "
          f"{diagnosis.size - len(indices_good_diagnosis)} with a different one.")

    mask = np.zeros(diagnosis.size, np.bool_)
    mask[indices_good_diagnosis] = 1
    return mask


def calc_set_size(set_sizes_rel, length):
    offset = 0
    set_off_size = []
    for set_size_rel in set_sizes_rel:
        set_size = int(length * set_size_rel)
        # start,end
        set_off_size.append((offset, offset + set_size))
        offset += set_size

    return set_off_size


def calculate_split(nrows, set_sizes_rel, single_seg_mask, seed: int = None):
    indices, seed = random_indices(nrows, seed)

    indices_single_mask = indices[single_seg_mask]
    # create some lists with indices into the original data
    set_sizes = calc_set_size(set_sizes_rel, nrows)
    set_indices = [indices_single_mask[start:end] for start, end in set_sizes]
    _, lastend = set_sizes[-1]
    training_set_indices = np.append(indices_single_mask[lastend:], indices[single_seg_mask == 0])

    return training_set_indices, set_indices


def plot_hist(ax, hist, bins, title: str = ""):
    # print(np.min(hist[np.nonzero(hist)]),np.max(hist))
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    ax.set_title(title)


def plot_set(set: np.ndarray, name: str = '', bins=None, bin_default: Union[str, int] = 'fd'):
    print()
    print(name)

    if bins is None:
        bins = [bin_default for _ in range(set.shape[1])]
        bins[0] = 2
        bins[1] = 2

    fig, ax = plt.subplots(set.shape[1])
    fig.suptitle(name)
    fig.canvas.set_window_title(name)

    for i in range(data.shape[1]):
        # fig, ax = plt.subplots()
        hist, hist_bins = np.histogram(set[:, i], bins[i])
        print(np.min(hist))
        plot_hist(ax[i], hist, hist_bins, title=fieldnames[i + 2])


def custom_bins():
    bins = [None] * 7
    bins[0] = [0, 0.5, 1.0]
    bins[1] = [0, 0.5, 1.0]
    bins[2] = [60.0, 107.0, 220, 255.9]
    bins[3] = [58.0, 130.0, 210.0, 255]
    bins[4] = [8, 22, 46, 66, 84, 110]
    bins[5] = [0.0, 0.0203, 0.04, 0.16, 0.4, 0.75, 0.99]
    bins[6] = [0, 0.012, 0.025, 0.1, 0.5, 1.0, 1.5, 2.0, 2.761]
    bins[6] = [e * 1e7 for e in bins[6]]

    return bins

if __name__ == '__main__':
    # arguments
    path = r"C:\Users\pdcwi\Downloads\features.csv"
    set_sizes_rel = [0.15, 0.2]
    do_plot = True
    fieldnames = ['uuid', 'seg_id', 'hair', 'plaster', 'mean', 'median', 'stddev', 'rel_size', 'abs_size', 'diagnosis']
    log_directory = ''
    csv_delimiter = ";"

    # input reading
    data_text, data, single_seg_mask = read_data(path, csv_delimiter)
    diagnosis_accepted_mask = diagnosis_mask(data_text[:, 2])

    # filter out irrelevant diagonses
    data = data[diagnosis_accepted_mask]
    data_text = data_text[diagnosis_accepted_mask]
    single_seg_mask = single_seg_mask[diagnosis_accepted_mask]
    nrows = data.shape[0]

    bins = custom_bins()
    for i in range(data.shape[1]):
        hist, _ = np.histogram(data[:, i], bins[i])
        print(np.min(hist))


    def ensure_set_min_counts(bins, set_count, data, min=[50, 50, 25, 50, 50, 50, 36]):

        def check_feature_done(i: int) -> bool:
            for b in bin_counts[i]:
                if not b >= min[i]:
                    return False
            return True

        used_indices = []
        set_indices = []
        for set_idx in range(set_count):
            bin_counts = [[0] * (len(bin) - 1) for bin in bins]
            set_indices.append([])

            for i in range(data.shape[1]):
                if check_feature_done(i): continue
                for row, feature in enumerate(data[:, i]):
                    if row in used_indices: continue

                    # check to which hist bin the feature belongs
                    if check_feature_done(i): break
                    for k in range(len(bins[i]) - 1):
                        if bin_counts[i][k] < min[i] and feature >= bins[i][k] and feature <= bins[i][k + 1]:
                            # add whole row to indices and update other bin_counts accordingly
                            set_indices[set_idx].append(row)
                            used_indices.append(row)

                            for l, otherfeatues in enumerate(data[row, :]):
                                for j in range(len(bins[l]) - 1):
                                    if otherfeatues >= bins[l][j] and otherfeatues <= bins[l][j + 1]:
                                        bin_counts[l][j] += 1

        print([len(seti) for seti in set_indices], set_indices)
        return set_indices


    training_set_indices, set_indices = calculate_split(nrows, set_sizes_rel, single_seg_mask, seed=None)

    training_set_indices, _ = ensure_set_min_counts(custom_bins(), 2, data)
    training_set = data[training_set_indices]
    training_set_text = data_text[training_set_indices]
    sets = [data[i] for i in set_indices]
    sets_text = [data_text[i] for i in set_indices]

    # statistics
    sum = training_set.shape[0]
    print(f"trainig set size: {training_set.shape[0]} or about {training_set.shape[0]/nrows*100}% ")
    for idx, set in enumerate(sets):
        sum += set[:, 0].size
        print(f"set_{idx} size: {set.shape[0]} or about {set.shape[0]/nrows*100}%")
    # assert sum == nrows

    print("saving")
    np.savetxt(os.path.join(log_directory, 'training_set.csv'), training_set_text, delimiter=csv_delimiter, fmt="%s")
    for idx, set in enumerate(sets_text):
        np.savetxt(os.path.join(log_directory, f'set_{idx}_{set.shape[0]/nrows*100}.csv'),
                   set,
                   delimiter=csv_delimiter,
                   fmt="%s")

    if do_plot:
        print('creating plots')
        plot_set(data, 'whole data set')

        bins = custom_bins()
        plot_set(training_set, 'training_set', bins)
        for idx, set in enumerate(sets):
            plot_set(set, f'set_{idx}_({set.shape[0]/nrows*100}%)', bins)

        plt.show()

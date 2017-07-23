import argparse
import os
import shutil
from os import path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType, TFRecordWriter, TFRecordOptions

from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample, Segmentation


class NoSegmentationsError(Exception):
    pass


class RecordSaver(object):
    def __init__(self, root_directory: str, data_set: str,
                 compression_type: TFRecordCompressionType = TFRecordCompressionType.ZLIB,
                 rescale: bool = False):

        self._root_directory = os.path.abspath(root_directory)
        self._target_directory = os.path.abspath(os.path.join(root_directory, data_set))
        self._data_set = data_set
        self._options = TFRecordOptions(compression_type)
        self._rescale = rescale

    def clear_existing_records(self):
        shutil.rmtree(self._target_directory, ignore_errors=True)

    def write_sample(self, sample: MoleSample, segmentation: Segmentation):
        # Generate target path and create necessary directories
        target_directory, target_path = self._generate_sample_path(sample, segmentation)
        os.makedirs(target_directory, exist_ok=True)

        transformed_image, transformed_segmentation = self._transform_sample(sample, segmentation)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': self._float_list(transformed_image),
            'segmentation': self._float_list(transformed_segmentation)}))

        with TFRecordWriter(target_path, self._options) as writer:
            writer.write(example.SerializeToString())

    def write_meta_data(self, sample_uuids: List[Tuple[str, str]]):
        os.makedirs(self._root_directory)
        meta_path = os.path.join(self._root_directory, f"{self._data_set}.txt")
        with open(meta_path, mode="w") as f:
            f.write(os.linesep.join(
                (f"{sample_uuid}_{segmentation_id}" for sample_uuid, segmentation_id in sample_uuids)))

    def _generate_sample_path(self, sample: MoleSample, segmentation: Segmentation) -> Tuple[str, str]:
        sample_directory = path.join(self._target_directory, sample.uuid[0:2])
        full_path = path.join(sample_directory, f"{sample.uuid}_{segmentation.source_id}.tfrecord")
        return sample_directory, full_path

    def _transform_sample(self, sample: MoleSample, segmentation: Segmentation) -> Tuple[np.ndarray, np.ndarray]:
        sample_data = sample.image
        segmentation_data = segmentation.mask

        # Rescale the images if necessary
        if self._rescale:
            # TODO: Target resolution is currently hardcoded, could parameterize
            sample_data = self._resize_image(sample_data, 512)
            segmentation_data = self._resize_segmentation(segmentation_data, 512)

        # Transform into tanh range
        sample_data = (sample_data.astype(np.float32) / 255.0 - 0.5) * 2.0  # [0, 255] -> [-1, 1]
        segmentation_data = (segmentation_data.astype(np.float32) - 0.5) * 2.0  # [0, 1] -> [-1, 1]

        return sample_data, segmentation_data

    @staticmethod
    def _resize_image(sample_image: np.ndarray, size: int) -> np.ndarray:
        assert sample_image.dtype == np.uint8
        image = Image.fromarray(sample_image, mode="RGB")

        image.thumbnail((size, size), Image.LANCZOS)
        padding = Image.new("RGB", (size, size), (0, 0, 0))  # Black
        padding.paste(image, ((padding.size[0] - image.size[0]) // 2, (padding.size[1] - image.size[1]) // 2))
        return np.asarray(padding, dtype=np.uint8)

    @staticmethod
    def _resize_segmentation(segmentation: np.ndarray, size: int) -> np.ndarray:
        assert segmentation.dtype == np.uint8
        image = Image.fromarray(np.squeeze(segmentation), mode="1")

        image.thumbnail((size, size), Image.NEAREST)
        padding = Image.new("1", (size, size), 0)  # Background
        padding.paste(image, ((padding.size[0] - image.size[0]) // 2, (padding.size[1] - image.size[1]) // 2))
        return np.asarray(padding, dtype=np.uint8).reshape((size, size, 1))

    @staticmethod
    def _float_list(array: np.ndarray):
        return tf.train.Feature(float_list=tf.train.FloatList(value=array.reshape([-1])))


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Save one or more data sets to disk")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--target-directory", type=str, default=None,
                        help="Base directory into which the data sets are exported")
    parser.add_argument("--rescale", action="store_true",
                        help="Rescale all images to 512x512 pixels. Non-matching aspect ratios are padded.")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only export the data set lists, no images. Image directories are not touched.")
    parser.add_argument("datasets", metavar="SET", type=str, nargs="+", help="One or more data sets to export")

    return parser


if __name__ == "__main__":

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    with DatabaseConnection(
            args.database_host, args.database, args.database_port,
            username=args.database_username, password=args.database_password) as db:

        # Export each data set individually
        for data_set in args.datasets:
            print(f"Saving data set {data_set}")
            saver = RecordSaver(args.target_directory, data_set, rescale=args.rescale)

            if not args.metadata_only:
                print(f"Removing old tfrecords...")
                saver.clear_existing_records()

                print(f"Loading tfrecords...")

                sample_count = 0

                for sample, segmentations in db.get_data_set_samples(data_set):
                    for segmentation in segmentations:
                        saver.write_sample(sample, segmentation)
                    print(f"[{sample_count}]: Saved {sample.uuid} with {len(segmentations)} segmentations")

                    sample_count += 1

            print(f"Saving meta data file...")
            saver.write_meta_data(db.get_data_set_ids(data_set))

        print("Done")

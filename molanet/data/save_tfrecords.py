import argparse
import os
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType, TFRecordWriter, TFRecordOptions

from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample


class RecordSaver(object):
    def __init__(self, rootdir: str, compressiontype: TFRecordCompressionType = TFRecordCompressionType.ZLIB):
        self.rootdir = rootdir
        self.options = TFRecordOptions(compressiontype)

    def get_filename(self, sample: MoleSample) -> str:
        return path.join(self.rootdir, str(sample.use_case.name).lower(), sample.uuid[0:2], f"{sample.uuid}.tfrecord")

    def get_or_make_filename(self, sample: MoleSample) -> str:
        use_dir = path.join(self.rootdir, str(sample.use_case.name).lower())
        if not path.isdir(use_dir): os.mkdir(use_dir)
        sub_dir = path.join(use_dir, sample.uuid[0:2])
        if not path.isdir(sub_dir): os.mkdir(sub_dir)
        return self.get_filename(sample)

    def _resize_image(self, image: np.ndarray, size: int):
        # TODO aspect ratio
        actual_image = Image.fromarray(image)
        image = actual_image.resize((size, size), Image.BICUBIC)
        return image

    def _resize_segmentation(self, segmentation: np.ndarray, size: int):
        # TODO aspect ratio
        actual_image = Image.fromarray(np.squeeze(segmentation))
        image = actual_image.resize((size, size), Image.NEAREST)
        return image

    def _image_feature(self, image: np.ndarray):
        bytes = np.array(image, dtype=np.float32).tobytes()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def writeSample(self, sample: MoleSample):
        file = self.get_or_make_filename(sample)

        image = self._resize_image(sample.image, 512)
        segmentation = self._resize_segmentation(sample.segmentations[0].mask, 512)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': self._image_feature(image),
            'segmentation': self._image_feature(segmentation)}))

        with TFRecordWriter(file, self.options) as writer:
            writer.write(example.SerializeToString())


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Dump molanet images into tfrecords")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")
    parser.add_argument("--basedir", type=str, default=None, help="directory to dump records into")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="how many records to dump. -1 for all starting at offset")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    with DatabaseConnection(args.database_host, args.database, username=args.database_username,
                            password=args.database_password) as db:

        saved_count = args.offset
        for sample in db.get_samples(args.offset, args.batch_size):
            saver = RecordSaver(args.basedir)
            saver.writeSample(sample)

            print(f"[{saved_count}]: saved {sample.uuid}")
            saved_count += 1

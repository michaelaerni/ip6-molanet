import argparse
import os
from os import path

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType, TFRecordWriter, TFRecordOptions

from molanet.data.database import DatabaseConnection
from molanet.data.entities import MoleSample


class RecordSaver(object):
    def __init__(self, rootdir: str, compressiontype: TFRecordCompressionType = TFRecordCompressionType.ZLIB):
        self.rootdir = rootdir
        self.comressiontype = compressiontype
        self.options = TFRecordOptions(compressiontype)

    def get_path(self, sample: MoleSample) -> str:
        print(sample.uuid)
        subfolder = sample.uuid[0:2]
        folder = path.join(self.rootdir, subfolder)
        return folder

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def writeSample(self, sample: MoleSample):
        folder = self.get_path(sample)
        if not path.isdir(folder):
            os.mkdir(folder)

        file = path.join(self.rootdir, folder, f"{sample.uuid}.tfrecord")
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': self._int64_feature(sample.dimensions[0]),
            'width': self._int64_feature(sample.dimensions[1]),
            'image_bytes': self._bytes_feature(sample.image.tobytes()),
            'mask_bytes': self._bytes_feature(sample.segmentations[0].mask.tobytes())}))

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
        for sample in db.get_samples(args.offset, args.batch_size):
            print(sample)
            saver = RecordSaver(args.basedir)
            saver.writeSample(sample)

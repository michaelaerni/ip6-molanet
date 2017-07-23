import argparse
import os
from enum import Enum
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType, TFRecordWriter, TFRecordOptions

from molanet.data.entities import MoleSample, UseCase


class Resize(Enum):
    SCALE = 0
    PADDED = 1

class RecordSaver(object):
    def __init__(self, rootdir: str, compressiontype: TFRecordCompressionType = TFRecordCompressionType.ZLIB,
                 log_saved_uuids=True, resize: Resize = Resize.SCALE):
        raise NotImplementedError("This needs to be rewritten to not dump all data sets and use the new database structure")
        self.rootdir = rootdir
        self.options = TFRecordOptions(compressiontype)
        self.log_saved_uuids = log_saved_uuids
        self.resize = resize

        for logfile in [self.get_logfile(UseCase.__getattr__(name)) for name in UseCase._member_names_]:
            try:
                os.remove(logfile)
            except OSError:
                pass

    def get_filename(self, sample: MoleSample) -> str:
        return path.join(self.rootdir, str(sample.use_case.name).lower(), sample.uuid[0:2], f"{sample.uuid}.tfrecord")

    def get_or_make_filename(self, sample: MoleSample) -> str:
        use_dir = path.join(self.rootdir, str(sample.use_case.name).lower())
        if not path.isdir(use_dir): os.mkdir(use_dir)
        sub_dir = path.join(use_dir, sample.uuid[0:2])
        if not path.isdir(sub_dir): os.mkdir(sub_dir)
        return self.get_filename(sample)

    def _resize_image(self, image: np.ndarray, size: int):
        image = Image.fromarray(image)
        if self.resize == Resize.SCALE:
            return image.resize((size, size), Image.ANTIALIAS)
        elif self.resize == Resize.PADDED:
            image.thumbnail((size, size), Image.ANTIALIAS)
            padding = Image.new('RGB',
                                (size, size),
                                (0, 0, 0))  # Black
            padding.paste(image, ((padding.size[0] - image.size[0]) // 2, (padding.size[1] - image.size[1]) // 2))
            return padding
        else:
            raise ValueError()

    def _resize_segmentation(self, segmentation: np.ndarray, size: int):
        image = Image.fromarray(np.squeeze(segmentation))
        if self.resize == Resize.SCALE:
            return image.resize((size, size), Image.NEAREST)
        elif self.resize == Resize.PADDED:
            image.thumbnail((size, size), Image.NEAREST)
            padding = image.resize((512, 512))
            padding.paste(0, (0, 0, 512, 512))  # fill black
            padding.paste(image, ((padding.size[0] - image.size[0]) // 2, (padding.size[1] - image.size[1]) // 2))
            return padding
        else:
            raise ValueError()

    def _float_list(self, array: np.ndarray):
        return tf.train.Feature(float_list=tf.train.FloatList(value=array.reshape([-1])))

    def _image_feature(self, image: Image):
        # Normalize into tanh range
        image_array = np.asarray(image, dtype=np.float32)
        image_array = (image_array / 255.0 - 0.5) * 2.0
        return self._float_list(image_array)

    def _segmentation_feature(self, segmentation: Image):
        # Normalize into tanh range
        segmentation_array = np.asarray(segmentation, dtype=np.float32)
        segmentation_array = (segmentation_array - 0.5) * 2.0  # Already in range [0, 1]
        return self._float_list(segmentation_array)

    def get_logfile(self, usecase: UseCase):
        return path.join(self.rootdir, f"{usecase.name}.txt")

    def log_saved(self, sample: MoleSample):
        with open(self.get_logfile(sample.use_case), "a") as logfile:
            logfile.write(f"{sample.uuid}\n")

    def write_sample(self, sample: MoleSample) -> bool:
        if len(sample.segmentations) == 0: return False

        file = self.get_or_make_filename(sample)

        image = self._resize_image(sample.image, 512)
        segmentation = self._resize_segmentation(sample.segmentations[0].mask, 512)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': self._image_feature(image),
            'segmentation': self._segmentation_feature(segmentation)}))

        with TFRecordWriter(file, self.options) as writer:
            writer.write(example.SerializeToString())
        if self.log_saved_uuids:
            self.log_saved(sample)
        return True


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
    parser.add_argument("--resize", type=int, default=0,
                        help="0: keep aspect ratio and pad with 0's. 1: resize by scaling (ignore aspect ratio)")

    return parser


if __name__ == "__main__":
    raise NotImplementedError("This needs to be rewritten to not dump all data sets and use the new database structure")

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    resize = Resize.PADDED if args.resize == 0 else Resize.SCALE
    saver = RecordSaver(args.basedir, resize=resize)

    with DatabaseConnection(args.database_host, args.database, username=args.database_username,
                            password=args.database_password) as db:
        sample_count = args.offset
        for sample in db.get_samples(args.offset, args.batch_size):
            if saver.write_sample(sample):
                print(f"[{sample_count}]: saved {sample.uuid}")
            else:
                print(f"[{sample_count}]: failed to save {sample.uuid}")
            sample_count += 1

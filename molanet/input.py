import io
import os
from typing import Tuple, List

import tensorflow as tf


def _read_record(
        input_producer,
        size,
        compression_type):

    reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(compression_type))
    _, serialized_example = reader.read(input_producer, name="read_record")

    raw_features = tf.parse_single_example(serialized_example, features={
        "image": tf.FixedLenFeature([size, size, 3], tf.float32),
        "segmentation": tf.FixedLenFeature([size, size, 1], tf.float32)
    }, name="parse_record")

    image = raw_features["image"]
    segmentation = raw_features["segmentation"]

    return image, segmentation


def _load_paths(input_directory: str, data_set_name: str) -> List[str]:
    input_directory = os.path.abspath(input_directory)
    sample_directory = os.path.join(input_directory, data_set_name)

    # Read uuids into memory and create paths
    with io.open(os.path.join(input_directory, f"{data_set_name}.txt")) as f:
        return [os.path.join(sample_directory, uuid[:2], uuid).strip("\n") + ".tfrecord" for uuid in f.readlines()]


def create_fixed_input_pipeline(
        input_directory: str,
        data_set_name: str,
        batch_size: int,
        epochs: int,
        image_size: int,
        seed: int = None,
        compression_type: tf.python_io.TFRecordCompressionType = tf.python_io.TFRecordCompressionType.ZLIB,
        min_after_dequeue: int = 100,
        thread_count: int = 1,
        name: str = "fixed") -> Tuple[tf.Tensor, tf.Tensor]:

    # TODO: Documentation

    with tf.name_scope(f"input_pipeline/{name}"):
        uuids = _load_paths(input_directory, data_set_name)
        print(f"Input pipeline has acess to {len(uuids)} samples")

        # Create an input producer which shuffles uuids, use seed if supplied
        input_producer = tf.train.string_input_producer(uuids, epochs, seed=seed)

        # Add record reading operation
        image, segmentation = _read_record(input_producer, image_size, compression_type)

        # Calculate using safety margin
        capacity = min_after_dequeue + (thread_count + 1) * batch_size

        image_batch, segmentation_batch = tf.train.batch(
            [image, segmentation],
            batch_size,
            thread_count,
            capacity,
            name=f"shuffle_batch")

        return image_batch, segmentation_batch


def create_static_input_pipeline(
        input_directory: str,
        data_set_name: str,
        batch_size: int,
        image_size: int,
        compression_type: tf.python_io.TFRecordCompressionType = tf.python_io.TFRecordCompressionType.ZLIB,
        min_after_dequeue: int = 100,
        thread_count: int = 1,
        name: str = "static") -> Tuple[tf.Tensor, tf.Tensor, int]:

    # TODO: Documentation

    with tf.name_scope(f"input_pipeline/{name}"):
        uuids = _load_paths(input_directory, data_set_name)
        sample_count = len(uuids)

        # Create an input producer which rotates uuids
        input_producer = tf.train.string_input_producer(uuids, shuffle=False)

        # Add record reading operation
        image, segmentation = _read_record(input_producer, image_size, compression_type)

        # Calculate using safety margin
        capacity = min_after_dequeue + (thread_count + 1) * batch_size

        image_batch, segmentation_batch = tf.train.batch(
            [image, segmentation],
            batch_size,
            thread_count,
            capacity,
            name=f"fixed_batch")

        return image_batch, segmentation_batch, sample_count

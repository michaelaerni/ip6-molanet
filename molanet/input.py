import os
from typing import Tuple

import tensorflow as tf


class InputPipeline(object):
    # TODO: This only supports fixed size images, make flexible to support arbitrary sizes
    # TODO: When using arbitrary shapes and they have to be batched, it has to be handled somehow
    # TODO: Refactor common functionality in all pipelines

    def __init__(self, input_directory: str, data_set_name: str, image_size: int):
        if image_size < 1:
            raise ValueError("Image size must be bigger than 0")
        self._image_size = image_size

        # Some methods require absolute paths, sanitize it
        self._input_directory = os.path.abspath(input_directory)
        if not os.path.isdir(self._input_directory):
            raise ValueError(f"Input directory {self._input_directory} does not exist")

        self._sample_directory = os.path.join(self._input_directory, data_set_name)
        if not os.path.isdir(self._sample_directory):
            raise ValueError(f"Sample directory {self._sample_directory} does not exist")

        meta_file_path = os.path.join(self._input_directory, f"{data_set_name}.txt")
        if not os.path.isfile(meta_file_path):
            raise ValueError(f"Meta file {meta_file_path} does not exist")

        # Read uuids into memory and create paths
        with open(meta_file_path) as f:
            self._file_paths = [self._path_from_uuid(self._sample_directory, uuid) for uuid in f.readlines()]

    def create_pipeline(self):
        raise NotImplementedError("This method should be overridden by child classes")

    @property
    def sample_count(self) -> int:
        return len(self._file_paths)

    @staticmethod
    def _path_from_uuid(sample_directory: str, uuid: str) -> str:
        # uuid strings can contain a \n at the end which has to be removed
        return os.path.join(sample_directory, uuid[:2], uuid).strip("\n") + ".tfrecord"

    def _read_record(
            self,
            input_producer: tf.FIFOQueue,
            compression_type: tf.python_io.TFRecordCompressionType) -> Tuple[tf.Tensor, tf.Tensor]:

        reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(compression_type))
        _, serialized_example = reader.read(input_producer, name="read_record")

        raw_features = tf.parse_single_example(serialized_example, features={
            "image": tf.FixedLenFeature([self._image_size, self._image_size, 3], tf.float32),
            "segmentation": tf.FixedLenFeature([self._image_size, self._image_size, 1], tf.float32)
        }, name="parse_record")

        image = raw_features["image"]
        segmentation = raw_features["segmentation"]

        return image, segmentation


class TrainingPipeline(InputPipeline):
    # TODO: Document: This runs multi threaded

    def __init__(
            self,
            input_directory: str,
            data_set_name: str,
            image_size: int,
            batch_size: int,
            read_thread_count: int = 1,
            batch_thread_count: int = 1,
            min_after_dequeue: int = 100,
            compression_type: tf.python_io.TFRecordCompressionType = tf.python_io.TFRecordCompressionType.ZLIB,
            name: str = None
    ):
        super().__init__(input_directory, data_set_name, image_size)

        if batch_size < 1:
            raise ValueError("Batch sizes must at least contain one image")
        self._batch_size = batch_size

        if read_thread_count < 1:
            raise ValueError("There must be at least one reading thread")
        self._read_thread_count = read_thread_count

        if batch_thread_count < 1:
            raise ValueError("There must be at least one batching thread")
        self._batch_thread_count = batch_thread_count

        if min_after_dequeue < 0:
            raise ValueError("Can't have negative number of samples as min after dequeueing")
        self._min_after_dequeue = min_after_dequeue

        self._compression_type = compression_type
        self._name = name

    def create_pipeline(self):

        with tf.name_scope(f"input"), tf.name_scope(self._name, "training"):
            input_producer = tf.train.string_input_producer(self._file_paths, shuffle=True, name="filename_queue")

            # Read records, one reader per thread. Records are already in tanh range
            parsed_samples = [self._read_record(input_producer, self._compression_type)
                              for _ in range(self._read_thread_count)]

            # Calculate capacity for batch queue using safety margin
            capacity = self._min_after_dequeue + self._batch_thread_count * self._batch_size

            # Join threads and batch values, not shuffling because it was shuffled already in string input producer
            image_batch, segmentation_batch = tf.train.batch_join(
                parsed_samples,
                self._batch_size,
                capacity,
                enqueue_many=False,  # Each tensor represents only a single example
                name="shuffle_batch_join"
            )

            return image_batch, segmentation_batch


class EvaluationPipeline(InputPipeline):
    # TODO: Document: This runs single threaded to allow for deterministic behaviour

    def __init__(
            self,
            input_directory: str,
            data_set_name: str,
            image_size: int,
            batch_size: int,
            batch_thread_count: int = 1,
            min_after_dequeue: int = 100,
            compression_type: tf.python_io.TFRecordCompressionType = tf.python_io.TFRecordCompressionType.ZLIB,
            name: str = None
    ):
        super().__init__(input_directory, data_set_name, image_size)

        if batch_size < 1:
            raise ValueError("Batch sizes must at least contain one image")
        self._batch_size = batch_size

        if batch_thread_count < 1:
            raise ValueError("There must be at least one batching thread")
        self._batch_thread_count = batch_thread_count

        if min_after_dequeue < 0:
            raise ValueError("Can't have negative number of samples as min after dequeueing")
        self._min_after_dequeue = min_after_dequeue

        self._compression_type = compression_type
        self._name = name

    def create_pipeline(self):

        with tf.name_scope(f"input"), tf.name_scope(self._name, "evaluation"):
            input_producer = tf.train.string_input_producer(self._file_paths, shuffle=False, name="filename_queue")

            # Read records, just one because the reads happen single threaded. Records are already in tanh range
            parsed_sample = self._read_record(input_producer, self._compression_type)

            # Calculate capacity using safety margin
            capacity = self._min_after_dequeue + self._batch_thread_count * self._batch_size

            # Join threads and batch values
            image_batch, segmentation_batch = tf.train.batch(
                parsed_sample,
                self._batch_size,
                self._batch_thread_count,
                capacity,
                enqueue_many=False,  # Each tensor represents only a single example
                name="batch"
            )

            return image_batch, segmentation_batch


# TODO: Data augmentation functions assume NHWC

# TODO: Could implement random noise for augmentation


def _rotate_90(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    # 90° = Transpose rows and columns -> mirror horizontal
    return _flip_horizontal(tf.transpose(sample_tensor, perm=[1, 0, 2]))


def _rotate_180(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    # 180° = Mirror horizontal and vertical = Reverse rows and columns
    return _flip_horizontal(_flip_vertical(sample_tensor))


def _rotate_270(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    # 270° = Transpose rows and columns -> mirror vertical
    return _flip_vertical(tf.transpose(sample_tensor, perm=[1, 0, 2]))


def _flip_horizontal(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    return tf.reverse(sample_tensor, axis=[1])  # Mirror horizontally (= reverse column order)


def _flip_vertical(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    return tf.reverse(sample_tensor, axis=[0])  # Mirror vertically (= reverse row order)


def _random_flip_single(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    return tf.cond(
        tf.less(tf.random_uniform([], minval=0.0, maxval=1.0), 0.5),  # Calculate random mirroring, 50% chance to mirror
        lambda: _flip_horizontal(sample_tensor),  # Flip horizontal
        lambda: sample_tensor)  # Don't mirror, leave as is


def _random_rotate_single(sample_tensor: tf.Tensor) -> tf.Tensor:
    tf.assert_rank(sample_tensor, 3)

    # Choose random rotation, 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
    rotation = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)

    return tf.case({
        tf.equal(rotation, tf.constant(1)): lambda: _rotate_90(sample_tensor),
        tf.equal(rotation, tf.constant(2)): lambda: _rotate_180(sample_tensor),
        tf.equal(rotation, tf.constant(3)): lambda: _rotate_270(sample_tensor)
    }, default=lambda: sample_tensor, exclusive=True)


def random_rotate_flip_rgb(images: tf.Tensor, segmentations: tf.Tensor, name: str = None) -> Tuple[tf.Tensor, tf.Tensor]:
    # TODO: Document: Rotates randomly [0, 90, 180, 270] degrees and optionally flips horizontal, achieves all transform

    with tf.name_scope("augmentation"), tf.name_scope(name, "random_rotate_flip"):
        # Concat images and segmentations for easier and faster handling
        concatenated_samples = tf.concat([images, segmentations], axis=3, name="concatenate_samples")

        # Apply rotations
        rotated_samples = tf.map_fn(_random_rotate_single, concatenated_samples)

        # Randomly mirror samples individually
        mirrored_samples = tf.map_fn(_random_flip_single, rotated_samples)

        # Extract images and segmentations again
        augmented_images = mirrored_samples[:, :, :, :3]
        augmented_segmentations = mirrored_samples[:, :, :, 3:]

        return augmented_images, augmented_segmentations


def random_contrast_rgb(images: tf.Tensor, lower: float, upper: float, name: str = None) -> tf.Tensor:
    if lower < 0 or lower >= upper:
        raise ValueError("Lower bound must be positive and strictly lower than upper bound")

    with tf.name_scope("augmentation"), tf.name_scope(name, "random_contrast"):
        # Calculate random adjustments for each image in batch
        images_shape = tf.shape(images)
        batch_size = images_shape[0]
        height = images_shape[1]
        width = images_shape[2]
        adjustments = tf.random_uniform([batch_size], minval=lower, maxval=upper, dtype=tf.float32)

        # Tile adjustments to get uniform increase/decrease on all 3 channels of each image
        adjustments = tf.tile(tf.reshape(adjustments, [batch_size, 1, 1, 1]), [1, height, width, 3])

        # Convert images from tanh into [0, 1] range to avoid unexpected behaviour
        # TODO: Is conversion from tanh back really necessary?
        images = tf.divide(tf.add(1.0, images), 2.0)

        # Apply contrast adjustment
        # TODO (x - mean) * contrast_factor + mean

        # Calculate mean for each image for each channel
        means = tf.tile(
            tf.reduce_mean(
                tf.reduce_mean(images, axis=2, keep_dims=True),
                axis=1, keep_dims=True),
            [1, height, width, 1])

        # Subtract mean from each pixel
        images = tf.subtract(images, means)

        # Scale by contrast factor and add means again
        adjusted_images = tf.add(tf.multiply(images, adjustments), means)

        # Convert back to tanh range and clamp
        return tf.clip_by_value(tf.multiply(2.0, tf.subtract(adjusted_images, 0.5)), -1.0, 1.0)


def random_brightness_rgb(images: tf.Tensor, lower: float, upper: float, name: str = None) -> tf.Tensor:
    if lower >= upper:
        raise ValueError("Lower bound must be strictly smaller than upper bound")

    with tf.name_scope("augmentation"), tf.name_scope(name, "random_brightness"):
        # Calculate random adjustments for each image in batch
        images_shape = tf.shape(images)
        batch_size = images_shape[0]
        height = images_shape[1]
        width = images_shape[2]
        adjustments = tf.random_uniform([batch_size], minval=lower, maxval=upper, dtype=tf.float32)

        # Tile adjustments to get uniform increase/decrease on all 3 channels of each image
        adjustments = tf.tile(tf.reshape(adjustments, [batch_size, 1, 1, 1]), [1, height, width, 3])

        # Apply adjustments, clamp into tanh range
        return tf.clip_by_value(tf.add(images, adjustments), -1.0, 1.0)

import os
from typing import Tuple, Callable, List, Optional

import tensorflow as tf


class ColorConverter(object):
    """
    Converts from a color space to another as a part of the input pipeline.
    """

    # TODO: Create convert_back method

    @staticmethod
    def convert(input_image: tf.Tensor) -> tf.Tensor:
        """
        Converts a single image tensor from a color space to another.
        This constructs graph operations.

        :param input_image: Image in the original color scheme with values in tanh range [-1, 1] and shape [h, w, c]
        :return: Tensor which represents the input image in the new color scheme
        """
        raise NotImplementedError("This method should be overridden by child classes")


class RGBToLabConverter(ColorConverter):
    """
    Converter from RGB to CIE Lab color space.
    """

    # TODO: Optimize

    @staticmethod
    def convert(input_image: tf.Tensor) -> tf.Tensor:
        def f(t):
            condition = tf.greater(t, (6.0 / 29.0) ** 3)
            return tf.where(condition, tf.pow(t, 1.0 / 3.0), t / (3.0 * ((6.0 / 29.0) ** 2.0)) + 4.0 / 29.0)

        # convert from [-1,1] to [0,1]
        input_image = (input_image + 1.0) / 2.0
        # rgb must be in range[0,1]
        rgb2xyz = tf.constant([[0.4124564, 0.3575761, 0.1804375],
                               [0.2126729, 0.7151522, 0.0721750],
                               [0.0193339, 0.1191920, 0.9503041]], dtype=tf.float32)

        xyz = tf.matmul(rgb2xyz, tf.transpose(tf.reshape(input_image, [-1, 3]))) * 100.0
        xyz = tf.reshape(tf.transpose(xyz), input_image.get_shape())

        x = tf.reshape(xyz[:, :, 0], [-1])
        y = tf.reshape(xyz[:, :, 1], [-1])
        z = tf.reshape(xyz[:, :, 2], [-1])

        Yn = 100.0
        Zn = 108.883
        Xn = 95.047
        l = 116 * f(y / Yn) - 16.0
        a = 500 * (f(x / Xn) - f(y / Yn))
        b = 200 * (f(y / Yn) - f(z / Zn))

        # l is in range [0,100[
        # a,b are in range [-128,128]
        # convert to tanh range [-1,1]
        l = ((l / 100) - 0.5) * 2.0
        a = (((a + 128) / 256) - 0.5) * 2.0
        b = (((b + 128) / 256) - 0.5) * 2.0

        lab_image = tf.reshape(tf.stack([l, a, b], axis=1), input_image.get_shape())
        return lab_image


class InputPipeline(object):

    class _NoopConverter(ColorConverter):
        @staticmethod
        def convert(input_image: tf.Tensor) -> tf.Tensor:
            return input_image

    # TODO: This only supports fixed size images, make flexible to support arbitrary sizes
    # TODO: When using arbitrary shapes and they have to be batched, it has to be handled somehow
    # TODO: Refactor common functionality in all pipelines

    def __init__(self, input_directory: str, data_set_name: str, image_size: int,
                 color_converter: Optional[ColorConverter] = None):
        if image_size < 1:
            raise ValueError("Image size must be bigger than 0")
        self._image_size = image_size

        self._color_converter = color_converter if color_converter is not None else self._NoopConverter()

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
            color_converter: Optional[ColorConverter] = None,
            augmentation_functions: List[Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]] = None,
            read_thread_count: int = 1,
            batch_thread_count: int = 1,
            min_after_dequeue: int = 100,
            compression_type: tf.python_io.TFRecordCompressionType = tf.python_io.TFRecordCompressionType.ZLIB,
            name: str = None
    ):
        super().__init__(input_directory, data_set_name, image_size, color_converter)

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

        self._augmentation_pipeline = augmentation_functions if augmentation_functions is not None else []

        self._compression_type = compression_type
        self._name = name

    def create_pipeline(self):

        with tf.name_scope(f"input"), tf.name_scope(self._name, "training"):
            input_producer = tf.train.string_input_producer(self._file_paths, shuffle=True, name="filename_queue")

            # Read and augment records, one reader per thread. Records are already in tanh range
            parsed_samples = [self._read_and_augment_record(input_producer)
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

    def _read_and_augment_record(self, input_producer: tf.FIFOQueue) -> Tuple[tf.Tensor, tf.Tensor]:
        # Read actual record
        image, segmentation = self._read_record(input_producer, self._compression_type)

        # Perform data augmentation
        for augmentation_function in self._augmentation_pipeline:
            image, segmentation = augmentation_function(image, segmentation)

        # Perform color scheme conversion
        image = self._color_converter.convert(image)

        return image, segmentation


class EvaluationPipeline(InputPipeline):
    # TODO: Document: This runs single threaded to allow for deterministic behaviour

    def __init__(
            self,
            input_directory: str,
            data_set_name: str,
            image_size: int,
            batch_size: int,
            color_converter: Optional[ColorConverter] = None,
            batch_thread_count: int = 1,
            min_after_dequeue: int = 100,
            compression_type: tf.python_io.TFRecordCompressionType = tf.python_io.TFRecordCompressionType.ZLIB,
            name: str = None
    ):
        super().__init__(input_directory, data_set_name, image_size, color_converter)

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
            parsed_image, parsed_segmentation = self._read_record(input_producer, self._compression_type)

            # Perform color scheme conversion
            parsed_image = self._color_converter.convert(parsed_image)

            # Calculate capacity using safety margin
            capacity = self._min_after_dequeue + self._batch_thread_count * self._batch_size

            # Join threads and batch values
            image_batch, segmentation_batch = tf.train.batch(
                [parsed_image, parsed_segmentation],
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


def random_rotate_flip_rgb(
        image: tf.Tensor, segmentation: tf.Tensor, name: str = None) -> Tuple[tf.Tensor, tf.Tensor]:
    tf.assert_rank(image, 3)

    # TODO: Document: Rotates randomly [0, 90, 180, 270] degrees and optionally flips horizontal, achieves all transform

    with tf.name_scope("augmentation"), tf.name_scope(name, "random_rotate_flip"):
        # Concat images and segmentations for easier and faster handling
        concatenated_sample = tf.concat([image, segmentation], axis=2, name="concatenate_sample")

        # Choose random rotation, 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
        rotation = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)

        rotated_sample = tf.case({
            tf.equal(rotation, tf.constant(1)): lambda: _rotate_90(concatenated_sample),
            tf.equal(rotation, tf.constant(2)): lambda: _rotate_180(concatenated_sample),
            tf.equal(rotation, tf.constant(3)): lambda: _rotate_270(concatenated_sample)
        }, default=lambda: concatenated_sample, exclusive=True)

        # Calculate random mirroring, 50% chance to mirror
        mirrored_sample = tf.cond(
            tf.less(tf.random_uniform([], minval=0.0, maxval=1.0), 0.5),
            lambda: _flip_horizontal(rotated_sample),  # Flip horizontal
            lambda: rotated_sample)  # Don't mirror, leave as is

        # Extract image and segmentation again
        augmented_image = mirrored_sample[:, :, :3]
        augmented_segmentation = mirrored_sample[:, :, 3:]

        return augmented_image, augmented_segmentation


def random_contrast_rgb(image: tf.Tensor, lower: float, upper: float, name: str = None) -> tf.Tensor:
    if lower < 0 or lower >= upper:
        raise ValueError("Lower bound must be positive and strictly lower than upper bound")

    tf.assert_rank(image, 3)

    with tf.name_scope("augmentation"), tf.name_scope(name, "random_contrast"):

        # Calculate random adjustment over whole image
        adjustment = tf.random_uniform([], minval=lower, maxval=upper, dtype=tf.float32)

        # Convert image from tanh into [0, 1] range to avoid unexpected behaviour
        # TODO: Is conversion from tanh back really necessary?
        image = tf.divide(tf.add(1.0, image), 2.0)

        # Apply contrast adjustment
        # TODO: Document: (x - mean) * contrast_factor + mean

        # Calculate mean for each channel
        images_shape = tf.shape(image)
        means = tf.tile(
            tf.reduce_mean(
                tf.reduce_mean(image, axis=1, keep_dims=True),
                axis=0, keep_dims=True),
            [images_shape[0], images_shape[1], 1])

        # Subtract mean from each pixel
        image = tf.subtract(image, means)

        # Scale by contrast factor and add means again
        adjusted_image = tf.add(tf.multiply(image, adjustment), means)

        # Convert back to tanh range and clamp
        return tf.clip_by_value(tf.multiply(2.0, tf.subtract(adjusted_image, 0.5)), -1.0, 1.0)


def random_brightness_rgb(image: tf.Tensor, lower: float, upper: float, name: str = None) -> tf.Tensor:
    if lower >= upper:
        raise ValueError("Lower bound must be strictly smaller than upper bound")

    tf.assert_rank(image, 3)

    with tf.name_scope("augmentation"), tf.name_scope(name, "random_brightness"):
        # Calculate random adjustments over whole image
        adjustment = tf.random_uniform([], minval=lower, maxval=upper, dtype=tf.float32)

        # Apply adjustments, clamp into tanh range
        return tf.clip_by_value(tf.add(image, adjustment), -1.0, 1.0)

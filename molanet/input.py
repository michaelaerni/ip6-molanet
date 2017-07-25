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

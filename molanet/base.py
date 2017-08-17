import csv
import logging
import ntpath
import os
import shutil
from typing import Union, Tuple, NamedTuple, List

import numpy as np
import tensorflow as tf
from PIL import Image

from molanet.input import InputPipeline, ColorConverter
from molanet.operations import use_cpu, select_device, jaccard_index, tanh_to_sigmoid

_log = logging.getLogger(__name__)


class NetworkFactory(object):
    """
    Factory for generator and discriminator networks.

    This is an abstract base class.
    Child classes should inherit override the network creation methods.
    Additional network configuration should be supplied in the child's constructor.
    """

    def create_generator(
            self,
            x: tf.Tensor,
            reuse: bool = False,
            use_gpu: bool = True,
            data_format: str = "NHWC"
    ) -> tf.Tensor:
        """
        Creates a generator network and optionally applies summary options where useful.

        :param x: Input for the created generator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param use_gpu: If True, operations will be created on the gpu. Defaults to True.
        :param data_format: Format of the data matrices, either "NHWC" or "NCHW". Defaults to "NHWC".
        :return: Output tensor of the created generator
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            return_input_tensor: bool = False,
            use_gpu: bool = True,
            data_format: str = "NHWC"
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Creates a discriminator network and optionally applies summary options where useful.

        :param x: Input tensor for the corresponding generator
        :param y: Tensor of the generated or real value for the input x
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param return_input_tensor: If True, the concatenated input tensor which is fed to the network is returned too.
        Defaults to False.
        :param use_gpu: If True, operations will be created on the gpu. Defaults to True.
        :param data_format: Format of the data matrices, either "NHWC" or "NCHW". Defaults to "NHWC".
        :return: Output tensor of the created discriminator as unscaled logits. If return_input_tensor is set to True,
        the concatenated input tensor which is feed to the network is returned too.
        """

        raise NotImplementedError("This method should be overridden by child classes")


class ObjectiveFactory(object):
    """
    Factory for generator and discriminator objectives (cost functions).

    This is an abstract base class.
    Child classes should inherit override the objective creation methods.
    Additional network configuration should be supplied in the child's constructor.
    """

    def create_generator_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            generator: tf.Tensor,
            generator_discriminator: tf.Tensor,
            apply_summary: bool = True,
            use_gpu: bool = True,
            data_format: str = "NHWC"
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        """
        Creates the generator loss function and optionally applies summary options.
        If apply_summary is True, a list of summary operations is returned together with the loss function.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator: Generated output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :param use_gpu: If True, operations will be created on the gpu. Defaults to True.
        :param data_format: Format of the data matrices, either "NHWC" or "NCHW". Defaults to "NHWC".
        :return: Loss function for the generator which can be used for optimization and if specified summary ops
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            generator: tf.Tensor,
            generator_discriminator: tf.Tensor,
            real_discriminator: tf.Tensor,
            apply_summary: bool = True,
            use_gpu: bool = True,
            data_format: str = "NHWC"
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        """
        Creates the discriminator loss function and optionally applies summary options.
        If apply_summary is True, a list of summary operations is returned together with the loss function.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator: Generated output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param real_discriminator: Discriminator logits for the ground truth output
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :param use_gpu: If True, operations will be created on the gpu. Defaults to True.
        :param data_format: Format of the data matrices, either "NHWC" or "NCHW". Defaults to "NHWC".
        :return: Loss function for the discriminator which can be used for optimization and if specified summary ops
        """

        raise NotImplementedError("This method should be overridden by child classes")


def _create_summaries(
        generator: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    generator_classes_raw = tanh_to_sigmoid(generator)
    generated_classes = tf.round(generator_classes_raw)
    real_classes = tanh_to_sigmoid(y)
    generated_positives = tf.reduce_sum(generated_classes)
    real_positives = tf.reduce_sum(real_classes)
    real_negatives = tf.reduce_sum(1.0 - real_classes)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(generated_classes, real_classes), dtype=tf.float32))
    true_positives = tf.reduce_sum(tf.cast(
        tf.logical_and(
            tf.equal(real_classes, tf.ones_like(real_classes)),
            tf.equal(generated_classes, real_classes)),
        dtype=tf.float32))
    true_negatives = tf.reduce_sum(tf.cast(
        tf.logical_and(
            tf.equal(real_classes, tf.zeros_like(real_classes)),
            tf.equal(generated_classes, real_classes)),
        dtype=tf.float32))
    precision = tf.cond(generated_positives > 0, lambda: true_positives / generated_positives, lambda: 1.0)
    recall = tf.cond(real_positives > 0, lambda: true_positives / real_positives, lambda: 1.0)
    f1_score = tf.cond(tf.logical_and(precision > 0, recall > 0),
                       lambda: 2.0 * precision * recall / (precision + recall),
                       lambda: 0.0)
    specificity = tf.cond(real_negatives > 0, lambda: true_negatives / real_negatives, lambda: 1.0)

    jaccard_similarity = tf.reduce_mean(jaccard_index(
        values=generator_classes_raw,
        labels=real_classes
    ))

    return accuracy, precision, recall, f1_score, specificity, jaccard_similarity


def _create_concatenated_images(
        x: tf.Tensor, y: tf.Tensor, generated_y: tf.Tensor,
        color_converter: ColorConverter, data_format: str = "NHWC") -> tf.Tensor:

    if data_format == "NCHW":
        # Convert from NCHW back to NHWC
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        generated_y = tf.transpose(generated_y, [0, 2, 3, 1])

    # Generate difference image, RED is false positives, BLUE is false negatives
    difference = tf.subtract(generated_y, y)
    difference_positive = tf.clip_by_value(difference, 0.0, 1.0)
    difference_negative = tf.abs(tf.clip_by_value(difference, -1.0, 0.0))
    difference_image = tf.concat([difference_positive, tf.zeros_like(difference), difference_negative], axis=3)

    return tf.cast(tf.round(tf.concat([
        (color_converter.convert_back(x) + 1.0) / 2.0 * 255.0,
        tf.tile((y + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
        tf.tile((generated_y + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
        difference_image * 255.0
    ], axis=2)), dtype=tf.uint8)


class TrainingOptions(NamedTuple):
    """
    Options used to specify various aspects of model training.
    """

    summary_directory: str
    max_iterations: int
    training_summary_interval: int = 20
    cv_summary_interval: int = 100
    save_model_interval: int = 1000
    discriminator_iterations: int = 1
    session_configuration: tf.ConfigProto = None
    use_gpu: bool = True
    data_format: str = "NHWC"


class NetworkTrainer(object):
    """
    Utility to train a model, optionally restoring it from a checkpoint.

    This class handles cross-validation evaluation, model saving,
    logging and TensorFlow sessions.
    """

    def __init__(
            self,
            training_pipeline: InputPipeline,
            cv_pipeline: InputPipeline,
            network_factory: NetworkFactory,
            objective_factory: ObjectiveFactory,
            training_options: TrainingOptions,
            learning_rate: float,
            beta1: float = 0.9,
            beta2: float = 0.999):
        """
        Create a new network trainer
        :param training_pipeline: Input pipeline used for training
        :param cv_pipeline: Input pipeline used for cross-validation
        :param network_factory: Factory to create training and evaluation networks
        :param objective_factory: Factory to create generator and discriminator losses
        :param training_options: Options controlling the training process
        :param learning_rate: Learning rate to use in the Adam optimizer
        :param beta1: Beta1 to use in the Adam optimizer
        :param beta2: Beta2 to use in the Adam optimizer
        """

        self._training_options = training_options
        self._restored_iteration = None

        # Create input pipelines
        with use_cpu():
            self._training_pipeline = training_pipeline
            self._train_x, self._train_y, _ = training_pipeline.create_pipeline()
            self._cv_pipeline = cv_pipeline
            self._cv_x, self._cv_y, _ = self._cv_pipeline.create_pipeline()

        # Create training graph
        with tf.name_scope("training"):

            # Create networks
            self._generator = network_factory.create_generator(self._train_x, use_gpu=self._training_options.use_gpu,
                                                               data_format=self._training_options.data_format)
            self._discriminator_generated = network_factory.create_discriminator(
                self._train_x, self._generator, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)
            self._discriminator_real = network_factory.create_discriminator(
                self._train_x, self._train_y, reuse=True, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)

            # Create losses
            self._generator_loss, generator_summary = objective_factory.create_generator_loss(
                self._train_x, self._train_y,
                self._generator, self._discriminator_generated, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)
            self._discriminator_loss, discriminator_summary = objective_factory.create_discriminator_loss(
                self._train_x, self._train_y,
                self._generator, self._discriminator_generated, self._discriminator_real,
                use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)

            with tf.device(select_device(self._training_options.use_gpu)):
                # Create optimizers
                trainable_variables = tf.trainable_variables()
                variables_discriminator = [var for var in trainable_variables if var.name.startswith("discriminator")]
                variables_generator = [var for var in trainable_variables if var.name.startswith("generator")]

                self._optimizer_generator = tf.train.AdamOptimizer(learning_rate, beta1, beta2, name="adam_generator")
                self._optimizer_discriminator = tf.train.AdamOptimizer(learning_rate, beta1, beta2, name="adam_discriminator")

                self._op_generator = self._optimizer_generator.minimize(self._generator_loss, var_list=variables_generator)
                self._op_discriminator = self._optimizer_discriminator.minimize(self._discriminator_loss, var_list=variables_discriminator)

            with use_cpu():
                # Iteration counter
                self._global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
                self._step_op = tf.assign_add(self._global_step, 1)

            # Create summary operation
            accuracy, precision, recall, f1_score, specificity, jaccard_similarity = _create_summaries(self._generator, self._train_y)
            summary_operations = [
                tf.summary.scalar("accuracy", accuracy),
                tf.summary.scalar("precision", precision),
                tf.summary.scalar("recall", recall),
                tf.summary.scalar("f1_score", f1_score),
                tf.summary.scalar("specificity", specificity),
                tf.summary.scalar("jaccard_similarity", jaccard_similarity)
            ]

            self._train_saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

            # Merge summaries
            self._train_summary = tf.summary.merge(summary_operations + generator_summary + discriminator_summary)
            self._train_summary_writer = tf.summary.FileWriter(
                os.path.join(self._training_options.summary_directory, "training"), graph=tf.get_default_graph())

        # Create CV graph
        with tf.name_scope("cv"):
            # Create networks
            generator = network_factory.create_generator(
                self._cv_x, reuse=True, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)
            discriminator_generated = network_factory.create_discriminator(
                self._cv_x, generator, reuse=True, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)
            discriminator_real = network_factory.create_discriminator(
                self._cv_x, self._cv_y, reuse=True, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)

            # Create losses
            _, generator_summary = objective_factory.create_generator_loss(
                self._cv_x, self._cv_y, generator, discriminator_generated, use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)
            _, discriminator_summary = objective_factory.create_discriminator_loss(
                self._cv_x, self._cv_y, generator, discriminator_generated, discriminator_real,
                use_gpu=self._training_options.use_gpu,
                data_format=self._training_options.data_format)

            # Create other summary options
            accuracy, precision, recall, f1_score, specificity, jaccard_similarity = _create_summaries(generator, self._cv_y)

            # Create summary operation
            summary_operations = [
                tf.summary.scalar("accuracy", accuracy),
                tf.summary.scalar("precision", precision),
                tf.summary.scalar("recall", recall),
                tf.summary.scalar("f1_score", f1_score),
                tf.summary.scalar("specificity", specificity),
                tf.summary.scalar("jaccard_similarity", jaccard_similarity)
            ]

            with use_cpu():
                # Concatenated images
                self._concatenated_images_op = _create_concatenated_images(
                    self._cv_x,
                    self._cv_y,
                    generator,
                    self._cv_pipeline.color_converter,
                    self._training_options.data_format
                )

            # Merge summaries
            self._cv_summary = tf.summary.merge(summary_operations + generator_summary + discriminator_summary)
            self._cv_summary_writer = tf.summary.FileWriter(
                os.path.join(self._training_options.summary_directory, "cv"))

    def __enter__(self):
        self._sess = tf.Session(config=self._training_options.session_configuration)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.close()
        self._sess = None

    def train(self):
        """
        Trains the model until a maximum number of iterations as specified in the training options is reached.

        This method requires this trainer to have __enter__ called previously, otherwise no session exists
        and calls to this method will fail.
        """
        if self._sess is None:
            raise RuntimeError("A running session is required to start training")

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        _log.info("Starting queue runners...")
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)

        save_model_path = os.path.join(self._training_options.summary_directory, "model.ckpt")
        save_image_path = os.path.join(self._training_options.summary_directory, "images/")

        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

        with tf.device(select_device(self._training_options.use_gpu)):
            init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

        tf.get_default_graph().finalize()

        if self._restored_iteration is None:
            self._sess.run(init_ops)

        iteration = self._sess.run(self._global_step)

        _log.info("Starting training")

        try:
            while not coord.should_stop():
                current_iteration = iteration
                iteration = self._sess.run(self._step_op)

                if current_iteration % self._training_options.save_model_interval == 0:
                    self._train_saver.save(self._sess, save_model_path, global_step=self._global_step)
                    _log.info(f"Saved model from iteration {iteration}")

                # Run CV validation
                if current_iteration % self._training_options.cv_summary_interval == 0:
                    _log.info("Evaluating CV set...")

                    # TODO: Make configurable?
                    IMAGE_OUTPUT_COUNT = 5
                    image_output_iterations = min(IMAGE_OUTPUT_COUNT, self._cv_pipeline.sample_count)

                    # Generate individual summaries
                    cv_summaries = []
                    cv_images = []
                    for _ in range(image_output_iterations):
                        current_summary, current_image = self._sess.run(
                            (self._cv_summary, self._concatenated_images_op))
                        cv_summaries.append(current_summary)
                        cv_images.append(current_image)

                    for _ in range(image_output_iterations, self._cv_pipeline.sample_count):
                        cv_summaries.append(self._sess.run(self._cv_summary))

                    # Convert summaries into numpy array and calculate the average for all tags
                    # TODO: Validate that only scalar summaries
                    summary_tags = [entry.tag for entry in tf.Summary.FromString(cv_summaries[0]).value]
                    tag_index_mapping = {tag: idx for idx, tag in enumerate(summary_tags)}
                    summary_values = np.zeros((len(cv_summaries), len(summary_tags)), dtype=np.float32)
                    for idx, current_summary in enumerate(cv_summaries):
                        for entry in tf.Summary.FromString(current_summary).value:
                            summary_values[idx, tag_index_mapping[entry.tag]] = entry.simple_value

                    real_summary = np.mean(summary_values, axis=0)

                    # Create output summary proto
                    value_list = [
                        tf.Summary.Value(tag=tag, simple_value=real_summary[idx])
                        for idx, tag in enumerate(summary_tags)]
                    result_proto = tf.Summary(value=value_list)

                    # Write summary
                    self._cv_summary_writer.add_summary(result_proto, current_iteration)

                    # Write images
                    # TODO: Don't use hardcoded size
                    for idx, current_image in enumerate(cv_images):
                        output_image = np.reshape(current_image, (512, 512 * 4, 3))
                        Image.fromarray(output_image, "RGB").save(
                            os.path.join(save_image_path, f"sample_{idx:02d}_{iteration:08d}.png"))

                # Train discriminator
                for _ in range(self._training_options.discriminator_iterations):
                    self._sess.run(self._op_discriminator)

                # Train generator, optionally output summary
                if current_iteration % self._training_options.training_summary_interval == 0:
                    _, current_summary = self._sess.run([self._op_generator, self._train_summary])

                    self._train_summary_writer.add_summary(current_summary, iteration)

                    _log.info(f"Iteration {iteration} done")
                else:
                    self._sess.run(self._op_generator)

                # Check for iteration limit reached
                if iteration > self._training_options.max_iterations:
                    coord.request_stop()

        except Exception as ex:
            coord.request_stop(ex)
        finally:
            coord.request_stop()

            _log.info("Waiting for threads to finish...")
            coord.join(threads)

            # Close writers AFTER threads stopped to make sure summaries are written
            self._train_summary_writer.close()
            self._cv_summary_writer.close()

        _log.info("Training finished")

    def restore(self, iteration):
        """
        Restores the model checkpoint for the given iteration into the currently active session.

        This method requires this trainer to have __enter__ called previously, otherwise no session exists
        and calls to this method will fail.

        The model checkpoint is read relatively to the output directory specified in the training options.
        :param iteration: Iteration to restore
        """
        if self._sess is None:
            raise RuntimeError("A running session is required to restore a model")

        self._train_saver.restore(self._sess, os.path.join(self._training_options.summary_directory, f"model.ckpt-{iteration}"))
        self._restored_iteration = self._sess.run(self._global_step)
        _log.info(f"Iteration {self._restored_iteration} restored")


class NetworkEvaluator(object):
    """
    Utility to evaluate a previously trained model restored from a checkpoint.

    Evaluation consists of running all input pipeline samples once through the generator.
    All generated images as well as a summary file are written to the specified output directory.

    This class handles TensorFlow sessions internally.
    """
    def __init__(
            self,
            pipeline: InputPipeline,
            network_factory: NetworkFactory,
            checkpoint_file: str,
            output_directory: str,
            use_gpu: bool = False,
            data_format: str = "NHWC"
    ):
        """
        Creates a new network evaluator.
        :param pipeline: Pipeline providing the samples to evaluate
        :param network_factory: Network factory to create a generator network
        :param checkpoint_file: Path to the checkpoint file to be restored
        :param output_directory: Directory into which the evaluation results are written
        :param use_gpu: If True, the GPU will be used instead of the CPU. Defaults to False.
        :param data_format: Data format of the samples. Defaults to "NHWC".
        """
        self._pipeline = pipeline

        self._checkpoint_file = checkpoint_file

        self._output_directory = os.path.abspath(output_directory)
        self._image_directory = os.path.join(self._output_directory, "images")
        self._output_file = os.path.join(self._output_directory, "results.csv")

        # Create graph
        with use_cpu():
            self._x, self._y, self._file_names = self._pipeline.create_pipeline()

        # Create networks
        generator = network_factory.create_generator(self._x, use_gpu=use_gpu, data_format=data_format)

        with use_cpu():
            # Create basic summary
            self._accuracy, self._precision, self._recall, self._f1_score, self._specificity, self._jaccard_similarity \
                = _create_summaries(generator, self._y)

            # Concatenated images
            self._concatenated_images = _create_concatenated_images(
                self._x,
                self._y,
                generator,
                self._pipeline.color_converter,
                data_format
            )

        self._saver = tf.train.Saver()

        tf.get_default_graph().finalize()

    def __enter__(self):
        self._sess = tf.Session()

        self._saver.restore(self._sess, self._checkpoint_file)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.close()
        self._sess = None

    def evaluate(self):
        """
        Evaluate all samples in the input pipeline once and write the output into the specified directory.

        This method requires this evaluator to have __enter__ called previously, otherwise no session exists
        and calls to this method will fail.
        """
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        _log.info("Starting queue runners...")
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)

        # Prepare output directory
        if os.path.exists(self._output_directory):
            shutil.rmtree(self._output_directory, ignore_errors=True)
        os.makedirs(self._output_directory)
        os.makedirs(self._image_directory)

        _log.info("Starting evaluation")

        sample_ids = []
        evaluation_numbers = np.zeros((self._pipeline.sample_count, 6))

        ops = [
            self._accuracy,
            self._precision,
            self._recall,
            self._f1_score,
            self._specificity,
            self._jaccard_similarity,
            self._file_names,
            self._concatenated_images
        ]

        try:
            while not coord.should_stop():
                for idx in range(self._pipeline.sample_count):
                    accuracy, precision, recall, f1_score, specificity, jaccard_similarity, work_item, image \
                        = self._sess.run(ops)

                    # Convert work item to path, remove offset counter at the end and file extension
                    file_path, _ = os.path.splitext(work_item[0].decode("UTF-8").split(":")[-2])
                    sample_id = ntpath.basename(file_path)

                    # Store summary numbers
                    sample_ids.append(sample_id)
                    evaluation_numbers[idx, 0] = accuracy
                    evaluation_numbers[idx, 1] = precision
                    evaluation_numbers[idx, 2] = recall
                    evaluation_numbers[idx, 3] = f1_score
                    evaluation_numbers[idx, 4] = specificity
                    evaluation_numbers[idx, 5] = jaccard_similarity

                    # Save image
                    output_image = np.reshape(image, (512, 512 * 4, 3))  # TODO: Remove fixed size
                    Image.fromarray(output_image, "RGB").save(
                        os.path.join(self._image_directory, f"{sample_id}.png"))

                    _log.info(f"[{idx + 1}] Evaluated {sample_id}")

                coord.request_stop()

        except Exception as ex:
            coord.request_stop(ex)
        finally:
            coord.request_stop()

            _log.info("Waiting for threads to finish...")
            coord.join(threads)

        with open(self._output_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "uuid",
                "segmentation_id",
                "accuarcy",
                "precision",
                "recall",
                "f1_score",
                "specificity",
                "jaccard_similarity"
            ])

            for idx, sample_id in enumerate(sample_ids):
                uuid = sample_id.split("_")[0]
                segmentation_id = sample_id.split("_")[1]
                writer.writerow([
                    uuid,
                    segmentation_id,
                    evaluation_numbers[idx, 0],
                    evaluation_numbers[idx, 1],
                    evaluation_numbers[idx, 2],
                    evaluation_numbers[idx, 3],
                    evaluation_numbers[idx, 4],
                    evaluation_numbers[idx, 5]
                ])

        # Generate overall summary
        total_accuracy, total_precision, total_recall, total_f1_score, total_specificity, total_jaccard_similarity = \
            np.mean(evaluation_numbers, axis=0)
        _log.info("Evaluation done")
        _log.info(f"Average accuracy: {total_accuracy}")
        _log.info(f"Average precision: {total_precision}")
        _log.info(f"Average recall: {total_recall}")
        _log.info(f"Average f1 score: {total_f1_score}")
        _log.info(f"Average specificity: {total_specificity}")
        _log.info(f"Average jaccard similarity: {total_jaccard_similarity}")

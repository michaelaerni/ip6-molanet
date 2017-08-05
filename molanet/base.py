import csv
import ntpath
import os
from typing import Union, Tuple, NamedTuple, List

import numpy as np
import shutil
import tensorflow as tf
from PIL import Image

from molanet.input import InputPipeline
from molanet.operations import use_cpu, select_device


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
            use_gpu: bool = True
    ) -> tf.Tensor:
        """
        Creates a generator network and optionally applies summary options where useful.

        :param x: Input for the created generator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param use_gpu: If True, operations will be created on the gpu. Defaults to True.
        :return: Output tensor of the created generator
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            return_input_tensor: bool = False,
            use_gpu: bool = True
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Creates a discriminator network and optionally applies summary options where useful.

        :param x: Input tensor for the corresponding generator
        :param y: Tensor of the generated or real value for the input x
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param return_input_tensor: If True, the concatenated input tensor which is fed to the network is returned too.
        Defaults to False.
        :param use_gpu: If True, operations will be created on the gpu. Defaults to True.
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
            use_gpu: bool = True
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
            use_gpu: bool = True
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
        :return: Loss function for the discriminator which can be used for optimization and if specified summary ops
        """

        raise NotImplementedError("This method should be overridden by child classes")


def _create_summaries(generator: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    generated_classes = tf.round((generator + 1.0) / 2.0)
    real_classes = (y + 1.0) / 2.0
    generated_positives = tf.reduce_sum(generated_classes)
    real_positives = tf.reduce_sum(real_classes)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(generated_classes, real_classes), dtype=tf.float32))
    true_positives = tf.reduce_sum(tf.cast(
        tf.logical_and(
            tf.equal(real_classes, tf.ones_like(real_classes)),
            tf.equal(generated_classes, real_classes)),
        dtype=tf.float32))
    # TODO: Zero handling in precision, recall and f1 are a bit wonky, discuss and fix this
    precision = tf.cond(generated_positives > 0, lambda: true_positives / generated_positives, lambda: 1.0)
    recall = tf.cond(real_positives > 0, lambda: true_positives / real_positives, lambda: 1.0)
    f1_score = tf.cond(tf.logical_and(precision > 0, recall > 0),
                       lambda: 2.0 * precision * recall / (precision + recall),
                       lambda: 0.0)

    return accuracy, precision, recall, f1_score


class TrainingOptions(NamedTuple):
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
    # TODO: Doc

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
        self._training_options = training_options

        # Create input pipelines
        with use_cpu():
            self._training_pipeline = training_pipeline
            self._train_x, self._train_y, _ = training_pipeline.create_pipeline()
            self._cv_pipeline = cv_pipeline
            self._cv_x, self._cv_y, _ = self._cv_pipeline.create_pipeline()

        # Create training graph
        with tf.name_scope("training"):

            # Create networks
            self._generator = network_factory.create_generator(self._train_x, use_gpu=self._training_options.use_gpu)
            self._discriminator_generated = network_factory.create_discriminator(
                self._train_x, self._generator, use_gpu=self._training_options.use_gpu)
            self._discriminator_real = network_factory.create_discriminator(
                self._train_x, self._train_y, reuse=True, use_gpu=self._training_options.use_gpu)

            # Create losses
            self._generator_loss, generator_summary = objective_factory.create_generator_loss(
                self._train_x, self._train_y,
                self._generator, self._discriminator_generated, use_gpu=self._training_options.use_gpu)
            self._discriminator_loss, discriminator_summary = objective_factory.create_discriminator_loss(
                self._train_x, self._train_y,
                self._generator, self._discriminator_generated, self._discriminator_real,
                use_gpu=self._training_options.use_gpu)

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
            accuracy, precision, recall, f1_score = _create_summaries(self._generator, self._train_y)
            summary_operations = [
                tf.summary.scalar("accuracy", accuracy),
                tf.summary.scalar("precision", precision),
                tf.summary.scalar("recall", recall),
                tf.summary.scalar("f1_score", f1_score)
            ]

            # TODO: Input pipeline summary is lost

            self._train_saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

            # Merge summaries
            self._train_summary = tf.summary.merge(summary_operations + generator_summary + discriminator_summary)
            self._train_summary_writer = tf.summary.FileWriter(
                os.path.join(self._training_options.summary_directory, "training"), graph=tf.get_default_graph())

        # Create CV graph
        with tf.name_scope("cv"):
            # Create networks
            generator = network_factory.create_generator(
                self._cv_x, reuse=True, use_gpu=self._training_options.use_gpu)
            discriminator_generated = network_factory.create_discriminator(
                self._cv_x, generator, reuse=True, use_gpu=self._training_options.use_gpu)
            discriminator_real = network_factory.create_discriminator(
                self._cv_x, self._cv_y, reuse=True, use_gpu=self._training_options.use_gpu)

            # Create losses
            _, generator_summary = objective_factory.create_generator_loss(
                self._cv_x, self._cv_y, generator, discriminator_generated, use_gpu=self._training_options.use_gpu)
            _, discriminator_summary = objective_factory.create_discriminator_loss(
                self._cv_x, self._cv_y, generator, discriminator_generated, discriminator_real,
                use_gpu=self._training_options.use_gpu)

            # Create other summary options
            accuracy, precision, recall, f1_score = _create_summaries(generator, self._cv_y)

            # Create summary operation
            summary_operations = [
                tf.summary.scalar("accuracy", accuracy),
                tf.summary.scalar("precision", precision),
                tf.summary.scalar("recall", recall),
                tf.summary.scalar("f1_score", f1_score)
            ]

            with use_cpu():
                # Concatenated images
                self._concatenated_images_op = tf.cast(tf.round(tf.concat([
                    (self._cv_pipeline.color_converter.convert_back(self._cv_x) + 1.0) / 2.0 * 255.0,
                    tf.tile((self._cv_y + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
                    tf.tile((generator + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
                    tf.tile(tf.abs(tf.subtract(generator, self._cv_y)), multiples=[1, 1, 1, 3]) * 255.0
                ], axis=2)), dtype=tf.uint8)

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
        if self._sess is None:
            raise RuntimeError("A running session is required to start training")

        # TODO: Remove prints everywhere

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        print("Starting queue runners...")
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)

        save_model_path = os.path.join(self._training_options.summary_directory, "model.ckpt")
        save_image_path = os.path.join(self._training_options.summary_directory, "images/")

        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

        with tf.device(select_device(self._training_options.use_gpu)):
            init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

        tf.get_default_graph().finalize()

        # TODO: Does this work with restore?
        self._sess.run(init_ops)
        iteration = self._sess.run(self._global_step)

        print("Starting training")

        try:
            while not coord.should_stop():
                current_iteration = iteration
                iteration = self._sess.run(self._step_op)

                if current_iteration % self._training_options.save_model_interval == 0:
                    self._train_saver.save(self._sess, save_model_path, global_step=iteration)
                    print(f"Saved model from iteration {iteration}")

                # Run CV validation
                if current_iteration % self._training_options.cv_summary_interval == 0:
                    print("Evaluating CV set...")

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

                    print(f"Iteration {iteration} done")
                else:
                    self._sess.run(self._op_generator)

                # Check for iteration limit reached
                if iteration >= self._training_options.max_iterations:
                    coord.request_stop()

        except Exception as ex:
            coord.request_stop(ex)
        finally:
            coord.request_stop()

            print("Waiting for threads to finish...")
            coord.join(threads)

            # Close writers AFTER threads stopped to make sure summaries are written
            self._train_summary_writer.close()
            self._cv_summary_writer.close()

        print("Training finished")

    def restore(self, iteration):
        if self._sess is None:
            raise RuntimeError("A running session is required to restore a model")

        self._train_saver.restore(self._sess, os.path.join(self._training_options.summary_directory, f"model.ckpt-{iteration}"))


class NetworkEvaluator(object):
    def __init__(
            self,
            pipeline: InputPipeline,
            network_factory: NetworkFactory,
            checkpoint_file: str,
            output_directory: str,
            use_gpu: bool = False
    ):
        # TODO: Assert batch_size == 1
        self._pipeline = pipeline

        self._checkpoint_file = checkpoint_file

        self._output_directory = os.path.abspath(output_directory)
        self._image_directory = os.path.join(self._output_directory, "images")
        self._output_file = os.path.join(self._output_directory, "results.csv")

        # Create graph
        with use_cpu():
            self._x, self._y, self._file_names = self._pipeline.create_pipeline()

        # Create networks
        generator = network_factory.create_generator(self._x, use_gpu=use_gpu)
        # discriminator_generated = network_factory.create_discriminator(self._x, generator, use_gpu=use_gpu)
        # discriminator_real = network_factory.create_discriminator(self._x, self._y, reuse=True, use_gpu=use_gpu)

        # TODO: Refactor loss output in a way that objective can also be evaluated

        with use_cpu():
            # Create basic summary
            self._accuracy, self._precision, self._recall, self._f1_score = _create_summaries(generator, self._y)

            # Concatenated images
            self._concatenated_images = tf.cast(tf.round(tf.concat([
                (self._pipeline.color_converter.convert_back(self._x) + 1.0) / 2.0 * 255.0,
                tf.tile((self._y + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
                tf.tile((generator + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
                tf.tile(tf.abs(tf.subtract(generator, self._y)), multiples=[1, 1, 1, 3]) * 255.0
            ], axis=2)), dtype=tf.uint8)

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
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        print("Starting queue runners...")
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)

        # Prepare output directory
        if os.path.exists(self._output_directory):
            shutil.rmtree(self._output_directory, ignore_errors=True)
        os.makedirs(self._output_directory)
        os.makedirs(self._image_directory)

        print("Starting evaluation")

        sample_ids = []
        evaluation_numbers = np.zeros((self._pipeline.sample_count, 4))

        ops = [
            self._accuracy,
            self._precision,
            self._recall,
            self._f1_score,
            self._file_names,
            self._concatenated_images
        ]

        try:
            while not coord.should_stop():
                for idx in range(self._pipeline.sample_count):
                    accuracy, precision, recall, f1_score, work_item, image = self._sess.run(ops)

                    # Convert work item to path, remove offset counter at the end and file extension
                    file_path, _ = os.path.splitext(work_item[0].decode("UTF-8").split(":")[-2])
                    sample_id = ntpath.basename(file_path)

                    # Store summary numbers
                    sample_ids.append(sample_id)
                    evaluation_numbers[idx, 0] = accuracy
                    evaluation_numbers[idx, 1] = precision
                    evaluation_numbers[idx, 2] = recall
                    evaluation_numbers[idx, 3] = f1_score

                    # Save image
                    output_image = np.reshape(image, (512, 512 * 4, 3))  # TODO: Remove fixed size
                    Image.fromarray(output_image, "RGB").save(
                        os.path.join(self._image_directory, f"{sample_id}.png"))

                    print(f"[{idx + 1}] Evaluated {sample_id}")

                coord.request_stop()

        except Exception as ex:
            coord.request_stop(ex)
        finally:
            coord.request_stop()

            print("Waiting for threads to finish...")
            coord.join(threads)

        with open(self._output_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["uuid", "segmentation_id", "accuarcy", "precision", "recall", "f1_score"])

            for idx, sample_id in enumerate(sample_ids):
                uuid, segmentation_id = sample_id.split("_")
                writer.writerow([
                    uuid,
                    segmentation_id,
                    evaluation_numbers[idx, 0],
                    evaluation_numbers[idx, 1],
                    evaluation_numbers[idx, 2],
                    evaluation_numbers[idx, 3]
                ])

        # Generate overall summary
        total_accuracy, total_precision, total_recall, total_f1_score = np.mean(evaluation_numbers, axis=0)
        print("Evaluation done")
        print("Average accuracy:", total_accuracy)
        print("Average precision:", total_precision)
        print("Average recall:", total_recall)
        print("Average f1 score:", total_f1_score)

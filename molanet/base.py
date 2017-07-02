import os
import shutil
from typing import Union, Tuple, NamedTuple

import numpy as np
import tensorflow as tf
from PIL import Image


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
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates a generator network and optionally applies summary options where useful.

        :param x: Input for the created generator
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Output tensor of the created generator
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True,
            return_input_tensor: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Creates a discriminator network and optionally applies summary options where useful.

        :param x: Input tensor for the corresponding generator
        :param y: Tensor of the generated or real value for the input x
        :param reuse: If False, the weights cannot exist yet, if True they will be reused. Defaults to False.
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :param return_input_tensor: If True, the concatenated input tensor which is fed to the network is returned too.
        Defaults to False.
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
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates the generator loss function and optionally applies summary options.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator: Generated output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Loss function for the generator which can be used for optimization
        """

        raise NotImplementedError("This method should be overridden by child classes")

    def create_discriminator_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            generator: tf.Tensor,
            generator_discriminator: tf.Tensor,
            real_discriminator: tf.Tensor,
            apply_summary: bool = True
    ) -> tf.Tensor:
        """
        Creates the discriminator loss function and optionally applies summary options.

        :param x: Input tensor which is fed to the generator
        :param y: Ground truth output for the given x
        :param generator: Generated output for the given x
        :param generator_discriminator: Discriminator logits for the generated output corresponding to the x
        :param real_discriminator: Discriminator logits for the ground truth output
        :param apply_summary: If True, summary operations will be added to the graph. Defaults to True.
        :return: Loss function for the discriminator which can be used for optimization
        """

        raise NotImplementedError("This method should be overridden by child classes")


class TrainingOptions(NamedTuple):
    summary_directory: str
    save_summary_interval: int = 10
    save_model_interval: int = 1000


class NetworkTrainer(object):
    # TODO: Doc

    def __init__(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            network_factory: NetworkFactory,
            objective_factory: ObjectiveFactory,
            training_options: TrainingOptions,
            learning_rate: float,
            beta1: float = 0.9,
            beta2: float = 0.999):
        self._training_options = training_options

        self._x = x
        self._y = y

        # Create networks
        self._generator = network_factory.create_generator(x)
        self._discriminator_generated = network_factory.create_discriminator(x, self._generator)
        self._discriminator_real = network_factory.create_discriminator(x, y, reuse=True)

        # Create losses
        self._generator_loss = objective_factory.create_generator_loss(
            x, y, self._generator, self._discriminator_generated)
        self._discriminator_loss = objective_factory.create_discriminator_loss(
            x, y, self._generator, self._discriminator_generated, self._discriminator_real)

        # Create optimizers
        trainable_variables = tf.trainable_variables()
        variables_discriminator = [var for var in trainable_variables if var.name.startswith("discriminator")]
        variables_generator = [var for var in trainable_variables if var.name.startswith("generator")]

        self._optimizer_generator = tf.train.AdamOptimizer(learning_rate, beta1, beta2, name="adam_generator")
        self._optimizer_discriminator = tf.train.AdamOptimizer(learning_rate, beta1, beta2, name="adam_discriminator")

        self._op_generator = self._optimizer_generator.minimize(self._generator_loss, var_list=variables_generator)
        self._op_discriminator = self._optimizer_discriminator.minimize(self._discriminator_loss, var_list=variables_discriminator)

        # Iteration counter
        self._global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.uint16)

    def train(self, sess: tf.Session):
        # TODO: Remove prints everywhere

        # Create summaries
        generated_classes = tf.round((self._generator + 1.0) / 2.0)
        real_classes = (self._y + 1.0) / 2.0
        generated_positives = tf.reduce_sum(generated_classes)
        real_positives = tf.reduce_sum(real_classes)
        # TODO: Accuracy seems to be wrong
        accuracy = tf.reduce_mean(tf.cast(tf.equal(generated_classes, real_classes), dtype=tf.float32))
        true_positives = tf.reduce_sum(tf.cast(
            tf.logical_and(
                tf.equal(real_classes, tf.ones_like(real_classes)),
                tf.equal(generated_classes, real_classes)),
            dtype=tf.float32))
        precision = true_positives / generated_positives
        recall = true_positives / real_positives
        f1_score = 2.0 * precision * recall / (precision + recall)

        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("precision", precision)
        tf.summary.scalar("recall", recall)
        tf.summary.scalar("f1_score", f1_score)

        # TODO: Better summary handling
        summary = tf.summary.merge_all()

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step_update = tf.assign_add(self._global_step, 1)

        iteration = sess.run(self._global_step)

        operations = [
            self._op_discriminator,
            self._op_generator,
            step_update,
            self._global_step]

        save_model_path = os.path.join(self._training_options.summary_directory, "model.ckpt")
        save_image_path = os.path.join(self._training_options.summary_directory, "images/")

        shutil.rmtree(save_image_path, ignore_errors=True)
        os.makedirs(save_image_path)

        concatenated_images = tf.cast(tf.round(tf.concat([
            (self._x + 1.0) / 2.0 * 255.0,
            tf.tile((self._y + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
            tf.tile((self._generator + 1.0) / 2.0 * 255.0, multiples=[1, 1, 1, 3]),
            tf.tile(tf.abs(tf.subtract(self._generator, self._y)), multiples=[1, 1, 1, 3]) * 255.0
        ], axis=2)), dtype=tf.uint8)

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        summary_writer = tf.summary.FileWriter(self._training_options.summary_directory, sess.graph)
        try:
            while not coord.should_stop():
                if iteration % self._training_options.save_model_interval == 0:
                    saver.save(sess, save_model_path, global_step=iteration)

                if iteration % self._training_options.save_summary_interval == 0:
                    step_result = sess.run(operations + [summary, concatenated_images])
                    iteration = step_result[-3]
                    current_summary = step_result[-2]
                    summary_writer.add_summary(current_summary, iteration)

                    generated_images = step_result[-1]
                    # TODO: Don't use hardcoded size
                    # Take first image for output
                    output_image = np.reshape(generated_images[0], (512, 512 * 4, 3))
                    Image.fromarray(output_image, "RGB").save(
                        os.path.join(save_image_path, f"sample_{iteration:08d}.png"))

                    print(f"Iteration {iteration} done")
                else:
                    step_result = sess.run(operations)
                    iteration = step_result[-1]

        except tf.errors.OutOfRangeError:
            print("Epoch limit reached, training stopped")
        finally:
            summary_writer.close()
            coord.request_stop()

        print("Waiting for threads to finish...")
        coord.join(threads)

        print("Training finished")

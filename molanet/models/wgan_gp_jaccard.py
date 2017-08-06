from typing import List, Tuple, Union

import tensorflow as tf

from molanet.base import ObjectiveFactory, NetworkFactory
from molanet.operations import use_cpu, select_device, jaccard_index, tanh_to_sigmoid


class WassersteinJaccardFactory(ObjectiveFactory):

    def __init__(self, gradient_lambda: float, network_factory: NetworkFactory, seed: int = None):
        self._gradient_lambda = gradient_lambda
        self._seed = seed
        self._network_factory = network_factory

    def create_discriminator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                                  generator_discriminator: tf.Tensor, real_discriminator: tf.Tensor,
                                  apply_summary: bool = True, use_gpu: bool = True,
                                  data_format: str = "NHWC") -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        with use_cpu():
            batch_size = tf.shape(x)[0]
            epsilons = tf.random_uniform((batch_size,), 0.0, 1.0, seed=self._seed)

            gradient_input = tf.multiply(epsilons, y) + tf.multiply(tf.ones_like(epsilons) - epsilons, generator)

        with tf.device(select_device(use_gpu)):
            gradient_discriminator, gradient_discriminator_input = self._network_factory.create_discriminator(
                x,
                gradient_input,
                reuse=True,
                return_input_tensor=True,
                use_gpu=use_gpu,
                data_format=data_format)

            # Generated samples should be close to their inverse jaccard index => Loss defined
            # Generator samples have to be first converted into range [0, 1]
            loss_generated = tf.constant(1.0) - tf.reduce_mean(jaccard_index(
                values=tanh_to_sigmoid(generator_discriminator),
                labels=tanh_to_sigmoid(y)
            ))

            # Real samples should all be 0 => No loss
            loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_discriminator,
                labels=tf.zeros_like(real_discriminator)
            ))

            gradient = tf.gradients(gradient_discriminator, gradient_discriminator_input)
            gradient_norm = tf.norm(gradient)
            gradient_penalty_raw = tf.reduce_mean((gradient_norm - tf.ones_like(gradient_norm)) ** 2)

        with use_cpu():
            gradient_penalty = tf.multiply(tf.constant(self._gradient_lambda, dtype=tf.float32), gradient_penalty_raw)
            loss = loss_generated + loss_real + gradient_penalty

        if apply_summary:
            summary_operations = [
                tf.summary.scalar("discriminator_loss_real", loss_real),
                tf.summary.scalar("discriminator_loss_generated", loss_generated),
                tf.summary.scalar("discriminator_gradient_penalty", gradient_penalty),
                tf.summary.scalar("discriminator_gradient_norm", gradient_norm),
                tf.summary.scalar("discriminator_loss", loss)
            ]

            return loss, summary_operations
        else:
            return loss

    def create_generator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                              generator_discriminator: tf.Tensor, apply_summary: bool = True, use_gpu: bool = True,
                              data_format: str = "NHWC") -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:

        with tf.device(select_device(use_gpu)):
            loss = tf.reduce_mean(
                # Generator should have 0 loss
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=generator_discriminator,
                    labels=tf.zeros_like(generator_discriminator))
            )

        if apply_summary:
            return loss, tf.summary.scalar("generator_loss", loss)
        else:
            return loss

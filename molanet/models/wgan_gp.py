import logging
from typing import List, Tuple, Union

import tensorflow as tf

from molanet.base import ObjectiveFactory, NetworkFactory
from molanet.operations import use_cpu, select_device

_log = logging.getLogger(__name__)


class WassersteinGradientPenaltyFactory(ObjectiveFactory):
    """
    Loss factory to create Wasserstein GAN loss with gradient penalty.
    """
    def __init__(self, gradient_lambda: float, network_factory: NetworkFactory,
                 l1_lambda: float = 0.0, seed: int = None):
        """
        Creates a new Wasserstein GAN loss factory.
        :param gradient_lambda: Lambda to multiply with discriminator gradient norm
        :param network_factory: Network factory used to create the correlating generators and discriminators
        :param l1_lambda: Lambda to multiply with the l1 distance between generated and real samples for the generator.
        If 0, no l1 loss will be used.
        :param seed: Seed which is used to initialize random values
        """
        self._gradient_lambda = gradient_lambda
        self._seed = seed
        self._network_factory = network_factory
        self._l1_lambda = l1_lambda

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

            loss_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_discriminator,
                labels=tf.zeros_like(generator_discriminator)
            ))

            loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_discriminator,
                labels=tf.ones_like(real_discriminator)
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
        summary_ops = []

        with tf.device(select_device(use_gpu)):
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=generator_discriminator,
                    labels=tf.ones_like(generator_discriminator))
            )

        if self._l1_lambda > 0.0:
            _log.info(f"Using l1 loss, lambda={self._l1_lambda}")

            with tf.device(select_device(use_gpu)):
                l1_loss = tf.reduce_mean(tf.abs(tf.subtract(y, generator)))

            with use_cpu():
                if apply_summary:
                    summary_ops.append(tf.summary.scalar("generator_loss_l1", l1_loss))
                    summary_ops.append(tf.summary.scalar("generator_loss_discriminator", loss))

                loss = loss + tf.constant(self._l1_lambda, dtype=tf.float32) * l1_loss

        if apply_summary:
            summary_ops.append(tf.summary.scalar("generator_loss", loss))
            return loss, summary_ops
        else:
            return loss

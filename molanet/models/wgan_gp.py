import tensorflow as tf

from molanet.base import ObjectiveFactory, NetworkFactory


class WassersteinGradientPenaltyFactory(ObjectiveFactory):

    def __init__(self, gradient_lambda: float, network_factory: NetworkFactory, seed: int = None):
        self._gradient_lambda = gradient_lambda
        self._seed = seed
        self._network_factory = network_factory

    def create_discriminator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                                  generator_discriminator: tf.Tensor, real_discriminator: tf.Tensor,
                                  apply_summary: bool = True) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        epsilons = tf.random_uniform((batch_size,), 0.0, 1.0, seed=self._seed)

        gradient_input = tf.multiply(epsilons, y) + tf.multiply(tf.ones_like(epsilons) - epsilons, generator)
        gradient_discriminator, gradient_discriminator_input = self._network_factory.create_discriminator(
            x,
            gradient_input,
            reuse=True,
            apply_summary=False,
            return_input_tensor=True)

        gradient = tf.gradients(gradient_discriminator, gradient_discriminator_input)
        gradient_norm = tf.norm(gradient)
        gradient_penalty = self._gradient_lambda * tf.reduce_mean((gradient_norm - tf.ones_like(gradient_norm)) ** 2)

        loss_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generator_discriminator,
            labels=tf.zeros_like(generator_discriminator)
        ))

        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_discriminator,
            labels=tf.ones_like(real_discriminator)
        ))

        loss = loss_generated + loss_real + gradient_penalty

        if apply_summary:
            tf.summary.scalar("discriminator_loss_real", loss_real)
            tf.summary.scalar("discriminator_loss_generated", loss_generated)
            tf.summary.scalar("discriminator_gradient_penalty", gradient_penalty)
            tf.summary.scalar("discriminator_gradient_norm", gradient_norm)
            tf.summary.scalar("discriminator_loss", loss)

        return loss

    def create_generator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                              generator_discriminator: tf.Tensor, apply_summary: bool = True) -> tf.Tensor:
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_discriminator,
                labels=tf.ones_like(generator_discriminator))
        )

        if apply_summary:
            tf.summary.scalar("generator_loss", loss)

        return loss

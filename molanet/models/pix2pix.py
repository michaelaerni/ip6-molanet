from typing import Union, Tuple

import tensorflow as tf

from molanet.base import NetworkFactory, ObjectiveFactory
from molanet.operations import leaky_relu


class Pix2PixFactory(NetworkFactory):

    def __init__(
            self,
            spatial_extent: int,
            weight_initializer=tf.truncated_normal_initializer(stddev=0.02)):
        import math

        if math.log2(spatial_extent) != int(math.log2(spatial_extent)):
            raise ValueError("spatial_extent must be a power of 2")

        self._spatial_extent = spatial_extent
        self._weight_initializer = weight_initializer

    def create_generator(self, x: tf.Tensor, reuse: bool = False, apply_summary: bool = True) -> tf.Tensor:
        with tf.variable_scope("generator", reuse=reuse):
            input_tensor = x
            encoder_activations = []
            layer_index = 0
            min_feature_count = 32
            feature_count = min_feature_count
            max_feature_count = 512
            layer_size = self._spatial_extent
            batch_size = tf.shape(x)[0]

            # Encoder
            while layer_size > 1:
                use_batchnorm = layer_index > 0 and layer_size // 2 > 1
                filter_sizes = 5 if layer_index == 0 else 4
                input_tensor, _, _ = self._conv2d(input_tensor, feature_count, f"enc_{layer_index}",
                                                  filter_size=filter_sizes, use_batchnorm=use_batchnorm)
                encoder_activations.append(input_tensor)
                feature_count = min(max_feature_count, feature_count * 2)
                layer_size = layer_size // 2
                layer_index += 1

            layer_count = layer_index

            # Decoder
            # TODO: Initial image is not concatenated
            for idx in range(layer_count):
                use_batchnorm = idx < layer_count - 1
                keep_probability = tf.constant(0.5) if idx < 3 else tf.constant(1.0) # TODO: When to use dropout
                encoder_index = layer_count - idx - 1 - 1
                target_layer_size = 2 ** (idx + 1)
                do_activation = encoder_index > 0
                filter_sizes = 4 if idx < layer_count - 1 else 5
                feature_count = min(max_feature_count, min_feature_count * (2 ** encoder_index))\
                    if encoder_index >= 0 else 1
                input_tensor, _, _ = self._conv2d_transpose(input_tensor, feature_count,
                                                            target_layer_size,
                                                    f"dec_{idx}", keep_probability, batch_size,
                                                            filter_size=filter_sizes,
                                                            concat_activations=encoder_activations[encoder_index] if encoder_index >= 0 else None,
                                                            use_batchnorm=use_batchnorm,
                                                            do_activation=do_activation)

            return tf.tanh(input_tensor, name="dec_activation")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True,
            return_input_tensor: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        with tf.variable_scope("discriminator", reuse=reuse):
            concatenated_input = tf.concat((x, y), axis=3)
            input_tensor = concatenated_input
            layer_size = self._spatial_extent
            feature_count = 32
            max_feature_count = 512
            layer_index = 0

            while layer_size > 1:
                filter_sizes = 5 if layer_index == 0 else 4
                input_tensor, _, _ = self._conv2d(input_tensor, feature_count, str(layer_index), use_batchnorm=False,
                                                  filter_size=filter_sizes, do_activation=layer_size // 2 > 1)
                layer_size = layer_size // 2
                feature_count = min(max_feature_count, feature_count * 2) if layer_size // 2 > 1 else 1
                layer_index += 1

            if return_input_tensor:
                return input_tensor, concatenated_input
            else:
                return input_tensor

    def _weight_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=self._weight_initializer)

    def _bias_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

    def _conv2d(self, features, feature_count, name, filter_size=5, use_batchnorm=True, stride=2, do_activation=True):
        w = self._weight_variable("w_" + name, [filter_size, filter_size, features.get_shape()[-1], feature_count])
        b = self._bias_variable("b_" + name, [feature_count])
        conv = tf.nn.bias_add(tf.nn.conv2d(features, w, strides=[1, stride, stride, 1], padding="SAME"),
                              b)  # TODO: Padding?
        if use_batchnorm:
            bn = tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5)  # TODO: Params?
        else:
            bn = conv

        if do_activation:
            a = leaky_relu(bn, 0.2)
        else:
            a = bn

        return a, w, b

    def _conv2d_transpose(self, features, feature_count, output_size, name, keep_prob, batch_size, filter_size=5,
                          concat_activations=None, use_batchnorm=True, do_activation=True):
        w = self._weight_variable("w_" + name, [filter_size, filter_size, feature_count, features.get_shape()[-1]])
        b = self._bias_variable("b_" + name, [feature_count])
        conv = tf.nn.bias_add(
            tf.nn.conv2d_transpose(features, w, output_shape=[batch_size, output_size, output_size, feature_count],
                                   strides=[1, 2, 2, 1], padding="SAME"), b)  # TODO: Padding?
        if use_batchnorm:
            bn = tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5)  # TODO: Params?
        else:
            bn = conv
        d = tf.nn.dropout(bn, keep_prob)

        if do_activation:
            a = tf.nn.relu(d)
        else:
            a = d

        # Concat activations if available
        if concat_activations is not None:
            a = tf.concat([a, concat_activations], axis=3)

        return a, w, b


class Pix2PixLossFactory(ObjectiveFactory):

    def __init__(self, l1_lambda: float):
        self._l1_lambda = tf.constant(l1_lambda, dtype=tf.float32)

    def create_discriminator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                                  generator_discriminator: tf.Tensor, real_discriminator: tf.Tensor,
                                  apply_summary: bool = True) -> tf.Tensor:
        loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_discriminator,
                labels=tf.ones_like(generator_discriminator))
        )
        loss_generated = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_discriminator,
                labels=tf.zeros_like(generator_discriminator))
        )

        loss = loss_real + loss_generated

        if apply_summary:
            tf.summary.scalar("discriminator_loss", loss)
            tf.summary.scalar("discriminator_loss_real", loss_real)
            tf.summary.scalar("discriminator_loss_generated", loss_generated)

        return loss

    def create_generator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                              generator_discriminator: tf.Tensor, apply_summary: bool = True) -> tf.Tensor:
        loss_discriminator = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_discriminator,
                labels=tf.ones_like(generator_discriminator))
        )
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(y, generator)))

        loss = loss_discriminator + self._l1_lambda * l1_loss

        if apply_summary:
            tf.summary.scalar("generator_loss", loss)
            tf.summary.scalar("generator_loss_discriminator", loss_discriminator)
            tf.summary.scalar("generator_loss_l1_unscaled", l1_loss)

        return loss

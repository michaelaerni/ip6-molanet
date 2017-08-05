import math
from typing import Union, List

from molanet.base import NetworkFactory, ObjectiveFactory
from molanet.operations import *


class Pix2PixFactory(NetworkFactory):

    def __init__(
            self,
            spatial_extent: int,
            min_generator_features: int = 32,
            min_discriminator_features: int = 32,
            max_generator_features: int = 512,
            max_discriminator_features: int = 512,
            dropout_keep_probability: float = 0.5,
            dropout_layer_count: int = 2,
            use_batchnorm: bool = True,
            weight_initializer=tf.truncated_normal_initializer(stddev=0.02)):

        # TODO: weight_initializer is currently ignored

        if math.log2(spatial_extent) != int(math.log2(spatial_extent)):
            raise ValueError("spatial_extent must be a power of 2")

        self._spatial_extent = spatial_extent
        self._min_generator_features = min_generator_features
        self._min_discriminator_features = min_discriminator_features
        self._max_generator_features = max_generator_features
        self._max_discriminator_features = max_discriminator_features
        self._dropout_layer_count = dropout_layer_count
        self._dropout_keep_probability = tf.constant(dropout_keep_probability)
        self._use_batchnorm = use_batchnorm
        self._weight_initializer = weight_initializer

    def create_generator(self, x: tf.Tensor, reuse: bool = False, use_gpu: bool = True, data_format: str = "NHWC") -> tf.Tensor:
        with tf.variable_scope("generator", reuse=reuse), tf.device(select_device(use_gpu)):
            input_tensor = x
            encoder_activations = []
            layer_index = 0
            feature_count = self._min_generator_features
            layer_size = self._spatial_extent

            # Encoder
            while layer_size > 1:
                use_batchnorm = self._use_batchnorm and (layer_index > 0 and layer_size // 2 > 1)
                filter_sizes = 5 if layer_index == 0 else 4
                input_tensor, _, _ = conv2d(
                    input_tensor,
                    feature_count,
                    f"enc_{layer_index}",
                    filter_size=filter_sizes,
                    stride=2,
                    do_batchnorm=use_batchnorm,
                    data_format=data_format)
                encoder_activations.append(input_tensor)
                feature_count = min(self._max_generator_features, feature_count * 2)
                layer_size = layer_size // 2
                layer_index += 1

            layer_count = layer_index

            # Decoder
            # TODO: Initial image is not concatenated
            for idx in range(layer_count):
                use_batchnorm = self._use_batchnorm and idx < layer_count - 1
                keep_probability = self._dropout_keep_probability \
                    if idx < self._dropout_layer_count else tf.constant(1.0)
                encoder_index = layer_count - idx - 1 - 1
                target_layer_size = 2 ** (idx + 1)
                do_activation = encoder_index > 0
                filter_sizes = 4 if idx < layer_count - 1 else 5
                feature_count = min(self._max_generator_features, self._min_discriminator_features * (2 ** encoder_index))\
                    if encoder_index >= 0 else 1
                input_tensor, _, _ = conv2d_transpose(
                    input_tensor,
                    feature_count,
                    f"dec_{idx}",
                    filter_size=filter_sizes,
                    output_shape_2d=(target_layer_size, target_layer_size),
                    keep_probability=keep_probability,
                    concat_activations=encoder_activations[encoder_index] if encoder_index >= 0 else None,
                    stride=2,
                    do_batchnorm=use_batchnorm,
                    do_activation=do_activation,
                    data_format=data_format)

            return tf.tanh(input_tensor, name="dec_activation")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            return_input_tensor: bool = False,
            use_gpu: bool = True, data_format: str = "NHWC") -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        with tf.variable_scope("discriminator", reuse=reuse), tf.device(select_device(use_gpu)):
            concat_axis = 3 if data_format == "NHWC" else 1

            concatenated_input = tf.concat((x, y), axis=concat_axis)
            input_tensor = concatenated_input
            layer_size = self._spatial_extent
            feature_count = self._min_discriminator_features
            layer_index = 0

            while layer_size > 1:
                filter_sizes = 5 if layer_index == 0 else 4
                input_tensor, _, _ = conv2d(
                    input_tensor,
                    feature_count,
                    str(layer_index),
                    filter_size=filter_sizes,
                    stride=2,
                    do_batchnorm=False,
                    do_activation=layer_size // 2 > 1,
                    data_format=data_format)
                layer_size = layer_size // 2
                feature_count = min(self._max_discriminator_features, feature_count * 2) if layer_size // 2 > 1 else 1
                layer_index += 1

            if return_input_tensor:
                return input_tensor, concatenated_input
            else:
                return input_tensor


class Pix2PixLossFactory(ObjectiveFactory):

    def __init__(self, l1_lambda: float):
        self._l1_lambda = tf.constant(l1_lambda, dtype=tf.float32)

    def create_discriminator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                                  generator_discriminator: tf.Tensor, real_discriminator: tf.Tensor,
                                  apply_summary: bool = True, use_gpu: bool = True,
                                  data_format: str = "NHWC") -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        with tf.device(select_device(use_gpu)):
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
            summary_ops = [
                tf.summary.scalar("discriminator_loss", loss),
                tf.summary.scalar("discriminator_loss_real", loss_real),
                tf.summary.scalar("discriminator_loss_generated", loss_generated)
            ]

            return loss, summary_ops
        else:
            return loss

    def create_generator_loss(self, x: tf.Tensor, y: tf.Tensor, generator: tf.Tensor,
                              generator_discriminator: tf.Tensor,
                              apply_summary: bool = True, use_gpu: bool = True,
                              data_format: str = "NHWC") -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        with tf.device(select_device(use_gpu)):
            loss_discriminator = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=generator_discriminator,
                    labels=tf.ones_like(generator_discriminator))
            )
            l1_loss = tf.reduce_mean(tf.abs(tf.subtract(y, generator)))

        loss = loss_discriminator + self._l1_lambda * l1_loss

        if apply_summary:
            summary_ops = [
                tf.summary.scalar("generator_loss", loss),
                tf.summary.scalar("generator_loss_discriminator", loss_discriminator),
                tf.summary.scalar("generator_loss_l1_unscaled", l1_loss)
            ]

            return loss, summary_ops
        else:
            return loss

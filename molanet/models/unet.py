import logging
import math
from typing import Union, List

from molanet.base import NetworkFactory, ObjectiveFactory
from molanet.operations import *

_log = logging.getLogger(__name__)


class UnetFactory(NetworkFactory):

    def __init__(
            self,
            spatial_extent: int,
            convolutions_per_level: int = 1,
            min_discriminator_features: int = 32,
            max_discriminator_features: int = 512,
            dropout_keep_probability: float = 0.5,
            dropout_layer_count: int = 2,
            use_batchnorm: bool = True,
            weight_initializer=tf.truncated_normal_initializer(stddev=0.02)):

        # TODO: weight_initializer is currently ignored

        if math.log2(spatial_extent) != int(math.log2(spatial_extent)):
            raise ValueError("spatial_extent must be a power of 2")

        self._spatial_extent = spatial_extent
        self._convolutions_per_level = convolutions_per_level
        self._min_discriminator_features = min_discriminator_features
        self._max_discriminator_features = max_discriminator_features
        self._dropout_layer_count = dropout_layer_count
        self._dropout_keep_probability = tf.constant(dropout_keep_probability)
        self._use_batchnorm = use_batchnorm
        self._weight_initializer = weight_initializer

    def create_generator(self, x: tf.Tensor, reuse: bool = False, use_gpu: bool = True, data_format: str = "NHWC") -> tf.Tensor:
        with tf.variable_scope("generator", reuse=reuse), tf.device(select_device(use_gpu)):
            feature_counts = [64, 128, 256, 512, 1024]
            current_layer = x
            encoder_level_layers = []
            encoder_level_shapes = []

            # TODO: Check where to use batch norm

            # Create encoder, level by level
            for level, feature_count in enumerate(feature_counts[:-1]):
                _log.debug(f"Level: {level}; Features: {feature_count}")
                _log.debug(f"Incoming: {current_layer.get_shape()}")

                # Convolve n times and keep size
                for conv_idx in range(self._convolutions_per_level):
                    current_layer, _, _ = conv2d(
                        current_layer,
                        feature_count,
                        f"enc_{level}_{conv_idx}",
                        filter_size=3,
                        stride=1,
                        do_batchnorm=True,
                        data_format=data_format)

                # Store current layer for skip connection
                encoder_level_layers.append(current_layer)
                if data_format == "NHWC":
                    level_shape = tf.shape(current_layer)[1], tf.shape(current_layer)[2]
                elif data_format == "NCHW":
                    level_shape = tf.shape(current_layer)[2], tf.shape(current_layer)[3]
                else:
                    raise ValueError(f"Unsupported data format {data_format}")

                encoder_level_shapes.append(level_shape)

                # Downsample
                current_layer, _, _ = conv2d(
                    current_layer,
                    feature_count,
                    f"enc_{level}_down",
                    filter_size=3,
                    stride=2,
                    do_batchnorm=True,
                    data_format=data_format)

            # Perform middle convolution
            _log.debug(f"Before middle: {current_layer.get_shape()}")
            current_layer, _, _ = conv2d(
                current_layer,
                feature_counts[-1],
                f"middle_in",
                filter_size=3,
                stride=1,
                do_batchnorm=True,
                data_format=data_format)
            _log.debug(f"Middle: {current_layer.get_shape()}")
            current_layer, _, _ = conv2d(
                current_layer,
                feature_counts[-1],
                f"middle_out",
                filter_size=3,
                stride=1,
                do_batchnorm=True,
                data_format=data_format)
            _log.debug(f"After middle: {current_layer.get_shape()}")

            # Now create decoder level by level
            for level, feature_count in reversed(list(enumerate(feature_counts[:-1]))):
                _log.debug(f"Level: {level}; Features: {feature_count}")

                # Upsample and concatenate
                current_layer, _, _ = conv2d_transpose(
                    current_layer,
                    feature_count,
                    f"dec_{level}_up",
                    filter_size=3,
                    output_shape_2d=encoder_level_shapes[level],
                    stride=2,
                    do_batchnorm=True,
                    concat_activations=encoder_level_layers[level],
                    data_format=data_format)
                _log.debug(f"Incoming: {current_layer.get_shape()}")

                # Convolve n times and keep size
                for conv_idx in range(self._convolutions_per_level):
                    current_layer, _, _ = conv2d(
                        current_layer,
                        feature_count,
                        f"dec_{level}_{conv_idx}",
                        filter_size=3,
                        stride=1,
                        do_batchnorm=True,
                        data_format=data_format)

            # Perform final convolution
            output_layer, _, _ = conv2d(
                current_layer,
                feature_count=1,
                name=f"out",
                filter_size=1,
                stride=1,
                do_batchnorm=False,  # TODO: Use batch norm here or not?
                do_activation=False,
                data_format=data_format)

            return tf.tanh(output_layer, name="dec_activation")

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

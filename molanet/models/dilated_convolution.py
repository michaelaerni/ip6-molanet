import math
from typing import Union

from molanet.base import NetworkFactory
from molanet.operations import *


class DilatedConvolutionFactory(NetworkFactory):
    def __init__(
            self,
            spatial_extent: int,
            min_discriminator_features: int = 32,
            max_discriminator_features: int = 512):

        if math.log2(spatial_extent) != int(math.log2(spatial_extent)):
            raise ValueError("spatial_extent must be a power of 2")

        self._spatial_extent = spatial_extent
        self._min_discriminator_features = min_discriminator_features
        self._max_discriminator_features = max_discriminator_features

    def create_discriminator(self, x: tf.Tensor, y: tf.Tensor, reuse: bool = False, return_input_tensor: bool = False,
                             use_gpu: bool = True,
                             data_format: str = "NHWC") -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
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
                    name=str(layer_index),
                    filter_size=filter_sizes,
                    stride=2,
                    do_batchnorm=False,
                    do_activation=layer_size // 2 > 1,
                    data_format=data_format
                )

                layer_size = layer_size // 2
                feature_count = min(self._max_discriminator_features, feature_count * 2) if layer_size // 2 > 1 else 1
                layer_index += 1

            if return_input_tensor:
                return input_tensor, concatenated_input
            else:
                return input_tensor

    def create_generator(self, x: tf.Tensor, reuse: bool = False,
                         use_gpu: bool = True, data_format: str = "NHWC") -> tf.Tensor:
        with tf.variable_scope("generator", reuse=reuse), tf.device(select_device(use_gpu)):
            # TODO: First try, use only dilated convolutions to see what happens
            input_feature_count = x.get_shape()[-1]

            # Create pre-filtering
            feature_multipliers = [1, 1, 2, 2, 4, 4, 8]
            current_layer = x
            for layer_idx, feature_count in enumerate([input_feature_count * m for m in feature_multipliers]):
                current_layer, _, _ = conv2d(
                    current_layer,
                    feature_count=feature_count,
                    name=f"pre_{layer_idx}",
                    filter_size=3,
                    data_format=data_format
                )
            pre_filtered = current_layer

            dilation_exponents = [0, 0, 1, 2, 3, 4, 5, 6, 0]
            feature_multipliers = [2, 2, 4, 8, 16, 16, 16, 16, 32]

            # Create dilated layers
            # TODO: Might fail due to padding
            current_layer = pre_filtered #x
            for layer_idx, (dilation_factor, feature_count) in enumerate(zip(
                    (2 ** e for e in dilation_exponents),
                    (input_feature_count * m for m in feature_multipliers))):

                current_layer, _, _ = conv2d_dilated(
                    current_layer,
                    feature_count=feature_count,
                    dilation_rate=dilation_factor,
                    name=f"context_{layer_idx}",
                    filter_size=3,
                    data_format=data_format
                )
            context = current_layer

            concat_axis = 3 if data_format == "NHWC" else 1
            concatenated_features = tf.concat([pre_filtered, context], axis=concat_axis)

            # Perform 1x1 convolution to produce output
            result, _, _ = conv2d(
                concatenated_features,
                feature_count=1,
                name="output",
                filter_size=1,
                do_batchnorm=False,
                do_activation=False,
                data_format=data_format
            )
            result = tf.nn.tanh(result)

            return result

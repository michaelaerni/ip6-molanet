import logging
from typing import Union

from molanet.base import NetworkFactory
from molanet.operations import *

_log = logging.getLogger(__name__)


class BigDiscPix2Pix(NetworkFactory):
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
            use_layernorm: bool = False,
            weight_initializer=tf.truncated_normal_initializer(stddev=0.02),
            multiply_mask: bool = False):

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
        self._use_layer_norm = use_layernorm
        self._weight_initializer = weight_initializer
        self._multiply_mask = multiply_mask

    def create_generator(self, x: tf.Tensor, reuse: bool = False, use_gpu: bool = True,
                         data_format: str = "NHWC") -> tf.Tensor:
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
                feature_count = min(self._max_generator_features,
                                    self._min_discriminator_features * (2 ** encoder_index)) \
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

    @classmethod
    def _leaky_relu_func(cls, features):
        return tf.maximum(0.2 * features, features)

    @classmethod
    def _layer_norm(cls, features, do_activation: bool = False):
        if do_activation:
            layer_normed = tf.contrib.layers.layer_norm(features)
            return leaky_relu(layer_normed, 0.2)
        else:
            return tf.contrib.layers.layer_norm(features)

    @classmethod
    def _conv_act_convstride2(cls, features: tf.Tensor, depth_maps: int, filter_size: int, data_format: str,
                              name: str,
                              dropout_keep_prob: float = None,
                              layer_norm: bool = False):
        conv, _, _ = conv2d(features, depth_maps, f"{name}",
                            filter_size,
                            stride=1,
                            do_batchnorm=False,
                            do_activation=False if layer_norm else True,
                            data_format=data_format,
                            weight_initializer=tf.uniform_unit_scaling_initializer(1.43))

        if dropout_keep_prob is not None:
            conv = tf.nn.dropout(conv, dropout_keep_prob, name=f"{name}_dropout")

        if layer_norm:
            # TODO reuse?
            conv = cls._layer_norm(conv, do_activation=True)

        strided, _, _ = conv2d(conv, depth_maps, f"{name}_strided",
                               filter_size,
                               stride=2,
                               do_batchnorm=False,
                               do_activation=False if layer_norm else True,
                               data_format=data_format)

        if layer_norm:
            # TODO reuse?
            strided = cls._layer_norm(strided, do_activation=True)

        return strided

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            return_input_tensor: bool = False,
            use_gpu: bool = True,
            data_format: str = "NHWC"
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        with tf.variable_scope("discriminator", reuse=reuse), tf.device(select_device(use_gpu)):
            concat_axis = 3 if data_format == "NHWC" else 1

            """
            heavily inspired by facebook AI research 2016 paper
            'Semantic Segmentation using Adversarial Networks'

            Stanford background dataset
            """

            # mask pipeline
            if self._multiply_mask:
                y = tf.multiply(y, x)
                _log.debug(f"y*x shape: {y.get_shape()}")

            mask, _, _ = conv2d(y, 64, "disc_y_conv", 5, 1, do_batchnorm=False,
                                do_activation=False if self._use_layer_norm else True,
                                data_format=data_format,
                                weight_initializer=tf.uniform_unit_scaling_initializer(1.43))
            if self._use_layer_norm:
                mask = self._layer_norm(mask, do_activation=True)

            # image pipeline
            x1, _, _ = conv2d(x, 16, "x1", 5, stride=1, do_batchnorm=False,
                              do_activation=False if self._use_layer_norm else True,
                              data_format=data_format,
                              weight_initializer=tf.uniform_unit_scaling_initializer(1.43))
            if self._use_layer_norm:
                x1 = self._layer_norm(x1, do_activation=True)

            x2, _, _ = conv2d(x1, 64, "x2", 5, stride=1, do_batchnorm=False,
                              do_activation=False if self._use_layer_norm else True,
                              data_format=data_format,
                              weight_initializer=tf.uniform_unit_scaling_initializer(1.43))
            if self._use_layer_norm:
                x2 = self._layer_norm(x2, do_activation=True)

            # concat
            xy = tf.concat([x2, mask], axis=concat_axis)

            # downsample to 1x1
            layer_size = self._spatial_extent
            k = 0
            xy_k = xy
            depth_map_count = self._min_discriminator_features
            while layer_size > 1:
                k += 1
                # use dropout on conv layers
                # increasing probability to drop
                # 0.1 0.2 0.3

                xy_k = self._conv_act_convstride2(xy_k,
                                                  depth_maps=depth_map_count,
                                                  filter_size=3,
                                                  data_format=data_format,
                                                  name=f"xy{k}",
                                                  layer_norm=self._use_layer_norm)
                layer_size = layer_size // 2
                depth_map_count = depth_map_count * 2 if depth_map_count < self._max_discriminator_features \
                    else self._max_discriminator_features

            output, _, _ = conv2d(xy_k, 1, "final", 1, 1, do_batchnorm=False, do_activation=False,
                                  data_format=data_format,
                                  weight_initializer=tf.uniform_unit_scaling_initializer(1.15))

            if return_input_tensor:
                return output, xy
            else:
                return output

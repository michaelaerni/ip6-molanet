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
            use_batchnorm: bool = True
    ):

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

            # Has to be concatenated here in order for tf.gradients to work
            concatenated_input = tf.concat((x, y), axis=concat_axis)
            if data_format == "NHWC":
                x = concatenated_input[:, :, :, :3]
                y = concatenated_input[:, :, :, 3:]
            elif data_format == "NCHW":
                x = concatenated_input[:, :3, :, :]
                y = concatenated_input[:, 3:, :, :]
            else:
                raise ValueError(f"Unsupported data format {data_format}")

            # Multiply mask input
            # Concatenate y as some parts of x could be zero, thus bad masks in black input areas would be ignored
            multiplied = tf.multiply(y, x)
            multiplied = tf.concat([multiplied, y], concat_axis)
            y = multiplied

            # Convolve mask branch once
            mask, _, _ = conv2d(
                y,
                feature_count=32,
                name="disc_y_conv",
                filter_size=5,
                stride=1,
                do_batchnorm=False,
                do_activation=True,
                data_format=data_format,
                padding="REFLECT",
                weight_initializer=tf.uniform_unit_scaling_initializer(1.43))

            # Convolve original image branch twice
            x1, _, _ = conv2d(
                x,
                feature_count=16,
                name="x1",
                filter_size=5,
                stride=1,
                do_batchnorm=False,
                do_activation=True,
                data_format=data_format,
                padding="REFLECT",
                weight_initializer=tf.uniform_unit_scaling_initializer(1.43))
            x2, _, _ = conv2d(
                x1,
                feature_count=32,
                name="x2",
                filter_size=5,
                stride=1,
                do_batchnorm=False,
                do_activation=True,
                data_format=data_format,
                padding="REFLECT",
                weight_initializer=tf.uniform_unit_scaling_initializer(1.43))

            # Concat input branches
            xy = tf.concat([x2, mask], axis=concat_axis, name="concat_input_branches")

            # Downsample to 1x1
            xy_k = xy
            depth_map_count = self._min_discriminator_features
            for k in range(9):  # 2^9 = 512

                # Produce bigger feature map for second to last level
                if k == 8:
                    depth_map_count *= 2

                _log.debug(f"Level {k} input: Shape={xy_k.get_shape()}, Features={depth_map_count}")

                # Convolve
                conv, _, _ = conv2d(
                    xy_k,
                    feature_count=depth_map_count,
                    name=f"xy_{k}",
                    filter_size=3,
                    stride=1,
                    do_batchnorm=False,
                    do_activation=True,
                    data_format=data_format,
                    padding="REFLECT",
                    weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
                )

                # Downsample
                xy_k, _, _ = conv2d(
                    conv,
                    feature_count=depth_map_count,
                    name=f"xy{k}_strided",
                    filter_size=3,
                    stride=2,
                    do_batchnorm=False,
                    do_activation=True,
                    data_format=data_format,
                    weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
                )

                _log.debug(f"Level {k} output: Shape={xy_k.get_shape()}")

                depth_map_count = min(self._max_discriminator_features, depth_map_count * 2)

            output, _, _ = conv2d(
                xy_k,
                feature_count=1,
                name="output",
                filter_size=1,
                stride=1,
                do_batchnorm=False,
                do_activation=False,
                data_format=data_format,
                padding="VALID",
                weight_initializer=tf.uniform_unit_scaling_initializer(1.0))

            if return_input_tensor:
                return output, concatenated_input
            else:
                return output

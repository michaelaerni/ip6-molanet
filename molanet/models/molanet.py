import logging
from typing import Union

from molanet.base import NetworkFactory, ObjectiveFactory
from molanet.operations import *

_log = logging.getLogger(__name__)


class MolanetFactory(NetworkFactory):

    def __init__(
            self,
            convolutions_per_level: int = 2,
            min_discriminator_features: int = 64,
            max_discriminator_features: int = 512
    ):
        self._convolutions_per_level = convolutions_per_level
        self._min_discriminator_features = min_discriminator_features
        self._max_discriminator_features = max_discriminator_features

    def create_generator(self, x: tf.Tensor, reuse: bool = False, use_gpu: bool = True, data_format: str = "NHWC") -> tf.Tensor:
        with tf.variable_scope("generator", reuse=reuse), tf.device(select_device(use_gpu)):
            feature_counts = [64, 128, 256, 512, 1024]
            current_layer = x
            encoder_level_layers = []
            encoder_level_shapes = []

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
                        padding="REFLECT",
                        do_batchnorm=True,
                        data_format=data_format,
                        weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
                    )

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
                    padding="REFLECT",
                    do_batchnorm=True,
                    data_format=data_format,
                    weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
                )

            # Perform middle convolution
            _log.debug(f"Before middle: {current_layer.get_shape()}")
            current_layer, _, _ = conv2d(
                current_layer,
                feature_counts[-1],
                f"middle_in",
                filter_size=3,
                stride=1,
                padding="REFLECT",
                do_batchnorm=True,
                data_format=data_format,
                weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
            )
            _log.debug(f"Middle: {current_layer.get_shape()}")
            current_layer, _, _ = conv2d(
                current_layer,
                feature_counts[-1],
                f"middle_out",
                filter_size=3,
                stride=1,
                padding="REFLECT",
                do_batchnorm=True,
                data_format=data_format,
                weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
            )
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
                    data_format=data_format,
                    weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
                )
                _log.debug(f"Incoming: {current_layer.get_shape()}")

                # Convolve n times and keep size
                for conv_idx in range(self._convolutions_per_level):
                    current_layer, _, _ = conv2d(
                        current_layer,
                        feature_count,
                        f"dec_{level}_{conv_idx}",
                        filter_size=3,
                        stride=1,
                        padding="REFLECT",
                        do_batchnorm=True,
                        data_format=data_format,
                        weight_initializer=tf.uniform_unit_scaling_initializer(1.43)
                    )

            # Perform final convolution
            output_layer, _, _ = conv2d(
                current_layer,
                feature_count=1,
                name=f"out",
                filter_size=1,
                stride=1,
                do_batchnorm=False,
                do_activation=False,
                padding="VALID",  # VALID as 1x1 convolutions always result in same size
                data_format=data_format,
                weight_initializer=tf.uniform_unit_scaling_initializer(1.15)
            )

            return tf.tanh(output_layer, name="dec_activation")

    def create_discriminator(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            reuse: bool = False,
            return_input_tensor: bool = False,
            use_gpu: bool = True, data_format: str = "NHWC"
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

class MolanetLossFactory(ObjectiveFactory):

    def __init__(
            self,
            gradient_lambda: float,
            network_factory: NetworkFactory,
            use_jaccard: bool,
            l1_lambda: float = 0.0,
            seed: int = None
    ):
        self._gradient_lambda = gradient_lambda
        self._seed = seed
        self._network_factory = network_factory
        self._use_jaccard = use_jaccard
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

            if self._use_jaccard:
                # Generated samples should be close to their inverse jaccard index => Loss defined
                # Generator samples have to be first converted into range [0, 1]
                generated_jaccard_loss = tf.constant(1.0) - jaccard_index(
                    values=tanh_to_sigmoid(generator),
                    labels=tanh_to_sigmoid(y)
                )
                loss_generated = tf.reduce_mean((generated_jaccard_loss - generator_discriminator) ** 2 / 2.0)

                # Real samples should all be 0 => No loss
                loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=real_discriminator,
                    labels=tf.zeros_like(real_discriminator)
                ))
            else:
                # Use traditional loss
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
            labels = tf.zeros_like(generator_discriminator) \
                if self._use_jaccard \
                else tf.ones_like(generator_discriminator)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=generator_discriminator,
                    labels=labels)
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

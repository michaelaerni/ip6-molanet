import tensorflow as tf
from molanet.base import NetworkFactory
from molanet.operations import leaky_relu


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))


def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def conv2d(features, feature_count, name, use_batchnorm=True, stride=2, do_activation=True):
    w = weight_variable("w_" + name, [4, 4, features.get_shape()[-1], feature_count])
    b = bias_variable("b_" + name, [feature_count])
    conv = tf.nn.bias_add(tf.nn.conv2d(features, w, strides=[1, stride, stride, 1], padding="SAME"), b)  # TODO: Padding?
    if use_batchnorm:
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5)  # TODO: Params?
    else:
        bn = conv

    if do_activation:
        a = leaky_relu(bn, 0.2)
    else:
        a = bn

    return a, w, b


def conv2d_transpose(features, feature_count, output_size, name, keep_prob, batch_size, concat_activations=None, use_batchnorm=True, do_activation=True):

    w = weight_variable("w_" + name, [3, 3, feature_count, features.get_shape()[-1]])
    b = bias_variable("b_" + name, [feature_count])
    conv = tf.nn.bias_add(tf.nn.conv2d_transpose(features, w, output_shape=[batch_size, output_size, output_size, feature_count], strides=[1, 2, 2, 1], padding="SAME"), b)  # TODO: Padding?
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


class Pix2PixFactory(NetworkFactory):

    def __init__(self, spatial_extent: int):
        import math

        if math.log2(spatial_extent) != int(math.log2(spatial_extent)):
            raise ValueError("spatial_extent must be a power of 2")

        self._spatial_extent = spatial_extent

    def create_generator(self, source_tensor: tf.Tensor, reuse: bool = False, apply_summary: bool = True) -> tf.Tensor:
        with tf.variable_scope("generator", reuse=reuse):
            input_tensor = source_tensor
            encoder_activations = []
            layer_index = 0
            min_feature_count = 32
            feature_count = min_feature_count
            max_feature_count = 512
            layer_size = self._spatial_extent
            batch_size = tf.shape(source_tensor)[0]

            # Encoder
            while layer_size > 1:
                use_batchnorm = True # TODO: Correct batch norm usage
                input_tensor, _, _ = conv2d(input_tensor, feature_count, f"enc_{layer_index}",
                                                  use_batchnorm=use_batchnorm)
                encoder_activations.append(input_tensor)
                feature_count = min(max_feature_count, feature_count * 2)
                layer_size = layer_size // 2
                layer_index += 1

            layer_count = layer_index

            # Decoder
            # TODO: Initial image is not concatenated
            for idx in range(layer_count):
                use_batchnorm = True # TODO: When to use batchnorm
                keep_probability = tf.constant(0.5) if idx < 3 else tf.constant(1.0) # TODO: When to use dropout
                encoder_index = layer_count - idx - 1 - 1
                target_layer_size = 2 ** (idx + 1)
                do_activation = encoder_index > 0
                feature_count = min(max_feature_count, min_feature_count * (2 ** encoder_index))\
                    if encoder_index >= 0 else 3
                input_tensor, _, _ = conv2d_transpose(input_tensor, feature_count,
                                                    target_layer_size,
                                                    f"dec_{idx}", keep_probability, batch_size,
                                                    encoder_activations[encoder_index] if encoder_index >= 0 else None,
                                                    use_batchnorm=use_batchnorm,
                                                    do_activation=do_activation)

            return tf.tanh(input_tensor, name="dec_activation")

    def create_discriminator(
            self,
            source_tensor: tf.Tensor,
            target_tensor: tf.Tensor,
            reuse: bool = False,
            apply_summary: bool = True) -> tf.Tensor:
        with tf.variable_scope("discriminator", reuse=reuse):
            input_tensor = tf.concat((source_tensor, target_tensor), axis=3)
            layer_size = self._spatial_extent
            feature_count = 32
            max_feature_count = 512
            layer_index = 0

            while layer_size > 1:
                input_tensor, _, _ = conv2d(input_tensor, feature_count, str(layer_index), use_batchnorm=False,
                                      do_activation=layer_size // 2 > 1)
                layer_size = layer_size // 2
                feature_count = min(max_feature_count, feature_count * 2) if layer_size // 2 > 1 else 1
                layer_index += 1

            return input_tensor

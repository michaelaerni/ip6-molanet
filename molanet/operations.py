from typing import Callable, Tuple

import tensorflow as tf


# TODO: Docstrings


def leaky_relu(features, alpha=0.0):
    return tf.maximum(alpha * features, features)


def select_device(use_gpu: bool) -> Callable[[tf.Operation], str]:
    def _selector(op: tf.Operation) -> str:
        # Do not assign device placement for variables as it breaks Tensorflow somehow
        if op.type == "VariableV2":
            return ""

        return "/gpu:0" if use_gpu else "/cpu:0"
    return _selector


def use_cpu():
    return tf.device("/cpu:0")


def jaccard_index(labels: tf.Tensor, values: tf.Tensor):
    # TODO: Document: how and why jacquard
    # TODO: Could this be used with logits?
    # TODO: Document: Labels have to be either 0 or 1

    batch_size = tf.shape(labels)[0]

    # Reshape inputs to only have one dimension left, first one is considered batch
    labels = tf.reshape(labels, [batch_size, -1])
    values = tf.reshape(values, [batch_size, -1])

    # Intersection is point wise multiplication
    intersection = tf.multiply(labels, values)

    # Union is the clipped sum
    union = tf.clip_by_value(tf.add(labels, values), 0.0, 1.0)

    # TODO: Is dividing element-wise better for numerical stability? It is for sure slower...
    intersection_sum = tf.reduce_sum(intersection, axis=1)
    union_sum = tf.reduce_sum(union, axis=1)

    # Handle cases where whether labels nor values contain any positives => Are same set, therefore index is 1
    # TODO: Is this assumption correct?
    return tf.where(
        tf.greater(union_sum, 0),
        tf.divide(intersection_sum, union_sum),  # Use normal loss formulation
        tf.ones_like(intersection_sum))  # Both sets contain no values, therefore they are equal


def conv2d_dilated(
        input_tensor: tf.Tensor,
        feature_count: int,
        dilation_rate: int,
        name: str,
        filter_size: int = 3,
        data_format: str = "NHWC"
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    with tf.name_scope(f"conv2d_dilated_{name}"):

        # Check data format and select input feature count
        tf.assert_rank(input_tensor, 4)
        input_shape = input_tensor.get_shape()
        if data_format == "NHWC":
            input_feature_count = input_shape[-1]
        elif data_format == "NCHW":
            input_feature_count = input_shape[1]
        else:
            raise ValueError(f"Unsupported data format {data_format}")

        # Create variables
        w = tf.get_variable(
            f"w_{name}",
            shape=[filter_size, filter_size, input_feature_count, feature_count],
            dtype=tf.float32,
            initializer=tf.uniform_unit_scaling_initializer(1.43))  # TODO: Use correct initializer
        b = tf.get_variable(f"b_{name}",
                            shape=[feature_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        # Perform convolution
        conv = tf.nn.convolution(
            input_tensor,
            filter=w,
            padding="SAME",  # TODO: Make padding configurable
            dilation_rate=[dilation_rate, dilation_rate],
            data_format=data_format)

        result = tf.nn.bias_add(conv, b, data_format=data_format)
        result = tf.nn.relu(result)

        return result, w, b


def conv2d_transpose(
        input_tensor: tf.Tensor,
        feature_count: int,
        name: str,
        filter_size: int,
        output_shape_2d: Tuple[tf.Tensor, tf.Tensor],
        keep_probability: tf.Tensor = None,
        concat_activations: tf.Tensor = None,
        stride: int = 1,
        do_batchnorm: bool = True,
        do_activation: bool = True,
        data_format: str = "NHWC"
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    with tf.name_scope(f"conv2d_transpose_{name}"):

        # Check data format and select input feature count
        tf.assert_rank(input_tensor, 4)
        input_shape = input_tensor.get_shape()
        batch_size = tf.shape(input_tensor)[0]
        if data_format == "NHWC":
            input_feature_count = input_shape[-1]
            output_shape = [batch_size, output_shape_2d[0], output_shape_2d[1], feature_count]
            concat_axis = 3
            strides = [1, stride, stride, 1]
        elif data_format == "NCHW":
            input_feature_count = input_shape[1]
            output_shape = [batch_size, feature_count, output_shape_2d[0], output_shape_2d[1]]
            concat_axis = 1
            strides = [1, 1, stride, stride]
        else:
            raise ValueError(f"Unsupported data format {data_format}")

        # Create variables
        w = tf.get_variable(
            f"w_{name}",
            shape=[filter_size, filter_size, feature_count, input_feature_count],
            dtype=tf.float32,
            initializer=tf.uniform_unit_scaling_initializer(1.43))  # TODO: Use correct initializer
        b = tf.get_variable(f"b_{name}",
                            shape=[feature_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        # Perform convolution
        conv = tf.nn.conv2d_transpose(
            input_tensor,
            filter=w,
            output_shape=output_shape,
            strides=strides,
            padding="SAME",  # TODO: Make padding configurable
            data_format=data_format)

        result = tf.nn.bias_add(conv, b, data_format=data_format)

        if do_batchnorm:
            result = tf.contrib.layers.batch_norm(result, decay=0.9, epsilon=1e-5, fused=True, data_format=data_format)  # TODO: Params?

        if keep_probability is not None:
            result = tf.nn.dropout(result, keep_probability)

        if do_activation:
            result = tf.nn.relu(result)

        if concat_activations is not None:
            result = tf.concat([result, concat_activations], axis=concat_axis)

        return result, w, b


def conv2d(
        input_tensor: tf.Tensor,
        feature_count: int,
        name: str,
        filter_size: int,
        stride: int = 1,
        do_batchnorm: bool = True,
        do_activation: bool = True,
        data_format: str = "NHWC"
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    with tf.name_scope(f"conv2d_{name}"):
        # Check data format and select input feature count
        tf.assert_rank(input_tensor, 4)
        input_shape = input_tensor.get_shape()
        if data_format == "NHWC":
            input_feature_count = input_shape[-1]
            strides = [1, stride, stride, 1]
        elif data_format == "NCHW":
            input_feature_count = input_shape[1]
            strides = [1, 1, stride, stride]
        else:
            raise ValueError(f"Unsupported data format {data_format}")

        # Create variables
        w = tf.get_variable(
            f"w_{name}",
            shape=[filter_size, filter_size, input_feature_count, feature_count],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))  # TODO: Use correct initializer
        b = tf.get_variable(f"b_{name}",
                            shape=[feature_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(
            input_tensor,
            filter=w,
            strides=strides,
            padding="SAME",  # TODO: Make padding configurable
            data_format=data_format)

        result = tf.nn.bias_add(conv, b, data_format=data_format)

        if do_batchnorm:
            result = tf.contrib.layers.batch_norm(result, decay=0.9, epsilon=1e-5, fused=True, data_format=data_format)  # TODO: Params?

        if do_activation:
            result = leaky_relu(result, 0.2)

        return result, w, b

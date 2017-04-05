import tensorflow as tf
from molanet.operations import *


def dcgan_generator(
    z,
    batch_size,
    scope="generator"
):
    with tf.variable_scope(scope):
        # Project z into a 4*4*256 high level representation and reshape into 4x4x256 tensor
        w_proj = weight_variable("w_proj", [z.shape[1], 4*4*256]) # TODO: Use z shape instead of hardcoded
        b_proj = bias_variable("b_proj", [4*4*256])
        fc_proj = tf.nn.bias_add(tf.matmul(z, w_proj), b_proj)
        bn_proj = tf.contrib.layers.batch_norm(fc_proj, decay=0.9, epsilon=1e-5)
        a_proj = tf.nn.relu(bn_proj)
        z_proj = tf.reshape(a_proj, [-1, 4, 4, 256], "z_proj")

        # 4x4x256 -> 8x8x128
        w1 = weight_variable("w1", [5, 5, 128, 256])
        b1 = bias_variable("b1", [128])
        conv1 = tf.nn.bias_add(tf.nn.conv2d_transpose(z_proj, w1, output_shape=[batch_size, 8, 8, 128], strides=[1, 2, 2, 1], padding="SAME"), b1)
        bn1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, epsilon=1e-5)
        a1 = tf.nn.relu(bn1)

        # 8x8x128 -> 16x16x64
        w2 = weight_variable("w2", [5, 5, 64, 128])
        b2 = bias_variable("b2", [64])
        conv2 = tf.nn.bias_add(tf.nn.conv2d_transpose(a1, w2, output_shape=[batch_size, 16, 16, 64], strides=[1, 2, 2, 1], padding="SAME"), b2)
        bn2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, epsilon=1e-5)
        a2 = tf.nn.relu(bn2)

        # 16x16x64 -> 32x32x1
        w_out = weight_variable("w_out", [5, 5, 1, 64])
        b_out = bias_variable("b_out", [1])
        conv_out = tf.nn.bias_add(tf.nn.conv2d_transpose(a2, w_out, output_shape=[batch_size, 32, 32, 1], strides=[1, 2, 2, 1], padding="SAME"), b_out)
        a_out = tf.nn.tanh(conv_out)

        return (
            a_out,
            [w_proj, w1, w2, w_out],
            [b_proj, b1, b2, b_out])


def dcgan_discriminator(
    x,
    class_count,
    scope="discriminator",
    reuse=False
):
    with tf.variable_scope(scope, reuse=reuse):
        # 32x32x1 -> 16x16x32
        w1 = weight_variable("w1", [5, 5, 1, 32])
        b1 = bias_variable("b1", [32])
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding="SAME"), b1)
        #bn1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, epsilon=1e-5)  # TODO: Correct position? How to use? Params?
        a1 = leaky_relu(conv1, 0.2)

        # 16x16x32 -> 8x8x64
        w2 = weight_variable("w2", [5, 5, 32, 64])
        b2 = bias_variable("b2", [64])
        conv2 = tf.nn.bias_add(tf.nn.conv2d(a1, w2, strides=[1, 2, 2, 1], padding="SAME"), b2)
        bn2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, epsilon=1e-5)  # TODO: Correct position? How to use? Params?
        a2 = leaky_relu(bn2, 0.2)

        # 8x8x64 -> 4x4x128
        w3 = weight_variable("w3", [5, 5, 64, 128])
        b3 = bias_variable("b3", [128])
        conv3 = tf.nn.bias_add(tf.nn.conv2d(a2, w3, strides=[1, 2, 2, 1], padding="SAME"), b3)
        bn3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, epsilon=1e-5)  # TODO: Correct position? How to use? Params?
        a3 = leaky_relu(bn3, 0.2)

        w_out = weight_variable("w_out", [4*4*128, class_count])
        b_out = bias_variable("b_out", [class_count])
        fc_out = tf.nn.bias_add(tf.matmul(tf.reshape(a3, [-1, 4*4*128]), w_out), b_out)
        a_out = tf.nn.sigmoid(fc_out)

        return (
            a_out,
            fc_out,
            [w1, w2, w3, w_out],
            [b1, b2, b3, b_out])


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))


def bias_variable(name, shape):
    # TODO: Correct initialization
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

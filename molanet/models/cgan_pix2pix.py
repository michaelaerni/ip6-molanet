from __future__ import division

import numpy as np
import tensorflow as tf

import molanet.operations as ops

use_gpu = False


def cgan_pix2pix_generator(
        image,
        input_size=256,
        input_color_channels=3,
        output_size=256,
        output_color_channels=3,
        batch_size=1,
        g_filter_dim=64
):
    # image is assumed to be 256 x 256 (xydim * xydim * zdim)

    with tf.variable_scope("generator"):
        # generator u-net

        s = output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
            s / 64), int(s / 128)

        # encoder , downsampling
        # 256x256 => 128x128
        e1 = conv2d(image, g_filter_dim, name='g_e1_conv')  # 128 x 128
        # 128x128=> 64x64
        e2 = conv2d(leaky_relu(e1), g_filter_dim * 2, name='g_e2_conv')
        bn_e2 = batch_norm(e2, 'g_bn_e2')
        # 64 x 64 => 32 x 32
        e3 = conv2d(leaky_relu(bn_e2), g_filter_dim * 4, name='g_e3_conv')
        bn_e3 = batch_norm(e3, 'g_bn_e3')
        # 32 x 32 => 16 x 16
        e4 = conv2d(leaky_relu(bn_e3), g_filter_dim * 8, name='g_e4_conv')
        bn_e4 = batch_norm(e4, 'g_bn_e4')
        # 16x16 => 8x8
        e5 = conv2d(leaky_relu(bn_e4), g_filter_dim * 8, name='g_e5_conv')
        bn_e5 = batch_norm(e5, 'g_bn_e5')
        # 8x8 => 4x4
        e6 = conv2d(leaky_relu(bn_e5), g_filter_dim * 8, name='g_e6_conv')
        bn_e6 = batch_norm(e6, 'g_bn_e6')
        # 4x4 => 2x2
        e7 = conv2d(leaky_relu(bn_e6), g_filter_dim * 8, name='g_e7_conv')
        bn_e7 = batch_norm(e7, 'g_bn_e7')
        # 2x2 => 1x1
        e8 = conv2d(leaky_relu(bn_e7), g_filter_dim * 8, name='g_e8_conv')
        bn_e8 = batch_norm(e8, 'g_bn_e8')


    def deconv2d_skipConn(features, s, skip_con, g_filter_dim_factor, deconv_name, bn_name):
        d = deconv2d(tf.nn.relu(features), [batch_size, s, s, g_filter_dim * g_filter_dim_factor], name=deconv_name)
        d = batch_norm(d, bn_name)
        d = tf.nn.dropout(d, 0.5)
        d = tf.concat([d, skip_con], 3)  # skipconnection
        return d

    # decoder with skip connections

    # deconvolve e8 and add skip connections to e7
    d1 = deconv2d_skipConn(bn_e8, s128, bn_e7, 8, 'g_d1', 'g_bn_d1')
    d2 = deconv2d_skipConn(d1, s64, bn_e6, 8, 'g_d2', 'g_bn_d3')
    # d3 is (8 x 8 x g_filter_dim*8*2)
    d3 = deconv2d_skipConn(d2, s32, bn_e5, 8, 'g_d3', 'g_bn_d3')
    # d4 is (16 x 16 x g_filter_dim*4*2)
    d4 = deconv2d_skipConn(d3, s16, bn_e4, 8, 'g_d4', 'g_bn_d4')
    # d5 is (32 x 32 x g_filter_dim*4*2)
    d5 = deconv2d_skipConn(d4, s8, bn_e3, 4, 'g_d5', 'g_bn_d5')
    # d6 is 64 x 64 x g_filter_dim*2*2
    d6 = deconv2d_skipConn(d5, s4, bn_e2, 2, 'g_d6', 'g_bn_d6')
    # d7 is 128 x 128 x g_filter_dim*2*2
    d7 = deconv2d_skipConn(d6, s2, e1, 1, 'g_d7', 'g_bn_d7')
    # d8 is 256 x 256
    d8 = deconv2d(tf.nn.relu(d7), [batch_size, s, s, output_color_channels], name='g_d8')

    return tf.nn.tanh(d8)


def cgan_pix2pix_discriminator(
        image: np.ndarray,
        batch_size=1,
        reuse=False,
        d_filter_dim=64
):
    with tf.variable_scope("discriminator", reuse=reuse):
        c1 = leaky_relu(conv2d(image, d_filter_dim, name='d_c1'))  # 256 => 128
        c2 = leaky_relu(conv2d(c1, d_filter_dim * 2, name='d_c2'))  # 128 => 64
        c3 = leaky_relu(conv2d(c2, d_filter_dim * 4, name='d_c3'))  # 64 => 32
        c4 = leaky_relu(conv2d(c3, d_filter_dim * 8, name='d_c4'))  # 32 => 16

        c4_reshaped = tf.reshape(c4, [batch_size, -1])
        bias = bias_variable('d_bias_linear', 1)
        weights = tf.get_variable("d_weights_linear", [c4_reshaped.get_shape().as_list()[1], 1], tf.float32,
                                  tf.random_normal_initializer(stddev=0.02))

        l1 = tf.nn.bias_add(tf.matmul(c4_reshaped, weights), bias)
        return tf.nn.sigmoid(l1), l1


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))


def bias_variable(name, shape):
    # TODO: using 0.0 or 0.1 as initial value?
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2,
           name="conv2d"):
    with tf.variable_scope(name):
        w = weight_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        bias = bias_variable('bias', [output_dim])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())  # TODO just add bias no reshaping
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def leaky_relu(features):
    return ops.leaky_relu(features, 0.2)


def batch_norm(features, name, decay=0.9, epsilon=1e-5):
    return tf.contrib.layers.batch_norm(features, decay=decay, updates_collections=None, epsilon=epsilon, scale=True,
                                        scope=name)

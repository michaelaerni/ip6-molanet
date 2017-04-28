from __future__ import division

import numpy as np
import tensorflow as tf

import molanet.operations as ops

use_gpu = False
IMAGE_SIZE = 32


class GlsGANModel(object):
    def __init__(self,
                 batch_size: int,
                 image_size: int,
                 src_color_channels: int,
                 target_color_channels: int,
                 glsgan_slope_alpha : float = 0.0,
                 l1_lambda: float = 100,
                 num_feature_maps: int = 64):
        self.batch_size = batch_size
        self.image_size = image_size
        self.src_color_dim = src_color_channels
        self.target_color_dim = target_color_channels
        self.L1_lambda = l1_lambda
        self.g_num_feature_maps = num_feature_maps
        self.d_num_feature_maps = num_feature_maps
        self.output_color_channels = target_color_channels
        self.output_size = image_size
        self.glsgan_alpha = glsgan_slope_alpha

        self.build_model()

    def build_model(self):
        self.real_data_source = tf.placeholder(tf.float32,
                                               [self.batch_size, self.image_size, self.image_size,
                                                self.src_color_dim],
                                               name='source_images')
        self.real_data_target = tf.placeholder(tf.float32,
                                               [self.batch_size, self.image_size, self.image_size,
                                                self.target_color_dim],
                                               name='target_images')

        self.real_A = self.real_data_source
        self.real_B = self.real_data_target

        self.fake_B = self.cgan_pix2pix_generator(self.real_A)



        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.cgan_pix2pix_discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.cgan_pix2pix_discriminator(self.fake_AB, reuse=True)

        # self.fake_B_sample = self.sampler(self.real_A) #TODO what is dis

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                      + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        #see glsgan paper and https://github.com/guojunq/glsgan/blob/master/glsgan.lua#L257
        def l1diff(x,y):
            dist = tf.reduce_sum(tf.abs(y-x))
            return dist
        pdist = self.L1_lambda * l1diff(self.real_B,self.fake_B) #TODO how to make this work with differently shaped tensors
        self.cost1 = pdist + self.d_loss_real - self.d_loss_fake

        self.glsloss = ops.leaky_relu(self.cost1,self.glsgan_alpha)
        #self.d_error_hinge = tf.reduce_mean(self.glsloss)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def cgan_pix2pix_generator(self, image: np.ndarray):
        # image is assumed to be 256 x 256 * 3
        with tf.variable_scope("generator"):
            # generator u-net

            # encoder , downsampling
            # 256x256 => 128x128
            # e1 = conv2d(image, self.g_num_feature_maps, name='g_e1_conv')  # 128 x 128
            # 128x128=> 64x64
            # e2 = conv2d(leaky_relu(e1), self.g_num_feature_maps * 2, name='g_e2_conv')
            # bn_e2 = batch_norm(e2, 'g_bn_e2')
            # 64 x 64 => 32 x 32
            # e3 = conv2d(leaky_relu(bn_e2), self.g_num_feature_maps * 4, name='g_e3_conv')
            # bn_e3 = batch_norm(e3, 'g_bn_e3')
            # 32 x 32 => 16 x 16
            e4 = conv2d(image, self.g_num_feature_maps, name='g_ee4_conv')
            # bn_e4 = batch_norm(e4, 'g_bn_e4')
            # 16x16 => 8x8
            e5 = conv2d(leaky_relu(e4), self.g_num_feature_maps * 2, name='g_e5_conv')
            bn_e5 = batch_norm(e5, 'g_bn_e5')
            # 8x8 => 4x4
            e6 = conv2d(leaky_relu(bn_e5), self.g_num_feature_maps * 4, name='g_e6_conv')
            bn_e6 = batch_norm(e6, 'g_bn_e6')
            # 4x4 => 2x2
            e7 = conv2d(leaky_relu(bn_e6), self.g_num_feature_maps * 8, name='g_e7_conv')
            bn_e7 = batch_norm(e7, 'g_bn_e7')
            # 2x2 => 1x1
            e8 = conv2d(leaky_relu(bn_e7), self.g_num_feature_maps * 8, name='g_e8_conv')
            bn_e8 = batch_norm(e8, 'g_bn_e8')

        # decoder with skip connections
        # deconvolve e8 and add skip connections to e7
        d1 = deconv2d_with_skipconn(bn_e8, bn_e7, self.batch_size, self.g_num_feature_maps * 8, 'g_d1', 'g_bn_d1',
                                    True)
        d2 = deconv2d_with_skipconn(d1, bn_e6, self.batch_size, self.g_num_feature_maps * 8, 'g_d2', 'g_bn_d2',
                                    True)
        # d3 is (8 x 8 x g_num_feature_maps*8*2)
        d3 = deconv2d_with_skipconn(d2, bn_e5, self.batch_size, self.g_num_feature_maps * 4, 'g_d3', 'g_bn_d3',
                                    True)
        # d4 is (16 x 16 x g_num_feature_maps*4*2)
        d4 = deconv2d_with_skipconn(d3, e4, self.batch_size, self.g_num_feature_maps * 2, 'g_d4', 'g_bn_d4')
        # d5 is (32 x 32 x self.g_num_feature_maps*4*2)
        # d5 = deconv2d_with_skipconn(d4, s8, bn_e3, self.batch_size, g_num_feature_maps * 4, 'g_d5', 'g_bn_d5')
        d5 = deconv2d(tf.nn.relu(d4),
                      [self.batch_size, self.output_size, self.output_size, self.output_color_channels],
                      name='g_d5')
        ## d6 is 64 x 64 x self.g_num_feature_maps*2*2
        # d6 = deconv2d_with_skipconn(d5, s4, bn_e2, self.batch_size, self.g_num_feature_maps * 2, 'g_d6', 'g_bn_d6')
        ## d7 is 128 x 128 x g_num_feature_maps*2*2
        # d7 = deconv2d_with#_skipconn(d6, s2, e1, self.batch_size, self.g_num_feature_maps, 'g_d7', 'g_bn_d7')
        ## d8 is 256 x 256
        # d8 = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, output_color_channels], name='g_d8')

        # return tf.nn.tanh(d8)
        return tf.nn.tanh(d5)

    def cgan_pix2pix_discriminator(
            self,
            image: np.ndarray,
            reuse=False
    ):
        with tf.variable_scope("discriminator", reuse=reuse):
            c1 = leaky_relu(conv2d(image, self.d_num_feature_maps, name='d_c1'))  # 256 => 128
            c2 = leaky_relu(conv2d(c1, self.d_num_feature_maps * 2, name='d_c2'))  # 128 => 64
            c3 = leaky_relu(conv2d(c2, self.d_num_feature_maps * 4, name='d_c3'))  # 64 => 32
            c4 = leaky_relu(conv2d(c3, self.d_num_feature_maps * 8, name='d_c4'))  # 32 => 16

            c4_reshaped = tf.reshape(c4, [self.batch_size, -1])
            bias = bias_variable('d_bias_linear', 1)
            weights = tf.get_variable("d_weights_linear", [c4_reshaped.get_shape().as_list()[1], 1], tf.float32,
                                      tf.random_normal_initializer(stddev=0.02))

            l1 = tf.nn.bias_add(tf.matmul(c4_reshaped, weights), bias)
            return tf.nn.sigmoid(l1), l1


def weight_variable(name, shape):
    # TODO test using random_normal_initializer instead of truncated_normal_initializer
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))


def bias_variable(name, shape):
    # TODO: using 0.0 or 0.1 as initial value?
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2,
           name='conv2d'):
    with tf.variable_scope(name):
        w = weight_variable('w', [k_h, k_w, input_.shape[-1], output_dim])
        bias = bias_variable('bias', [output_dim])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())  # TODO just add bias no reshaping
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2,
             name='deconv2d'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = weight_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = bias_variable('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv


def deconv2d_with_skipconn(features, skip_con, batch_size, num_filters, deconv_name, bn_name, with_dropout=False):
    assert features.shape[1] * 2 == skip_con.shape[1]
    assert features.shape[2] * 2 == skip_con.shape[2]
    assert skip_con.shape[1] == skip_con.shape[
        2]  # doesn't actually have to hold true in principle but it's good practice
    size_after_deconv = int(skip_con.shape[1])

    d = deconv2d(tf.nn.relu(features), [batch_size, size_after_deconv, size_after_deconv, num_filters],
                 name=deconv_name)
    d = batch_norm(d, bn_name)
    if with_dropout:
        d = tf.nn.dropout(d, 0.5)
    d = tf.concat([d, skip_con], 3)  # skipconnection
    return d


def leaky_relu(features):
    return ops.leaky_relu(features, 0.2)


def batch_norm(features, name, decay=0.9, epsilon=1e-5):
    return tf.contrib.layers.batch_norm(features, decay=decay, updates_collections=None, epsilon=epsilon, scale=True,
                                        scope=name)

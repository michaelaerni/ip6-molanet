import os

import math
import random

from molanet.models.cgan_pix2pix import *
import numpy as np
from PIL import Image

class Model(object):
    def __init__(self, batch_size, image_size, image_color_dim, l1_lambda=100):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_color_dim = image_color_dim
        self.L1_lambda = l1_lambda

    def build_model(self):
        self.real_data_source = tf.placeholder(tf.float32,
                                               [self.batch_size, self.image_size, self.image_size,
                                                self.image_color_dim],
                                               name='source_images')
        self.real_data_target = tf.placeholder(tf.float32,
                                               [self.batch_size, self.image_size, self.image_size,
                                                self.image_color_dim],
                                               name='target_images')

        self.real_A = self.real_data_source
        self.real_B = self.real_data_target

        self.fake_B = cgan_pix2pix_generator(self.real_A, batch_size=batch_size, g_filter_dim=num_feature_maps)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = cgan_pix2pix_discriminator(self.real_AB, batch_size=batch_size, reuse=False)
        self.D_, self.D_logits_ = cgan_pix2pix_discriminator(self.fake_AB, batch_size=batch_size, reuse=True)

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

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def load_random_samples(self, num_images,source_dir,target_dir ):
        data = np.random.choice(np.arange(num_images), self.batch_size)
        source, target = [load_image(sample_file,source_dir,target_dir) for sample_file in data]


def load_image(number: int, source_dir, target_dir):
    def get_filename(number: int) -> str:
        if (number == 0): return 'ISIC_0000000.jpg'
        digits = int(math.log(number, 10))
        zeros_to_add = 6 - int(digits)
        number = str(number)
        for x in range(zeros_to_add):
            number = '0' + number
        return 'ISIC_%s.jpg' % number

    def transformImageNameSource(name):
        return os.path.join(source_dir, name)

    def transformImageNameTarget(name: str):
        name = name.replace('.jpg', '_Segmentation.png')
        return os.path.join(target_dir, name)

    image_name = get_filename(number)
    source_image = Image.open(transformImageNameSource(image_name))
    target_image = Image.open(transformImageNameSource(image_name))

    # TODO think about proper resizing... is dis hacky? I don't know
    size = 256, 256
    source = source_image.thumbnail(size, Image.ANTIALIAS)
    target = target_image.thumbnail(size, Image.ANTIALIAS)

    return np.array(source), np.array(target)

def get_image_batch(batch_size,images_count,source_dir,target_dir):
    #TODO chances are we don't get fucked by rng
    indices = [random.randint(1, images_count) for _ in range(batch_size)]
    images = [ load_image(x,source_dir,target_dir) for x in indices]
    return images

#train
def train():
    restore_iteration = None
    saver = tf.train.Saver()
    batch_size = 1
    sess = None
    is_grayscale = True

    if restore_iteration is not None and restore_iteration > 0:
        iteration_start = restore_iteration + 1
        saver.restore(sess, "{model_directory}/model-{restore_iteration}.cptk")
    else:
        iteration_start = 0

    iterations = 50000
    for iteration in range(iteration_start, iterations):
        #TODO hardcoded
        batch_images = get_image_batch(batch_size,11402,
                                       source_dir=r'C:\Users\pdcwi\Documents\IP6 nonsynced\pix2pix-poc-data\training\source',
                                       target_dir=r'C:\Users\pdcwi\Documents\IP6 nonsynced\pix2pix-poc-data\training\target')


        #dealing with batchsize > 1
        batch = None
        if (is_grayscale):
            batch = np.array(batch_images).astype(np.float32)[:, :, :, None]
        else:
            batch = np.array(batch_images).astype(np.float32)



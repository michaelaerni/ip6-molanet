import datetime
import os
import random
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from molanet.models.cgan_pix2pix import cgan_pix2pix_discriminator, cgan_pix2pix_generator, IMAGE_SIZE


class Model(object):
    def __init__(self, batch_size, image_size, src_color_dim, target_color_dim, l1_lambda=100):
        self.batch_size = batch_size
        self.image_size = image_size
        self.src_color_dim = src_color_dim
        self.target_color_dim = target_color_dim
        self.L1_lambda = l1_lambda

    def build_model(self, num_feature_maps):
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

        self.fake_B = cgan_pix2pix_generator(self.real_A, batch_size=self.batch_size, g_filter_dim=num_feature_maps,
                                             output_color_channels=1, output_size=self.image_size)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = cgan_pix2pix_discriminator(self.real_AB, batch_size=self.batch_size, reuse=False)
        self.D_, self.D_logits_ = cgan_pix2pix_discriminator(self.fake_AB, batch_size=self.batch_size, reuse=True)

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


def load_image(name: str, source_dir, target_dir, size=IMAGE_SIZE):
    def transformImageNameSource(name):
        return os.path.join(source_dir, name)

    def transformImageNameTarget(name: str):
        name = name.replace('.jpg', '_Segmentation.png')
        return os.path.join(target_dir, name)

    source_image = Image.open(transformImageNameSource(name))
    target_image = Image.open(transformImageNameTarget(name))

    # TODO think about proper resizing... is dis hacky? I don't know
    size = size, size
    source = source_image.resize(size, Image.BICUBIC)
    target = target_image.resize(size, Image.NEAREST)
    target = target.convert('1')  # to black and white

    return np.array(source).astype(np.float32), np.array(target).astype(np.float32)


def get_image_batch(batch_size, source_file_names, source_dir, target_dir) -> [np.ndarray, np.ndarray]:
    # TODO chances are we don't get fucked by rng
    indices = [random.randint(0, len(source_file_names) - 1) for _ in range(batch_size)]
    images = [load_image(source_file_names[i], source_dir, target_dir) for i in indices]
    return images


# train
def train():
    source_dir = r'../../../pix2pix-poc-data/training/source'
    target_dir = r'../../../pix2pix-poc-data/training/target'
    sourcefiles = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    now = datetime.datetime.now()
    sample_dir = "./samples/sample-%d-%d-%d--%02d%02d" % (
        now.day, now.month, now.year, now.hour, now.minute)  # Generated samples
    model_directory = "./models"  # Model
    if not os.path.exists('./samples'):
        os.mkdir('./samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    use_random_image_as_sample = False
    batch_size = 1
    size = IMAGE_SIZE
    num_feature_maps = 64

    restore_iteration = None
    iterations = 50000
    is_grayscale = False

    with tf.Session() as sess:
        model = Model(batch_size=batch_size, image_size=size, src_color_dim=3, target_color_dim=1)
        model.build_model(num_feature_maps)

        saver = tf.train.Saver()
        if restore_iteration is not None and restore_iteration > 0:
            iteration_start = restore_iteration + 1
            saver.restore(sess, "{model_directory}/model-{restore_iteration}.cptk")
        else:
            iteration_start = 0

        # Optimizers
        disc_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)  # TODO: pix2pix params
        gen_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)  # TODO: pix2pix params

        disc_update_step = disc_optim.minimize(model.d_loss, var_list=model.d_vars)
        gen_update_step = gen_optim.minimize(model.g_loss, var_list=model.g_vars)

        sess.run(tf.global_variables_initializer())

        # logging
        writer = tf.summary.FileWriter("./logs", sess.graph)
        g_sum = tf.summary.merge([model.d__sum,
                                  model.fake_B_sum, model.d_loss_fake_sum, model.g_loss_sum])
        d_sum = tf.summary.merge([model.d_sum, model.d_loss_real_sum, model.d_loss_sum])

        start_time = time.time()
        for iteration in range(iteration_start, iterations):
            batch = get_image_batch(batch_size, sourcefiles,
                                    source_dir=source_dir,
                                    target_dir=target_dir)
            (batch_src, batch_target) = batch[0]
            batch_src = (batch_src / 255.0 - 0.5) * 2.0  # Transform into range -1, 1
            batch_target = (batch_target - 0.5) * 2.0  # Transform into range -1, 1

            batch_src = np.array(batch_src).astype(np.float32)[None, :, :, :]
            batch_target = np.array(batch_target).astype(np.float32)[None, :, :, None]

            _, summary_str = sess.run([disc_update_step, d_sum],
                                      feed_dict={model.real_data_source: batch_src,
                                                 model.real_data_target: batch_target})

            writer.add_summary(summary_str, iteration)

            # Update G network twice
            _, summary_str = sess.run([gen_update_step, g_sum],
                                      feed_dict={model.real_data_source: batch_src,
                                                 model.real_data_target: batch_target})
            writer.add_summary(summary_str, iteration)
            _, summary_str = sess.run([gen_update_step, g_sum],
                                      feed_dict={model.real_data_source: batch_src,
                                                 model.real_data_target: batch_target})
            writer.add_summary(summary_str, iteration)

            errD_fake = model.d_loss_fake.eval(
                {model.real_data_target: batch_target, model.real_data_source: batch_src})
            errD_real = model.d_loss_real.eval(
                {model.real_data_target: batch_target, model.real_data_source: batch_src})
            errG = model.g_loss.eval({model.real_data_target: batch_target, model.real_data_source: batch_src})

            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (iteration, time.time() - start_time, errD_fake + errD_real, errG))

            if iteration % 50 == 1:
                # gib nice picture output :)
                sample_model(sourcefiles, iteration, sess, source_dir, target_dir, model, sample_dir,
                             use_random_image_as_sample)

            if iteration % 500 == 2:
                save(sess, saver, model_directory, iteration)


def save(sess, saver, checkpoint_dir, step):
    model_name = "cgan_pix2pix.model"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def sample_model(filenames: [str], epoch: int, sess: tf.Session, source_dir: str, target_dir: str, model: Model,
                 sample_dir: str, rng: bool):
    batch = None
    if rng:
        batch = get_image_batch(1, filenames, source_dir, target_dir)
    else:
        batch = [load_image('ISIC_0000000.jpg', source_dir, target_dir)]

    (batch_src, batch_target) = batch[0]
    original_source = batch_src.copy()
    original_target = batch_target.copy()
    batch_src = (batch_src / 255.0 - 0.5) * 2.0  # Transform into range -1, 1
    batch_target = (batch_target - 0.5) * 2.0  # Transform into range -1, 1
    batch_src = np.array(batch_src).astype(np.float32)[None, :, :, :]
    batch_target = np.array(batch_target).astype(np.float32)[None, :, :, None]

    sample, d_loss, g_loss = sess.run(
        [model.fake_B, model.d_loss, model.g_loss],
        feed_dict={model.real_data_source: batch_src,
                   model.real_data_target: batch_target}
    )

    # convert images from [-1,1] to [0,255]
    # from shape [1,255,255,1] : tensor to shape [255,255] : nddarray
    sample = (tf.squeeze(sample).eval() + 1.0) / 2.0 * 255

    save_ndarrays_asimage(os.path.join(sample_dir, 'sample_%d.png' % epoch), original_source, sample,
                          original_target * 255)
    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


def save_ndarrays_asimage(filename: str, *arrays: np.ndarray):
    def fix_dimensions(array):
        if array.ndim > 3 or array.ndim < 2: raise ValueError('arrays must have 2 or 3 dimensions')
        if array.ndim == 2:
            array = np.repeat(array[:, :, np.newaxis], 3, axis=2)  # go from blackwhite to rgb
        return array

    if len(arrays) > 1:
        arrays = [fix_dimensions(array) for array in arrays]
        arrays = np.concatenate(arrays, axis=1)

    # arrays is just a big 3-dim matrix
    im = Image.fromarray(np.uint8(arrays))
    im.save(filename)


if __name__ == '__main__':
    train()

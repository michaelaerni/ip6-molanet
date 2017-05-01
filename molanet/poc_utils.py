import os
import random
import sys

sys.path.append(os.path.abspath("../.."))

import tensorflow as tf
import numpy as np
from PIL import Image

from molanet.models.cgan_pix2pix import IMAGE_SIZE
from molanet.models.cgan_pix2pix import Pix2PixModel


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


def transform_batch(image_batch):
    batch_src, batch_target = image_batch[0]
    batch_src = (batch_src / 255.0 - 0.5) * 2.0  # Transform into range -1, 1
    batch_target = (batch_target - 0.5) * 2.0  # Transform into range -1, 1

    batch_src = np.array(batch_src).astype(np.float32)[None, :, :, :]
    batch_target = np.array(batch_target).astype(np.float32)[None, :, :, None]

    if (len(image_batch) > 1):
        iterimages = iter(image_batch)
        next(iterimages)  # skip first
        for src, target in iterimages:
            src = (src / 255.0 - 0.5) * 2.0  # Transform into range -1, 1
            target = (target - 0.5) * 2.0  # Transform into range -1, 1
            src = np.array(src).astype(np.float32)[None, :, :, :]
            target = np.array(target).astype(np.float32)[None, :, :, None]
            batch_src = np.concatenate([batch_src, src], axis=0)
            batch_target = np.concatenate([batch_target, target], axis=0)
    return batch_src, batch_target


def save_ndarrays_asimage(filename: str, *arrays: np.ndarray):
    def fix_dimensions(array):
        if array.ndim > 3 or array.ndim < 2: raise ValueError('arrays must have 2 or 3 dimensions')
        if array.ndim == 2:
            array = np.repeat(array[:, :, np.newaxis], 3,
                              axis=2)  # go from blackwhite to rgb to make concat work seamless
        return array

    if len(arrays) > 1:
        arrays = [fix_dimensions(array) for array in arrays]
        arrays = np.concatenate(arrays, axis=1)

    # arrays is just a big 3-dim matrix
    im = Image.fromarray(np.uint8(arrays))
    im.save(filename)


def image_summary(model: Pix2PixModel, max_image_outputs: int = 3):
    max_image_outputs = min(model.batch_size, max_image_outputs)
    fake_B_rgb = tf.concat([model.fake_B, model.fake_B, model.fake_B], axis=3)
    real_B_rgb = tf.concat([model.real_B, model.real_B, model.real_B], axis=3)
    fake_image = tf.concat([model.real_A, real_B_rgb, fake_B_rgb, tf.abs(real_B_rgb - fake_B_rgb)], axis=2)
    return tf.summary.image(
        name='sample',
        max_outputs=max_image_outputs,
        tensor=fake_image)


def save(sess, saver, checkpoint_dir, step):
    model_name = "glsgan.model"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

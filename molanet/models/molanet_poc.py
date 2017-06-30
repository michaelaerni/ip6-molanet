import argparse
import os
import shutil
from datetime import datetime

import tensorflow as tf

from molanet.base import NetworkTrainer
from molanet.input import create_fixed_input_pipeline
from molanet.models.pix2pix import Pix2PixFactory
from molanet.models.wgan_gp import WassersteinGradientPenaltyFactory


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Molanet PoC script")

    parser.add_argument("--sampledir", type=str, help="Root sample directory")
    parser.add_argument("--metafile", type=str, help="CSV file containing the UUIDs of the training samples")
    parser.add_argument("--logdir", type=str, help="Directory into which summaries and checkpoints are written")
    parser.add_argument("--logsubdir", action='store_true',
                        help="creates a subdirectory in the Directory logdir into which summaries and checkpoints are written")

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    logdir: str
    if args.logsubdir:
        now = datetime.now()
        subdirname = f"tfrun_{now.day:02}_{now.month:02}_{now.hour:02}{now.minute:02}"
        logdir = os.path.join(args.logdir, subdirname)
    else:
        logdir = args.logdir

    shutil.rmtree(logdir, ignore_errors=True)
    os.makedirs(logdir)

    tf.reset_default_graph()
    input_x, input_y = create_fixed_input_pipeline(args.sampledir, args.metafile, 1, 20, 512,
                                                   thread_count=4, min_after_dequeue=10)
    print("Input pipeline created")
    network_factory = Pix2PixFactory(512)
    trainer = NetworkTrainer(
        input_x,
        input_y,
        network_factory,
        WassersteinGradientPenaltyFactory(10, network_factory, l1_lambda=0),
        log_every_n=10,
        learning_rate=0.0001, beta1=0, beta2=0.9)
    print("Trainer created")

    with tf.Session() as sess:
        print("Session started")
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        print("Adding debug image summaries")
        difference_image = tf.abs(tf.subtract(trainer._generator, input_y))
        tf.summary.image("input_x", input_x, max_outputs=1)
        tf.summary.image("input_y", input_y, max_outputs=1)
        tf.summary.image("segmentation", trainer._generator, max_outputs=1)
        tf.summary.image("generated_difference", difference_image, max_outputs=1)
        concatenated_images = tf.concat([
            input_x,
            tf.tile(input_y, multiples=[1, 1, 1, 3]),
            tf.tile(trainer._generator, multiples=[1, 1, 1, 3]),
        ], axis=2)
        tf.summary.image("concatenated_images", concatenated_images, max_outputs=1)

        print("Starting training")
        trainer.train(sess, logdir)

import os
import shutil

import tensorflow as tf

from molanet.base import NetworkTrainer
from molanet.input import create_fixed_input_pipeline
from molanet.models.pix2pix import Pix2PixFactory, Pix2PixLossFactory

if __name__ == "__main__":
    SUMMARY_DIR = "/tmp/molanet_summary"
    shutil.rmtree(SUMMARY_DIR, ignore_errors=True)
    os.makedirs(SUMMARY_DIR)

    tf.reset_default_graph()
    input_x, input_y = create_fixed_input_pipeline("/home/michael/temp/molanet/", "/home/michael/temp/molanet.csv", 4, 10, 512, thread_count=4)
    print("Input pipeline created")
    trainer = NetworkTrainer(input_x, input_y, Pix2PixFactory(512), Pix2PixLossFactory(0.001), 0.001)
    print("Trainer created")

    with tf.Session() as sess:
        print("Session started")
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        print("Adding debug image summaries")
        tf.summary.image("input_x", input_x, max_outputs=1)
        tf.summary.image("input_y", input_y, max_outputs=1)
        tf.summary.image("segmentation", trainer._generator, max_outputs=1)

        print("Starting training")
        trainer.train(sess, SUMMARY_DIR)

import argparse
import os
import shutil
from datetime import datetime

import tensorflow as tf

from molanet.base import NetworkTrainer, TrainingOptions
from molanet.input import create_fixed_input_pipeline, create_static_input_pipeline
from molanet.models.pix2pix import Pix2PixFactory
from molanet.models.wgan_gp import WassersteinGradientPenaltyFactory


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Molanet PoC script")

    parser.add_argument("--sampledir", type=str,
                        help="Root sample directory, containing set directories and meta files")
    parser.add_argument("--test-set", type=str, help="Name of the test set")
    parser.add_argument("--cv-set", type=str, help="Name of the cv set")
    parser.add_argument("--train-set", type=str, help="Name of the training set")
    parser.add_argument("--logdir", type=str, help="Directory into which summaries and checkpoints are written")
    parser.add_argument("--restore", type=int, help="If set, restores the model from logdir with the given iteration")
    parser.add_argument("--debug-placement", action="store_true", help="Output device placement")
    parser.add_argument("--logsubdir", action="store_true", help="Create a subdirectory in logdir for each new run")
    parser.add_argument("--discriminator-iterations", type=int, default=1, help="Number of discriminator iterations")
    parser.add_argument("--l1-lambda", type=int, default=0, help="Generator loss l1 lambda")

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    logdir: str
    if args.logsubdir and args.restore is None:
        now = datetime.now()
        subdirname = f"run_{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}"
        logdir = os.path.join(args.logdir, subdirname)
    else:
        logdir = args.logdir

    if args.restore is None:
        shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir)

    tf.reset_default_graph()

    # Create input pipelines
    training_data = create_fixed_input_pipeline(args.sampledir, args.train_set,
                                                   batch_size=1, epochs=20, image_size=512,
                                                   thread_count=4, min_after_dequeue=10, name="train")
    cv_data = create_static_input_pipeline(args.sampledir, args.cv_set,
                                                               batch_size=1, image_size=512, name="cv")
    print("Input pipeline created")

    network_factory = Pix2PixFactory(512)
    trainer = NetworkTrainer(
        training_data,
        cv_data,
        network_factory,
        WassersteinGradientPenaltyFactory(10, network_factory, l1_lambda=args.l1_lambda),
        training_options=TrainingOptions(
            summary_directory=logdir,
            discriminator_iterations=args.discriminator_iterations),
        learning_rate=0.0001, beta1=0, beta2=0.9)
    print("Trainer created")

    if args.debug_placement:
        print("Device placement logging is enabled")

    with tf.Session(config=tf.ConfigProto(log_device_placement=args.debug_placement)) as sess:
        print("Session started")
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        if args.restore is not None:
            trainer.restore(sess, args.restore)
            print(f"Iteration {args.restore} restored")

        print("Starting training")
        trainer.train(sess)

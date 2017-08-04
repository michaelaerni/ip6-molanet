import argparse
import os
import shutil
import sys
from datetime import datetime

import tensorflow as tf

from molanet.base import NetworkTrainer, TrainingOptions
from molanet.input import TrainingPipeline, \
    EvaluationPipeline, random_rotate_flip_rgb, random_contrast_rgb, random_brightness_rgb, RGBToLabConverter
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
    parser.add_argument("--gradient-lambda", type=int, default=10, help="Discriminator gradient penalty lambda")
    parser.add_argument("--l1-lambda", type=int, default=0, help="Generator loss l1 lambda")
    parser.add_argument("--convert-colors", action="store_true",
                        help="If specified, converts the input images from RGB to CIE Lab color space")
    parser.add_argument("--max-iterations", type=int,
                        help="Maximum number of iterations before training stops")
    parser.add_argument("--xla", action="store_true",
                        help="enable JIT compilation to XLA at the session level (gpu only)")
    return parser


def molanet_main(args:[str]):
    parser = create_arg_parser()
    args = parser.parse_args(args)

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

    AUGMENTATION_FUNCTIONS = [
        lambda image, segmentation: random_rotate_flip_rgb(image, segmentation),
        lambda image, segmentation: (random_contrast_rgb(image, 0.8, 1.2), segmentation),
        lambda image, segmentation: (random_brightness_rgb(image, -0.3, 0.3), segmentation)
    ]

    color_converter = RGBToLabConverter() if args.convert_colors else None

    # Create input pipelines
    # TODO: Image size is hardcoded
    training_pipeline = TrainingPipeline(args.sampledir, args.train_set, image_size=512,
                                         color_converter=color_converter,
                                         batch_size=1, read_thread_count=4, batch_thread_count=1,
                                         augmentation_functions=AUGMENTATION_FUNCTIONS, name="training")
    cv_pipeline = EvaluationPipeline(args.sampledir, args.cv_set, image_size=512,
                                     color_converter=color_converter,
                                     batch_size=1, batch_thread_count=1, name="cv")

    print("Input pipelines created")
    print(f"Training set size: {training_pipeline.sample_count}")
    print(f"CV set size: {cv_pipeline.sample_count}")

    if args.debug_placement:
        print("Device placement logging is enabled")

    config = tf.ConfigProto(log_device_placement=args.debug_placement)
    if args.xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        print("Enabled JIT XLA compilation")

    network_factory = Pix2PixFactory(
        spatial_extent=512,
        min_generator_features=64,
        min_discriminator_features=64,
        max_generator_features=1024,
        max_discriminator_features=1024,
        use_batchnorm=True,
        dropout_layer_count=2
    )

    trainer = NetworkTrainer(
        training_pipeline,
        cv_pipeline,
        network_factory,
        WassersteinGradientPenaltyFactory(args.gradient_lambda, network_factory, l1_lambda=args.l1_lambda),
        training_options=TrainingOptions(
            summary_directory=logdir,
            discriminator_iterations=args.discriminator_iterations,
            max_iterations=args.max_iterations,
            session_configuration=config,
            use_gpu=True),
        learning_rate=0.0001, beta1=0, beta2=0.9)
    print("Trainer created")

    with trainer:
        print("Session started")

        if args.restore is not None:
            trainer.restore(args.restore)
            print(f"Iteration {args.restore} restored")

        print("Starting training")
        trainer.train()

if __name__ == "__main__":
    molanet_main(sys.argv[1:]) # first argument is the name of this script

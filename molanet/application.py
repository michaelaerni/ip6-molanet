import argparse
import logging
from typing import Optional

from molanet.models.final_architecture import MolanetFactory
import tensorflow as tf
import os
import sys
import numpy as np
from PIL import Image


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Molanet Image Segmentation")

    parser.add_argument("mode", choices=["segment", "evaluate"])
    parser.add_argument("inputs", type=str, metavar="INPUT", nargs="+",
                        help="Input files or directories")

    parser.add_argument("--output", "-o", type=str, help="Output directory to write results into")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt", help="Model checkpoint to load parameters from")
    parser.add_argument("--gpu", action="store_true", help="Use GPU instead of CPU")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    return parser


def rescale_convert(input_file: str) -> np.ndarray:
    image = Image.open(input_file)
    image.thumbnail((512, 512), Image.LANCZOS)
    padding = Image.new("RGB", (512, 512), (0, 0, 0))  # Black
    padding.paste(image, ((padding.size[0] - image.size[0]) // 2, (padding.size[1] - image.size[1]) // 2))
    converted = (np.asarray(padding, dtype=np.float32) / 255.0 - 0.5) * 2.0
    return np.expand_dims(converted, axis=0)


def load_transform_lesion(input_file: str) -> np.ndarray:
    return rescale_convert(input_file)


def load_transform_segmentation(input_file: str) -> np.ndarray:
    return np.expand_dims(rescale_convert(input_file)[:, :, :, 0], axis=3)


def save_segmentation(segmentation: np.ndarray, path: str):
    rescaled = np.round((segmentation + 1.0) / 2.0 * 255.0).astype(np.uint8)
    image = Image.fromarray(np.reshape(rescaled, (512, 512)), mode="L")

    image.save(path)


def get_segmentation_file_name(lesion_file_path: str, output_directory: Optional[str] = None) -> str:
    if output_directory is None:
        output_directory = os.path.dirname(lesion_file_path)

    file_name = os.path.basename(lesion_file_path)
    file_name, extension = os.path.splitext(file_name)

    segmentation_file_name = f"{file_name}_mask{extension}"
    return os.path.join(output_directory, segmentation_file_name)


def perform_segmentation(
        input_file: str,
        network: tf.Tensor,
        lesion_image: tf.Tensor,
        _: tf.Tensor,
        session: tf.Session,
        output_directory: Optional[str] = None
):
    output_file = get_segmentation_file_name(input_file, output_directory)
    if os.path.exists(output_file):
        log.warning(f"Output file {output_file} already exists, will be ignored")
        return

    network_input = load_transform_lesion(input_file)
    segmentation = session.run(network, feed_dict={lesion_image: network_input})
    save_segmentation(segmentation, output_file)
    log.info(f"Saved segmentation of {input_file} to {output_file}")


def perform_evaluation(input_file: str,
        network: tf.Tensor,
        lesion_image: tf.Tensor,
        segmentation_mask: tf.Tensor,
        session: tf.Session,
        output_directory: Optional[str] = None
):
    segmentation_file = get_segmentation_file_name(input_file, output_directory)
    if not os.path.isfile(segmentation_file):
        log.warning(f"Segmentation file {segmentation_file} does not exist, will be ignored")
        return

    lesion_input = load_transform_lesion(input_file)
    segmentation_input = load_transform_segmentation(segmentation_file)
    score = session.run(network, feed_dict={lesion_image: lesion_input, segmentation_mask: segmentation_input})
    log.info(f"{input_file}: {np.reshape(score, (-1))}")


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging_level, format="%(message)s")
    log = logging.getLogger(__name__)

    checkpoint_path = os.path.abspath(args.checkpoint)

    # Create network factory to use
    network_factory = MolanetFactory(
        convolutions_per_level=1,
        min_discriminator_features=32,
        max_discriminator_features=512
    )

    # Create actual networks
    log.debug("Creating graph...")
    tf.reset_default_graph()
    lesion_image = tf.placeholder(tf.float32, shape=(1, 512, 512, 3))
    segmentation_mask = tf.placeholder(tf.float32, shape=(1, 512, 512, 1))
    if args.mode == "segment":
        network = network_factory.create_generator(lesion_image, use_gpu=args.gpu, data_format="NHWC")
        method = perform_segmentation
    elif args.mode == "evaluate":
        network = tf.sigmoid(
            network_factory.create_discriminator(lesion_image, segmentation_mask, use_gpu=args.gpu, data_format="NHWC"))
        method = perform_evaluation
    else:
        assert False  # Should never happen

    # Create remaining graph ops and finalize graph
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()
    log.debug("Graph created")

    # Start session
    with tf.Session() as sess:
        # Restore parameters
        log.debug("Restoring parameters...")
        saver.restore(sess, checkpoint_path)
        log.debug("Parameters restored")

        # Handle all specified inputs
        for current_input in (os.path.abspath(relative_path) for relative_path in args.inputs):
            if os.path.isfile(current_input):
                # Single file, just run it
                method(current_input, network, lesion_image, segmentation_mask, sess)
            elif os.path.isdir(current_input):
                # Directory, run on all files contained in that directory
                for current_file in (os.path.join(current_input, entry)
                                     for entry in os.listdir(current_input)
                                     if os.path.isfile(os.path.join(current_input, entry))):
                    method(current_input, network, lesion_image, segmentation_mask, sess)
            else:
                log.warning(f"Path {current_input} does not exist, will be ignored")

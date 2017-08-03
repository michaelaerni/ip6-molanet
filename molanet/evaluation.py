import argparse

from molanet.base import NetworkEvaluator
from molanet.input import EvaluationPipeline, RGBToLabConverter
from molanet.models.pix2pix import Pix2PixFactory


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Molanet evaluation")

    parser.add_argument("sampledir", type=str,
                        help="Root sample directory containing meta file and records")
    parser.add_argument("dataset", type=str, help="Data set name used for evaluation")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file which should be restored")
    parser.add_argument("output", type=str, help="Output directory to write results into")

    parser.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    pipeline = EvaluationPipeline(args.sampledir, args.dataset, image_size=512,
                                     color_converter=RGBToLabConverter(),
                                     batch_size=1, batch_thread_count=1)

    network_factory = Pix2PixFactory(
        512,
        min_generator_features=64,
        min_discriminator_features=64,
        max_generator_features=512,
        max_discriminator_features=512,
        dropout_layer_count=2,
        use_batchnorm=True)

    evaluator = NetworkEvaluator(pipeline, network_factory, args.checkpoint, args.output, args.gpu)

    with evaluator:
        evaluator.evaluate()

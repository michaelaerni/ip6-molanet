import argparse
import json
import os
import shutil

import png

from molanet.data.database import DatabaseConnection


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Dump all images including metadata from a database into a directory")

    parser.add_argument("output", type=str, help="Output directory into which the data is dumped")

    parser.add_argument("--offset", type=int, default=0, help="Starting offset in data set")

    parser.add_argument("--database-host", type=str, default="localhost", help="Target database host")
    parser.add_argument("--database", type=str, default="molanet", help="Target database name")
    parser.add_argument("--database-username", default=None, help="Target database username")
    parser.add_argument("--database-password", default=None, help="Target database password")
    parser.add_argument("--database-port", type=int, default=5432, help="Target database port")

    parser.add_argument("--keep-directory", help="Do not delete the output directory if it already exists")

    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    target_directory = os.path.abspath(args.output)

    with DatabaseConnection(
            args.database_host,
            args.database,
            username=args.database_username,
            password=args.database_password) as db:

        if not args.keep_directory and os.path.exists(target_directory):
            print(f"Removing existing target directory {target_directory}...")
            shutil.rmtree(target_directory)

        # Create directories
        image_directory = os.path.join(target_directory, "images/")
        segmentation_directory = os.path.join(target_directory, "segmentations/")
        meta_directory = os.path.join(target_directory, "meta/")
        os.makedirs(image_directory, exist_ok=True)
        os.makedirs(meta_directory, exist_ok=True)
        os.makedirs(segmentation_directory, exist_ok=True)

        dump_count = args.offset
        for sample in db.get_samples(offset=args.offset):

            # Write mole image
            with open(f"{image_directory}/{sample.uuid}.png", "w+b") as f:
                png.Writer(sample.dimensions[1], sample.dimensions[0], greyscale=False) \
                    .write(f, sample.image.reshape([sample.dimensions[0], -1]))

            # Write masks
            for segmentation in sample.segmentations:
                with open(f"{segmentation_directory}/{sample.uuid}_{segmentation.source_id}.png", "w+b") as f:
                    png.Writer(segmentation.dimensions[1], segmentation.dimensions[0], greyscale=True) \
                        .write(f, segmentation.mask.reshape([segmentation.dimensions[0], -1]) * 255)

            # Write metadata
            sample_meta = DatabaseConnection.sample_to_dict(sample, include_image=False)
            segmentation_meta = [
                DatabaseConnection.segmentation_to_dict(sample.uuid, segmentation, include_image=False)
                for segmentation in sample.segmentations]
            sample_meta["segmentations"] = segmentation_meta
            with open(f"{meta_directory}/{sample.uuid}.json", "w+") as f:
                json.dump(sample_meta, f)

            dump_count += 1
            print(f"[{dump_count}]: Dumped sample {sample.uuid} with {len(sample.segmentations)} segmentations")

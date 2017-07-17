import argparse
import os

import cv2
from PIL import Image

from molanet.data.data_analysis import contains_plaster


def create_Argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("check images for plasters")
    parser.add_argument("--images-dir", type=str, default=None)

    return parser


def getImages(paths: [str]):
    for path in paths:
        img = Image.open(path)
        yield img


if __name__ == '__main__':
    args = create_Argparser().parse_args()

    known_ids = [
        "048c6528-650e-396c-650f-cb6a276c3eab",
        "9edb5140-d4a5-835d-e3e1-64b6d9205254",
        "bc443d7e-391f-7925-31c3-25d926781615",
        "b46275d2-9862-fc6f-4b75-1c5252087839",
        "b177bc16-d29c-bed7-aa22-733fa4c96a78",
    ]

    path = args.images_dir
    paths = [os.path.join(path, it) for it in os.listdir(path)]
    # paths = [os.path.join(path, f"{it}.png") for it in known_ids]

    for idx, img in enumerate(paths):
        if (contains_plaster(cv2.imread(img))):
            print(img.split(os.sep)[-1])
        #experimentalplaster(img)
        # img.show()

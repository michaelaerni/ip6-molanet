import argparse
import os

import numpy as np
from PIL import Image


def isSkin(r: int, g: int, b: int) -> bool:
    """
        (R,G,B) is classified as skin if:
        R > 95 and G > 40 and B > 20 and
        max{R,G,B}−min{R,G,B} > 15 and
        |R−G| > 15
        and R > G and R > B

    see A Survey on Pixel-Based Skin Color Detection Techniques (2003), IN PROC. GRAPHICON-2003

    :param r: range [0,255]
    :param g: range [0,255]
    :param b: range [0,255]
    :return: true if skin color, false otherwise
    """
    return r > 95 and g > 40 and b > 20 and max(r, g, b) - min(r, g, b) > 15 and abs(
        int(r) - int(g)) > 15 and r > g and r > b


def hasPlaster(img: Image) -> bool:
    """
    Detects if an image has non skin colored regions

    :return: true if we are confident the image has a plaseter, false otherwise
    """
    a: np.ndarray = np.asarray(img, dtype=np.uint8)

    d = [tuple(v) for m2d in a for v in m2d]  # wtf stackoverflow
    assert len(d) == a.size // 3
    # print(c)

    skinpix = [(r, g, b) for r, g, b in d if isSkin(r, g, b)]

    print(f"skin pixels {len(skinpix)}, non-skin pixels: {len(d)-len(skinpix)}")


def experimentalplaster(img: Image):
    assert img.mode == 'RGB'
    img.show()
    img = img.copy()
    (width, height) = img.size
    pix = img.load()

    for y in range(height):
        for x in range(width):
            (r, g, b) = pix[x, y]
            if not isSkin(r, g, b) and not (r == 0 and b == 0 and g == 0):
                pix[x, y] = (255, 0, 0)

    img.show()


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

    known_ids = ["9edb5140-d4a5-835d-e3e1-64b6d9205254", "048c6528-650e-396c-650f-cb6a276c3eab"]

    path = args.images_dir
    paths = [os.path.join(path, it) for it in os.listdir(path)]
    # paths = [os.path.join(path, f"{it}.png") for it in known_ids]

    for idx, img in enumerate(getImages(paths)):
        print(idx)
        hasPlaster(img)
        experimentalplaster(img)
        if idx > 5: break
        # img.show()

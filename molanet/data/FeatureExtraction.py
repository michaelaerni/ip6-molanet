import argparse
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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


def hasPlaster(imagepath: str) -> bool:
    """
    Detects if an image has non skin colored regions
    the regions we wish to identify typically have the following properties:
    * sized about 1/10th the size of the image
    * very different color from the usual skin color (blue,yellow,purple,bright red)
    * often circular (but not alays)
    *

    :return: true if we are confident the image has a plaseter, false otherwise
    """
    img = cv2.imread(imagepath)

    # resize
    im512 = cv2.cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    # get 16 patches of size 128,128


    def analyzeYCrCb(img):
        """
           See: CHAI AND NGAN: FACE SEGMENTATION USING SKIN-COLOR MAP p. 555 in IEE
           RCR_MIN = 133
           RCR_MAX = 173
           RCB_MIN = 77
           RCB_MAX = 127

           convert RGB to YCrCb
           """

        RCR_MIN = 130
        RCR_MAX = 180
        RCB_MIN = 77
        RCB_MAX = 127

        imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        colors = ('y', 'r', 'b')
        hist = []

        for idx, color in enumerate(colors):
            h = cv2.calcHist([imgYCC], [idx], None, [256], [0, 256])
            n = cv2.normalize(h, 0, 1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            hist.append(n)

        lowerCr = sum(hist[1][0:RCR_MIN])
        higherCr = sum(hist[1][RCR_MAX:])
        lowerCb = sum(hist[2][0:RCB_MIN])
        higherCb = sum(hist[2][RCB_MAX:])

        crSkin = sum(hist[1][RCR_MIN:RCR_MAX])
        cbSkin = sum(hist[2][RCB_MIN:RCB_MAX])

        crOutside = lowerCr + higherCr
        cr = sum(hist[1])

        cboutside = lowerCb + higherCb
        cb = sum(hist[2])

        print(lowerCr, higherCr, lowerCr + higherCr)
        print(lowerCb, higherCb, lowerCb + higherCb)
        print(f"cr skin={crSkin/cr} noskin={crOutside/cr}")
        print(f"cb skin={cbSkin/cb} noskin={cboutside/cb}")

        # plotting
        plt.subplot(311)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.axes("off")
        plt.subplot(312)

        for idx, h in enumerate(hist):
            plt.plot(h, colors[idx])

        plt.xlim([0, 255])
        plt.show()

    return False


def countNonSkinPixels(img: Image) -> bool:
    a: np.ndarray = np.asarray(img, dtype=np.uint8)

    d = [tuple(v) for m2d in a for v in m2d]  # wtf stackoverflow
    assert len(d) == a.size // 3
    # print(c)

    skinpix = [(r, g, b) for r, g, b in d if isSkin(r, g, b)]

    print(f"skin pixels {len(skinpix)/len(d)}, non-skin pixels: {len(d)-len(skinpix)/len(d)}")

def experimentalplaster(img: Image):
    assert img.mode == 'RGB'
    img.show()
    img = img.copy()
    (width, height) = img.size
    pix = img.load()

    for y in range(height):
        for x in range(width):
            (r, g, b) = pix[x, y]
            if not (isSkin(r, g, b) or (r == 0 and b == 0 and g == 0)):
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
    paths = [os.path.join(path, f"{it}.png") for it in known_ids]

    for idx, img in enumerate(paths):
        hasPlaster(img)
        #experimentalplaster(img)
        if idx > 5: break
        # img.show()

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy import sum


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


def analyzeYCrCb(img: np.ndarray, debug: bool = False) -> float:
    """
    Checks if an image contains non-skin pixels



    :param img: image to be analyzed
    :param debug: print extensive debug info and show segmentation
    :return: [0,1] confidence the image contains non-skin pixels
    """



    """
    Using chroma analyis
       See: CHAI AND NGAN: FACE SEGMENTATION USING SKIN-COLOR MAP p. 555 in IEEE
       RCR_MIN = 133
       RCR_MAX = 173
       RCB_MIN = 77
       RCB_MAX = 127
    
       convert RGB to YCrCb
    """

    RCR_MIN = 133
    RCR_MAX = 173
    RCB_MIN = 77
    RCB_MAX = 127

    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    hist = []

    colors = ('y', 'r', 'b')
    for idx, color in enumerate(colors):
        h = cv2.calcHist([imgYCC], [idx], None, [256], [0, 256])
        n = cv2.normalize(h, 0, 1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        hist.append(n)

    crSkin = sum(hist[1][RCR_MIN:RCR_MAX])
    cbSkin = sum(hist[2][RCB_MIN:RCB_MAX])
    cr = sum(hist[1])
    cb = sum(hist[2])

    avgY = np.mean(img[:, :, 0])
    avgCr = np.mean(img[:, :, 1])
    avgCb = np.mean(img[:, :, 2])

    isNonSkinConfidence = max(1.0 - (cbSkin / cb), 1.0 - (crSkin / cr))

    # sometimes there are huge white areas which would be classified as non skin
    # if they are non-skin then Cr Cb are very close and Y is high

    if np.abs(avgCb - avgCr) < 6 or \
            ((avgY > 155 or avgY < 50) and np.abs(avgCb - avgCr) < 35):  isNonSkinConfidence = 0

    if debug:
        varY = np.median(img[:, :, 0])
        varCr = np.median(img[:, :, 1])
        varCb = np.median(img[:, :, 2])

        lowerCr = sum(hist[1][0:RCR_MIN])
        higherCr = sum(hist[1][RCR_MAX:])
        lowerCb = sum(hist[2][0:RCB_MIN])
        higherCb = sum(hist[2][RCB_MAX:])
        crOutside = lowerCr + higherCr

        print()
        print(f" avg={avgY},{varY} Cr={avgCr},{varCr} Cb={avgCb},{varCb}")
        print(f"cr (low,skin,high) {str(lowerCr), str(crSkin), str(higherCr)}")
        print(f"cb (low,skin,high) {str(lowerCb), str(cbSkin), str(higherCb),}")
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
    return isNonSkinConfidence


def hasPlaster(imagepath: str, minConfidence: float = 0.4) -> bool:
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
    im512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    def patches(img: np.ndarray, patchsize: int):
        for y in range(0, img.shape[0], patchsize):
            for x in range(0, img.shape[1], patchsize):
                yield img[y:y + patchsize, x:x + patchsize]

    for idx, patch in enumerate(patches(img, 256)):
        if (analyzeYCrCb(patch) > minConfidence):
            cv2.imshow(imagepath.split(os.sep)[-1], im512)
            analyzeYCrCb(patch, debug=True)
            break


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
        hasPlaster(img)
        #experimentalplaster(img)
        # img.show()

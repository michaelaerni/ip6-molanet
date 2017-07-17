import math
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt


def contains_hair(image, line_threshold=10) -> bool:
    threshold1 = 150
    threshold2 = 200
    aperture_size = 5
    rho = 1
    theta = np.pi / 180
    threshold_hough = 80
    min_line_length = 20

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, threshold1, threshold2, apertureSize=aperture_size, L2gradient=True)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold_hough, minLineLength=min_line_length)

    return lines is not None and len(lines) >= line_threshold


def calculate_mole_sizes(mask: np.ndarray) -> Tuple[float, int]:
    absolute_size = int(np.sum(mask) / np.max(mask) / mask.shape[2])
    absolute_height, absolute_width, _ = mask.shape
    relative_size = absolute_size / float(absolute_width * absolute_height)
    return relative_size, absolute_size


def calculate_pointwise_statistics(image) -> Tuple[float, float, float]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_values = grayscale.shape[0] * grayscale.shape[1]
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    normalized_histogram = histogram / float(total_values)

    mean = 0
    value_count = 0
    median = -1
    for value, probability in enumerate(normalized_histogram):
        mean += value * probability[0]
        value_count += histogram[value][0]

        if median < 0:
            if value_count >= total_values // 2:
                median = float(value)
            elif total_values % 2 == 0 and value_count == total_values // 2 - 1:
                median = (value + value - 1) / 2.0

    variance = 0
    for value, probability in enumerate(normalized_histogram):
        variance += probability[0] * (value - mean) ** 2

    return mean, median, math.sqrt(variance)


def contains_plaster(image: np.ndarray, minConfidence: float = 0.4, debug=False) -> bool:
    def patches(img: np.ndarray, patchsize: int):
        for y in range(0, img.shape[0], patchsize):
            for x in range(0, img.shape[1], patchsize):
                yield img[y:y + patchsize, x:x + patchsize]

    def analyzeYCrCb(img: np.ndarray, debug: bool = False) -> float:
        """
        Checks if an image contains non-skin pixels using chroma analyis

        See: CHAI AND NGAN: FACE SEGMENTATION USING SKIN-COLOR MAP p. 555 in IEEE
        :param img: image to be analyzed
        :param debug: print extensive debug info and show segmentation
        :return: [0,1] confidence the image contains non-skin pixels
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

    for idx, patch in enumerate(patches(image, 256)):
        if (analyzeYCrCb(patch) > minConfidence):
            if debug: analyzeYCrCb(patch, debug)
            return True

    return False

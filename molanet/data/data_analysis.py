from typing import Tuple

import cv2
import numpy as np
import math


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


def calculate_mole_sizes(mask) -> Tuple[float, int]:
    histogram = cv2.calcHist([mask], [0], None, [256], [0, 256])
    absolute_size = int(histogram[255])
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

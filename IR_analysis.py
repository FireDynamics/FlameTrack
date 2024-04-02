import numpy as np
import cv2
from collections import deque

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton


def read_IR_data(filename: str) -> np.ndarray:
    """
    Read the IR data from the file. The data is expected to be in the [Data] section of the file, separated by ';'
    :param filename: filepath to the IR data file
    :return: IR data as numpy array
    """
    with open(filename, 'r', encoding='latin-1') as f:
        line = f.readline()
        while line:
            if line.startswith('[Data]'):
                return np.genfromtxt((line.replace(',','.') for line in f.readlines()), delimiter=';')
            line = f.readline()

    raise ValueError('No data found in file, check file format!')


def dewarp_data(data, corners, target_pixels_width= None, target_pixels_height= None, target_ratio = None) -> np.ndarray:
    """
    Dewarp the data using the corners and the target pixels width and height
    :param target_ratio: target ratio of the dewarped data
    :param data: IR image data to be dewarped
    :param corners: selected corners of the data
    :param target_pixels_width: target width of the dewarped data
    :param target_pixels_height: target height of the dewarped data
    :return: dewarped data as numpy array
    """
    buffer = 1.1
    if all(x is None for x in [target_pixels_width, target_pixels_height, target_ratio]):
        raise ValueError('Either target_pixels_width and target_pixels_height or target_ratio must be provided')

    source_corners = np.array(corners, dtype=np.float32)
    if target_pixels_width is None and target_pixels_height is None:
        max_width = max(source_corners[1][0] - source_corners[0][0], source_corners[2][0] - source_corners[3][0])
        max_height = max(source_corners[2][1] - source_corners[1][1], source_corners[3][1] - source_corners[0][1])
        target_pixels_height = int(max(max_height, max_width * target_ratio)*buffer)
        target_pixels_width = int(target_pixels_height / target_ratio)
    target_corners = np.array(
        [[0, 0], [target_pixels_width, 0], [target_pixels_width, target_pixels_height], [0, target_pixels_height]],
        dtype=np.float32)

    # Use getPerspectiveTransform instead of findHomography. findHomography is useful for multiple points since it is
    # able to reject outliers, since only 4 points are used, getPerspectiveTransform is sufficient
    transformation_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)
    # apply the transformation matrix to the data
    dewarped_data = cv2.warpPerspective(data, transformation_matrix, (target_pixels_width, target_pixels_height))
    return dewarped_data



def sort_corner_points(points):
    """
    Sort the corner points in the following order: top-left, top-right, bottom-right, bottom-left
    :param points: corner points
    :return: sorted corner points
    """
    points = np.array(points)
    diff = np.diff(points, axis=1)
    summ = points.sum(axis=1)
    return np.array([
        points[np.argmin(summ)],
        points[np.argmin(diff)],
        points[np.argmax(summ)],
        points[np.argmax(diff)]
    ])





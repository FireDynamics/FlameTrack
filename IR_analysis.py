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


def dewarp_data(data, corners, target_pixels_width, target_pixels_height) -> np.ndarray:
    """
    Dewarp the data using the corners and the target pixels width and height
    :param data: IR image data to be dewarped
    :param corners: selected corners of the data
    :param target_pixels_width: target width of the dewarped data
    :param target_pixels_height: target height of the dewarped data
    :return: dewarped data as numpy array
    """

    source_corners = np.array(corners, dtype=np.float32)
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


def select_points_mpl(data, cmap='hot') -> np.ndarray:
    """
    Select points from the data using the mouse
    :param data: data to select points from
    :param cmap: colormap to use for the plot
    :return: points selected by the user
    """
    points = deque(maxlen=4)

    def on_click(event):
        if event.button is MouseButton.LEFT:
            points.append([event.xdata, event.ydata])
        lines = plt.gca().lines
        if len(lines) > 0:
            lines[-1].remove()
        x, y = list(zip(*points))
        x = list(x)
        y = list(y)
        x += [x[0]]
        y += [y[0]]
        plt.plot(x, y, linestyle='--', marker='x', color='red')
        plt.draw()

    def on_key_press(event):
        if event.key == 'q':
            print('Quitting')
            plt.close()

    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.connect('button_press_event', on_click)
    plt.connect('key_press_event', on_key_press)
    plt.show()
    return sort_corner_points(points)


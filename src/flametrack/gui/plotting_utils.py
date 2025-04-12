import numpy as np

def sort_corner_points(points) -> list:
    """
    Sort the points anti-clockwise starting from the top left corner
    :param points: list of points
    :return: sorted points
    """
    points = np.array(points)

    # Get origin
    origin = np.mean(points, axis=0)

    sort_by_angle = lambda x: np.arctan2(x[1] - origin[1], x[0] - origin[0])
    points = sorted(points, key=sort_by_angle, reverse=True)
    return points
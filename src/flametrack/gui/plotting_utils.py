import cv2
import numpy as np


def sort_corner_points(
    points, experiment_type="Room Corner", direction="clockwise"
) -> list:
    """
    Sort corner points depending on experiment type.
    - For 'room_corner' (6 points): Sort using angle-based method with defined start point
    - For 'lateral_flame_spread' (4 points): Sort using center angle
    """
    if experiment_type == "Room Corner":
        if len(points) != 6:
            raise ValueError("Room corner expects exactly 6 points.")

        pts = np.array(points)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sort_order = np.argsort(angles)

        if direction == "clockwise":
            sort_order = sort_order[::-1]

        sorted_pts = pts[sort_order]

        # Beginne mit dem Punkt ganz links oben
        top_idx = np.lexsort((sorted_pts[:, 1], sorted_pts[:, 0]))[0]
        sorted_pts = np.roll(sorted_pts, -top_idx, axis=0)

        return [tuple(pt) for pt in sorted_pts]

    if experiment_type == "Lateral Flame Spread":
        if len(points) != 4:
            raise ValueError("Lateral flame spread expects exactly 4 points.")

        pts = np.array(points)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

        sort_order = (
            np.argsort(-angles) if direction == "clockwise" else np.argsort(angles)
        )

        return [tuple(pts[i]) for i in sort_order]

    raise ValueError(f"Unknown experiment type: {experiment_type}")


# import numpy as np
#
# def sort_corner_points(points, direction: str = "clockwise") -> list:
#     if len(points) != 6:
#         raise ValueError("Expected exactly 6 points.")
#
#     pts = np.array(points)
#     center = np.mean(pts, axis=0)
#     angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
#     sort_order = np.argsort(angles)
#     if direction == "clockwise":
#         sort_order = sort_order[::-1]
#     sorted_pts = pts[sort_order]
#
#     # Startpunkt = niedrigster X (links), dann Y (oben)
#     top_idx = np.lexsort((sorted_pts[:, 1], sorted_pts[:, 0]))[0]
#     sorted_pts = np.roll(sorted_pts, -top_idx, axis=0)
#
#     return [tuple(pt) for pt in sorted_pts]


def rotate_points(points, image_shape, rotation_index):
    """
    Converts points from a np.rot90-rotated display frame back to unrotated image coordinates.

    :param points: Liste von (x, y)-Punkten im rotierten Bild
    :param image_shape: Form des UNROTIERTEN Bildes als (Höhe, Breite)
    :param rotation_index: 0 = 0°, 1 = 90° CCW, 2 = 180°, 3 = 270° CCW
    :return: Punkte im unrotierten Bildkoordinatensystem als Liste von (x, y)
    """
    k = rotation_index % 4
    if k == 0:
        return points

    img_h, img_w = image_shape[0], image_shape[1]
    pts = np.array(points, dtype=np.float32)
    rx, ry = pts[:, 0], pts[:, 1]

    if k == 1:  # 90° CCW: inverse is x = W-1-ry, y = rx
        x_out, y_out = img_w - 1 - ry, rx
    elif k == 2:  # 180°: inverse is x = W-1-rx, y = H-1-ry
        x_out, y_out = img_w - 1 - rx, img_h - 1 - ry
    else:  # 270° CCW (= 90° CW): inverse is x = ry, y = H-1-rx
        x_out, y_out = ry, img_h - 1 - rx

    return np.stack([x_out, y_out], axis=1).tolist()

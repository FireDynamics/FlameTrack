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
    Rotiert Punkte um das Zentrum des Bildes entsprechend der Kamerarotation.

    :param points: Liste von (x, y)-Punkten
    :param image_shape: Form des Bildes als (Höhe, Breite)
    :param rotation_index: 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
    :return: Rotierte Punkte als Liste von (x, y)
    """
    if rotation_index % 4 == 0:
        return points

    angle = -rotation_index * 90  # im Uhrzeigersinn
    center = (image_shape[1] / 2, image_shape[0] / 2)  # (x, y)

    # Rotationsmatrix (2x3)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Punkte homogenisieren
    points_np = np.array(points, dtype=np.float32)
    points_h = np.hstack([points_np, np.ones((len(points_np), 1))])  # (N, 3)

    # Transformation anwenden
    rotated = (rotation_matrix @ points_h.T).T
    return rotated.tolist()

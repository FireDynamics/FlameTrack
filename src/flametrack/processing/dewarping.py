import logging
import os
from datetime import datetime
from typing import Generator, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

# NEU: init/assert importieren
from flametrack.analysis.dataset_handler import (
    create_h5_file,
    init_h5_for_experiment,
    assert_h5_schema,
)
from flametrack.analysis.IR_analysis import (
    compute_remap_from_homography,
    get_dewarp_parameters,
)
from flametrack.gui.plotting_utils import rotate_points, sort_corner_points
from flametrack.utils.math_utils import estimate_resolution_from_points

DATATYPE = "IR"


def dewarp_room_corner_remap(
    experiment,
    points: Sequence[Tuple[float, float]],
    target_ratio: float,
    target_pixels_width: int,
    target_pixels_height: int,
    plate_width_mm: Optional[float] = None,
    plate_height_mm: Optional[float] = None,
    rotation_index: int = 0,
    filename: Optional[str] = None,
    frequency: int = 1,
    testing: bool = False,
) -> Generator[int, None, None]:
    """
    Dewarp IR frames from a room corner experiment using precomputed remap grids.
    """
    logging.info("[DEWARP] Starting room corner dewarping (REMAPPING)")

    if len(points) != 6:
        raise ValueError("Expected exactly 6 points for room corner dewarping.")
    if target_pixels_width <= 10 or target_pixels_height <= 10:
        raise ValueError("Target image size too small for meaningful dewarping.")

    points = np.array(points, dtype=np.float32)
    points_left = points[[0, 1, 4, 5]]
    points_right = points[[1, 2, 3, 4]]

    frame_shape = experiment.get_data(DATATYPE).get_frame(0, 0).shape
    selected_left = rotate_points(points_left, frame_shape, rotation_index)
    selected_right = rotate_points(points_right, frame_shape, rotation_index)

    dewarp_params_left = get_dewarp_parameters(
        selected_left, target_pixels_width, target_pixels_height, target_ratio
    )
    dewarp_params_right = get_dewarp_parameters(
        selected_right, target_pixels_width, target_pixels_height, target_ratio
    )

    if filename is None:
        processed_folder = os.path.join(experiment.folder_path, "processed_data")
        os.makedirs(processed_folder, exist_ok=True)
        filename = os.path.join(
            processed_folder, f"{experiment.exp_name}_results_RCE.h5"
        )

    if os.path.exists(filename):
        raise FileExistsError(filename)
    if experiment.h5_file is not None:
        experiment.h5_file.close()

    with create_h5_file(filename=filename) as h5_file:
        # Root-Attribute
        if plate_width_mm is not None:
            h5_file.attrs["plate_width_mm_left"] = float(plate_width_mm)
            h5_file.attrs["plate_width_mm_right"] = float(plate_width_mm)
        if plate_height_mm is not None:
            h5_file.attrs["plate_height_mm_left"] = float(plate_height_mm)
            h5_file.attrs["plate_height_mm_right"] = float(plate_height_mm)

        # Schema initialisieren + prüfen
        init_h5_for_experiment(h5_file, "Room Corner")
        assert_h5_schema(h5_file, "Room Corner")

        for side, params, pts in zip(
            ["left", "right"],
            [dewarp_params_left, dewarp_params_right],
            [selected_left, selected_right],
        ):
            # WICHTIG: require_group statt create_group (Gruppen können schon existieren)
            grp = h5_file.require_group(f"dewarped_data_{side}")

            # Attribute setzen/aktualisieren
            grp.attrs.update(
                {
                    "transformation_matrix": params["transformation_matrix"],
                    "target_pixels_width": params["target_pixels_width"],
                    "target_pixels_height": params["target_pixels_height"],
                    "target_ratio": params["target_ratio"],
                    "selected_points": pts,
                    "frame_range": [0, experiment.get_data(DATATYPE).get_frame_count()],
                    "points_selection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error_unit": "pixels",
                    "plate_width_mm": plate_width_mm,
                    "plate_height_mm": plate_height_mm,
                }
            )

            # Resolution (best effort)
            try:
                p0, p1, p3 = pts[0], pts[1], pts[3]
                res = estimate_resolution_from_points(
                    p0, p1, p3, plate_width_mm, plate_height_mm
                )
                grp.attrs["plate_width_mm"] = plate_width_mm
                grp.attrs["plate_height_mm"] = plate_height_mm
                grp.attrs.update(res)
            except Exception as e:
                logging.warning(f"[DEWARP] Resolution estimation failed for {side}: {e}")

            dset_h = params["target_pixels_height"]
            dset_w = params["target_pixels_width"]

            # Sauber neu anlegen
            if "data" in grp:
                del grp["data"]
            grp.create_dataset(
                "data",
                (dset_h, dset_w, 1),
                maxshape=(dset_h, dset_w, None),
                chunks=(dset_h, dset_w, 1),
                dtype=np.float32,
            )

            H_inv = np.linalg.inv(params["transformation_matrix"])
            src_x, src_y = compute_remap_from_homography(H_inv, dset_w, dset_h)

            if "src_x" in grp:
                del grp["src_x"]
            if "src_y" in grp:
                del grp["src_y"]
            grp.create_dataset("src_x", data=src_x)
            grp.create_dataset("src_y", data=src_y)

        # Frames verarbeiten
        data = experiment.get_data(DATATYPE)
        data_numbers = data.data_numbers
        start, end = (
            (len(data_numbers) // 2 - 1, len(data_numbers) // 2 + 1)
            if testing
            else (0, len(data_numbers))
        )

        for i, idx in enumerate(data_numbers[start:end:frequency]):
            frame = data.get_raw_frame(idx)

            for side in ["left", "right"]:
                grp = h5_file[f"dewarped_data_{side}"]
                dset = grp["data"]
                src_x_map, src_y_map = cv2.convertMaps(
                    grp["src_x"][()], grp["src_y"][()], cv2.CV_16SC2
                )

                dewarped = cv2.remap(
                    frame,
                    np.clip(src_x_map, 0, frame.shape[1] - 1),
                    np.clip(src_y_map, 0, frame.shape[0] - 1),
                    interpolation=cv2.INTER_LINEAR,
                )

                dset.resize((dset.shape[0], dset.shape[1], i + 1))
                dset[:, :, i] = dewarped

            yield i

    experiment.h5_file = h5py.File(filename, "r+")
    experiment.h5_path = filename



def dewarp_lateral_flame_spread(
    experiment,
    points: Sequence[Tuple[float, float]],
    target_ratio: float,
    target_pixels_width: int,
    target_pixels_height: int,
    plate_width_mm: Optional[float] = None,
    plate_height_mm: Optional[float] = None,
    rotation_index: int = 0,
    filename: Optional[str] = None,
    frequency: int = 1,
    testing: bool = False,
) -> Generator[int, None, None]:
    """
    Dewarp IR frames from a lateral flame spread (LFS) experiment.

    Args:
        experiment: Experiment object containing the IR data.
        points: List of 4 corner points selected by the user.
        target_ratio: Desired mm/pixel scaling ratio.
        target_pixels_width: Pixel width of the dewarped image.
        target_pixels_height: Pixel height of the dewarped image.
        plate_width_mm: Physical width of the sample plate in mm.
        plate_height_mm: Physical height of the sample plate in mm.
        rotation_index: Number of 90-degree rotations to apply.
        filename: Output HDF5 file path. If None, generated automatically.
        frequency: Frame skipping interval.
        testing: If True, only a small frame subset will be processed.

    Yields:
        Index of the currently processed frame.
    """
    logging.info("[DEWARP] Starting LFS dewarping")

    # Input validation
    if target_pixels_width is None or target_pixels_width <= 10:
        raise ValueError("target_pixels_width must be greater than 10")
    if target_pixels_height is None or target_pixels_height <= 10:
        raise ValueError("target_pixels_height must be greater than 10")
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError("Exactly 4 corner points are required for LFS dewarping.")

    sorted_points = sort_corner_points(points, experiment_type="Lateral Flame Spread")

    dewarp_params = get_dewarp_parameters(
        sorted_points,
        target_pixels_width=target_pixels_width,
        target_pixels_height=target_pixels_height,
        target_ratio=target_ratio,
    )

    if filename is None:
        processed_folder = os.path.join(experiment.folder_path, "processed_data")
        os.makedirs(processed_folder, exist_ok=True)
        filename = os.path.join(
            processed_folder, f"{experiment.exp_name}_results_RCE.h5"
        )
    print(f"[DEWARP] experiment.exp_name = {experiment.exp_name}")
    try:
        with h5py.File(filename, "r") as f:
            if "dewarped_data" in f or "dewarped_data_left" in f:
                raise FileExistsError(filename)
    except OSError:
        pass  # Datei ist defekt oder leer → einfach überschreiben
    if experiment.h5_file:
        experiment.h5_file.close()

    with create_h5_file(filename=filename) as h5_file:
        # NEU: Basis-Schema für LFS anlegen + prüfen
        init_h5_for_experiment(h5_file, "Lateral Flame Spread")
        assert_h5_schema(h5_file, "Lateral Flame Spread")

        # Root attributes (Quick Access)
        if plate_width_mm is not None:
            h5_file.attrs["plate_width_mm"] = float(plate_width_mm)
        if plate_height_mm is not None:
            h5_file.attrs["plate_height_mm"] = float(plate_height_mm)

        if "dewarped_data" in h5_file:
            del h5_file["dewarped_data"]
        if "edge_results" in h5_file:
            del h5_file["edge_results"]

        grp = h5_file.create_group("dewarped_data")
        h5_file.create_group("edge_results")

        grp.attrs.update(
            {
                "transformation_matrix": dewarp_params["transformation_matrix"],
                "target_pixels_width": dewarp_params["target_pixels_width"],
                "target_pixels_height": dewarp_params["target_pixels_height"],
                "target_ratio": dewarp_params["target_ratio"],
                "selected_points": sorted_points,
                "frame_range": [0, experiment.get_data(DATATYPE).get_frame_count()],
                "points_selection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plate_width_mm": plate_width_mm,
                "plate_height_mm": plate_height_mm,
            }
        )

        try:
            p0, p1, p3 = sorted_points[0], sorted_points[1], sorted_points[3]
            res = estimate_resolution_from_points(
                p0, p1, p3, plate_width_mm, plate_height_mm
            )
            grp.attrs.update(res)
        except Exception as e:
            logging.warning(f"[DEWARP] Resolution estimation failed: {e}")

        dset_h = dewarp_params["target_pixels_height"]
        dset_w = dewarp_params["target_pixels_width"]

        dset = grp.create_dataset(
            "data",
            (dset_h, dset_w, 1),
            maxshape=(dset_h, dset_w, None),
            chunks=(dset_h, dset_w, 1),
            dtype=np.float32,
        )

        data = experiment.get_data(DATATYPE)
        data_numbers = data.data_numbers

        start, end = (
            (len(data_numbers) // 2 - 1, len(data_numbers) // 2 + 1)
            if testing
            else (0, len(data_numbers))
        )

        for i, idx in enumerate(data_numbers[start:end:frequency]):
            frame = data.get_frame(idx, rotation_index)
            dewarped = cv2.warpPerspective(
                frame,
                dewarp_params["transformation_matrix"],
                (dset_w, dset_h),
                flags=cv2.INTER_LINEAR,
            )
            dset.resize((dset_h, dset_w, i + 1))
            dset[:, :, i] = dewarped
            yield i

    experiment.h5_file = h5py.File(filename, "r+")
    experiment.h5_path = filename


def rotate_image_and_points(
    image: np.ndarray, points: np.ndarray, angle_degrees: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate both image and corresponding points.

    Args:
        image: Input image.
        points: Nx2 array of (x, y) points.
        angle_degrees: Rotation angle in degrees.

    Returns:
        A tuple (rotated_image, rotated_points)
    """
    h_img, w_img = image.shape[:2]
    center = (w_img // 2, h_img // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    rotated_img = cv2.warpAffine(image, M, (w_img, h_img))
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    rotated_pts = (M @ points_h.T).T.astype(np.float32)

    return rotated_img, rotated_pts

# ==============================================================================
# ARCHIVED FUNCTION — no longer used in GUI (replaced by dewarp_room_corner_remap)
# ==============================================================================

# def dewarp_room_corner(...):  # ← original legacy code here
#
# def dewarp_room_corner(
#     experiment,
#     points,
#     target_ratio,
#     target_pixels_width=None,
#     target_pixels_height=None,
#     rotation_index=0,
#     filename=None,
#     frequency=1,
#     testing=False,
# ):
#     logging.info("[DEWARP] Starting room corner dewarping")
#
#     if len(points) != 6:
#         raise ValueError("Expected 6 points for room corner dewarping")
#
#     points = np.array(points)
#     selected_points_left = points[[0, 1, 4, 5]]
#     selected_points_right = points[[1, 2, 3, 4]]
#
#     dewarp_params_left = get_dewarp_parameters(
#         selected_points_left,
#         target_pixels_width=target_pixels_width,
#         target_pixels_height=target_pixels_height,
#         target_ratio=target_ratio,
#     )
#     dewarp_params_right = get_dewarp_parameters(
#         selected_points_right,
#         target_pixels_width=target_pixels_width,
#         target_pixels_height=target_pixels_height,
#         target_ratio=target_ratio,
#     )
#
#     if filename is None:
#         processed_folder = os.path.join(experiment.folder_path, "processed_data")
#         os.makedirs(processed_folder, exist_ok=True)
#         filename = os.path.join(
#             processed_folder, f"{experiment.exp_name}_results_RCE.h5"
#         )
#
#     if os.path.exists(filename):
#         raise FileExistsError(filename)
#
#     if experiment.h5_file is not None:
#         experiment.h5_file.close()
#
#     with create_h5_file(filename=filename) as h5_file:
#         h5_file.create_group("dewarped_data_left")
#         h5_file.create_group("dewarped_data_right")
#         h5_file.create_group("edge_results_left")
#         h5_file.create_group("edge_results_right")
#
#         for grp_name, dewarp_params, selected_pts in zip(
#             ["dewarped_data_left", "dewarped_data_right"],
#             [dewarp_params_left, dewarp_params_right],
#             [selected_points_left, selected_points_right],
#         ):
#             grp = h5_file[grp_name]
#             grp.attrs["transformation_matrix"] = dewarp_params["transformation_matrix"]
#             grp.attrs["target_pixels_width"] = dewarp_params["target_pixels_width"]
#             grp.attrs["target_pixels_height"] = dewarp_params["target_pixels_height"]
#             grp.attrs["target_ratio"] = dewarp_params["target_ratio"]
#             grp.attrs["selected_points"] = selected_pts
#             grp.attrs["frame_range"] = [
#                 0,
#                 experiment.get_data(DATATYPE).get_frame_count(),
#             ]
#             grp.attrs["points_selection_date"] = datetime.now().strftime(
#                 "%Y-%m-%d %H:%M:%S"
#             )
#
#             dset_h = dewarp_params["target_pixels_height"]
#             dset_w = dewarp_params["target_pixels_width"]
#
#             grp.create_dataset(
#                 "data",
#                 (dset_h, dset_w, 1),
#                 maxshape=(dset_h, dset_w, None),
#                 chunks=(dset_h, dset_w, 1),
#                 dtype=np.float32,
#             )
#
#         for progress in dewarp_RCE_exp(
#             experiment,
#             rotation_index,
#             testing=testing,
#             frequency=frequency,
#             data_type=DATATYPE,
#         ):
#             yield progress
#
#     experiment.h5_file = h5py.File(filename, "r+")

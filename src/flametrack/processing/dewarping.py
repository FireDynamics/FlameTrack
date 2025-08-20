import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, Optional, Sequence, Tuple, cast

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

from flametrack.analysis.dataset_handler import (
    assert_h5_schema,
    create_h5_file,
    init_h5_for_experiment,
)
from flametrack.analysis.ir_analysis import (
    compute_remap_from_homography,
    get_dewarp_parameters,
)
from flametrack.gui.plotting_utils import rotate_points, sort_corner_points
from flametrack.utils.math_utils import estimate_resolution_from_points


@dataclass(frozen=True)
class DewarpConfig:
    target_ratio: float
    target_pixels_width: int
    target_pixels_height: int
    plate_width_mm: Optional[float] = None
    plate_height_mm: Optional[float] = None
    rotation_index: int = 0
    frequency: int = 1
    testing: bool = False
    filename: Optional[str] = None  # optional override output path


@dataclass(frozen=True)
class CornerSets:
    left: NDArray[np.float32]
    right: NDArray[np.float32]  # für Room Corner; bei LFS nur "left" nutzen


DATATYPE = "IR"

# dewarping.py – Hilfsfunktionen einfügen


def _ensure_output_path(experiment: Any, filename: Optional[str]) -> str:
    """Erstellt Standardpfad, falls kein filename gesetzt ist."""
    if filename is None:
        processed_folder = os.path.join(experiment.folder_path, "processed_data")
        os.makedirs(processed_folder, exist_ok=True)
        return os.path.join(processed_folder, f"{experiment.exp_name}_results_RCE.h5")
    return filename


def _init_room_corner_schema(h5f: h5py.File) -> None:
    init_h5_for_experiment(h5f, "Room Corner")
    assert_h5_schema(h5f, "Room Corner")


def _init_lfs_schema(h5f: h5py.File) -> None:
    init_h5_for_experiment(h5f, "Lateral Flame Spread")
    assert_h5_schema(h5f, "Lateral Flame Spread")


def _write_root_plate_attrs(
    h5f: h5py.File, w_mm: Optional[float], h_mm: Optional[float], room_corner: bool
) -> None:
    if room_corner:
        if w_mm is not None:
            h5f.attrs["plate_width_mm_left"] = float(w_mm)
            h5f.attrs["plate_width_mm_right"] = float(w_mm)
        if h_mm is not None:
            h5f.attrs["plate_height_mm_left"] = float(h_mm)
            h5f.attrs["plate_height_mm_right"] = float(h_mm)
    else:
        if w_mm is not None:
            h5f.attrs["plate_width_mm"] = float(w_mm)
        if h_mm is not None:
            h5f.attrs["plate_height_mm"] = float(h_mm)


def _ensure_dataset(group: h5py.Group, shape_hw: tuple[int, int]) -> h5py.Dataset:
    """Legt 'data' neu an (float32, chunked, grow‑only in t‑Richtung)."""
    h, w = shape_hw
    if "data" in group:
        del group["data"]
    return group.create_dataset(
        "data",
        (h, w, 1),
        maxshape=(h, w, None),
        chunks=(h, w, 1),
        dtype=np.float32,
    )


def _compute_remap_maps(
    homography: NDArray[np.float32], w: int, h: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    homography_inv = np.linalg.inv(homography)
    src_x, src_y = compute_remap_from_homography(homography_inv, w, h)
    return src_x.astype(np.float32), src_y.astype(np.float32)


def _store_remap(
    group: h5py.Group, src_x: NDArray[np.float32], src_y: NDArray[np.float32]
) -> None:
    for name in ("src_x", "src_y"):
        if name in group:
            del group[name]
    group.create_dataset("src_x", data=src_x)
    group.create_dataset("src_y", data=src_y)


# pylint: disable=too-many-locals
def dewarp_room_corner_remap(
    experiment: Any,
    points: NDArray[np.float32] | Sequence[Tuple[float, float]],
    config: DewarpConfig,
) -> Generator[int, None, None]:
    """Dewarp für Room Corner anhand vorberechneter Remap‑Grids."""
    logging.info("[DEWARP] Room Corner (remap) – start")

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] != 6:
        raise ValueError("Expected exactly 6 points for room corner dewarping.")
    if config.target_pixels_width <= 10 or config.target_pixels_height <= 10:
        raise ValueError("Target image size too small for meaningful dewarping.")

    # linke/rechte 4er‑Ecken aufteilen + optional rotieren
    frame_shape = experiment.get_data(DATATYPE).get_frame(0, 0).shape
    pts_left = pts[[0, 1, 4, 5]]
    pts_right = pts[[1, 2, 3, 4]]
    sel_left = rotate_points(pts_left, frame_shape, config.rotation_index)
    sel_right = rotate_points(pts_right, frame_shape, config.rotation_index)

    params_left = get_dewarp_parameters(
        sel_left,
        config.target_pixels_width,
        config.target_pixels_height,
        config.target_ratio,
    )
    params_right = get_dewarp_parameters(
        sel_right,
        config.target_pixels_width,
        config.target_pixels_height,
        config.target_ratio,
    )

    out_path = _ensure_output_path(experiment, config.filename)

    # Falls Datei existiert: Benutzerfluss wie gehabt – aber gezielt OSError/FileExistsError behandeln
    if os.path.exists(out_path):
        raise FileExistsError(out_path)
    if experiment.h5_file is not None:
        experiment.h5_file.close()

    with create_h5_file(filename=out_path) as h5f:
        _write_root_plate_attrs(
            h5f, config.plate_width_mm, config.plate_height_mm, room_corner=True
        )
        _init_room_corner_schema(h5f)

        for side, params, sel in (
            ("left", params_left, sel_left),
            ("right", params_right, sel_right),
        ):
            grp = h5f.require_group(f"dewarped_data_{side}")
            grp.attrs.update(
                {
                    "transformation_matrix": params["transformation_matrix"],
                    "target_pixels_width": params["target_pixels_width"],
                    "target_pixels_height": params["target_pixels_height"],
                    "target_ratio": params["target_ratio"],
                    "selected_points": sel,
                    "frame_range": [0, experiment.get_data(DATATYPE).get_frame_count()],
                    "points_selection_date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "error_unit": "pixels",
                    "plate_width_mm": config.plate_width_mm,
                    "plate_height_mm": config.plate_height_mm,
                }
            )
            # „Best effort“: Auflösung schätzen
            try:
                p0, p1, p3 = sel[0], sel[1], sel[3]
                res = estimate_resolution_from_points(
                    p0, p1, p3, config.plate_width_mm, config.plate_height_mm
                )
                grp.attrs.update(res)
            except (ValueError, TypeError) as err:
                logging.warning(
                    "[DEWARP] Resolution estimation failed for %s: %s",
                    side,
                    err,
                    exc_info=False,
                )

            h_out = int(params["target_pixels_height"])
            w_out = int(params["target_pixels_width"])
            data_dset: h5py.Dataset = _ensure_dataset(grp, (h_out, w_out))

            # Remap vorbereiten
            src_x, src_y = _compute_remap_maps(
                np.asarray(params["transformation_matrix"], dtype=np.float32),
                w_out,
                h_out,
            )
            _store_remap(grp, src_x, src_y)

        # Frames schleifen
        data = experiment.get_data(DATATYPE)
        frames = data.data_numbers
        start, end = (
            (len(frames) // 2 - 1, len(frames) // 2 + 1)
            if config.testing
            else (0, len(frames))
        )

        for i, idx in enumerate(frames[start : end : config.frequency]):
            raw = data.get_raw_frame(idx)

            for side in ("left", "right"):
                grp = h5f[f"dewarped_data_{side}"]

                data_dset = cast(h5py.Dataset, grp["data"])
                if not isinstance(data_dset, h5py.Dataset):
                    raise TypeError("Expected HDF5 dataset at 'data'")

                src_x_dset: h5py.Dataset = grp["src_x"]
                src_y_dset: h5py.Dataset = grp["src_y"]
                if not isinstance(src_x_dset, h5py.Dataset) or not isinstance(
                    src_y_dset, h5py.Dataset
                ):
                    raise TypeError("Expected HDF5 datasets 'src_x' and 'src_y'")
                assert isinstance(src_x_dset, h5py.Dataset)
                assert isinstance(src_y_dset, h5py.Dataset)

                src_x = np.asarray(src_x_dset[()], dtype=np.float32)
                src_y = np.asarray(src_y_dset[()], dtype=np.float32)
                map_x, map_y = cv2.convertMaps(
                    src_x.astype(np.float32), src_y.astype(np.float32), cv2.CV_16SC2
                )

                # Grenzen clippen, um cv2.remap sicher zu füttern
                h_in, w_in = raw.shape[:2]
                remapped = cv2.remap(
                    raw,
                    np.clip(map_x, 0, w_in - 1),
                    np.clip(map_y, 0, h_in - 1),
                    interpolation=cv2.INTER_LINEAR,
                )

                # pylint: disable=no-member
                data_dset.resize((h_out, w_out, i + 1))  # pylint: disable=no-member
                data_dset[:, :, i] = remapped.astype(np.float32, copy=False)

            yield i

    experiment.h5_file = h5py.File(out_path, "r+")
    experiment.h5_path = out_path


# pylint: disable=too-many-locals
def dewarp_lateral_flame_spread(
    experiment: Any,
    points: Sequence[Tuple[float, float]],
    config: DewarpConfig,
) -> Generator[int, None, None]:
    """Dewarp für Lateral Flame Spread (warpPerspective)."""
    logging.info("[DEWARP] LFS – start")

    if config.target_pixels_width <= 10:
        raise ValueError("target_pixels_width must be greater than 10")
    if config.target_pixels_height <= 10:
        raise ValueError("target_pixels_height must be greater than 10")
    if len(points) != 4:
        raise ValueError("Exactly 4 corner points are required for LFS dewarping.")

    sorted_pts = sort_corner_points(points, experiment_type="Lateral Flame Spread")
    params = get_dewarp_parameters(
        sorted_pts,
        target_pixels_width=config.target_pixels_width,
        target_pixels_height=config.target_pixels_height,
        target_ratio=config.target_ratio,
    )

    out_path = _ensure_output_path(experiment, config.filename)

    # Datei nur blockieren, wenn schon sinnvolle Inhalte drin sind
    try:
        with h5py.File(out_path, "r") as h5f:
            if "dewarped_data" in h5f or "dewarped_data_left" in h5f:
                raise FileExistsError(out_path)
    except OSError:
        pass

    if experiment.h5_file:
        experiment.h5_file.close()

    with create_h5_file(filename=out_path) as h5f:
        _init_lfs_schema(h5f)
        _write_root_plate_attrs(
            h5f, config.plate_width_mm, config.plate_height_mm, room_corner=False
        )

        if "dewarped_data" in h5f:
            del h5f["dewarped_data"]
        if "edge_results" in h5f:
            del h5f["edge_results"]

        grp = h5f.create_group("dewarped_data")
        h5f.create_group("edge_results")

        grp.attrs.update(
            {
                "transformation_matrix": params["transformation_matrix"],
                "target_pixels_width": int(params["target_pixels_width"]),
                "target_pixels_height": int(params["target_pixels_height"]),
                "target_ratio": float(params["target_ratio"]),
                "selected_points": np.asarray(sorted_pts, dtype=np.float32),
                "frame_range": [0, experiment.get_data(DATATYPE).get_frame_count()],
                "points_selection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plate_width_mm": config.plate_width_mm,
                "plate_height_mm": config.plate_height_mm,
            }
        )

        # „Best effort“: Auflösung schätzen
        try:
            p0, p1, p3 = sorted_pts[0], sorted_pts[1], sorted_pts[3]
            res = estimate_resolution_from_points(
                p0, p1, p3, config.plate_width_mm, config.plate_height_mm
            )
            grp.attrs.update(res)
        except (ValueError, TypeError) as err:
            logging.warning(
                "[DEWARP] Resolution estimation failed: %s", err, exc_info=False
            )

        h_out = int(params["target_pixels_height"])
        w_out = int(params["target_pixels_width"])
        dset = _ensure_dataset(grp, (h_out, w_out))

        data = experiment.get_data(DATATYPE)
        frames = data.data_numbers
        start, end = (
            (len(frames) // 2 - 1, len(frames) // 2 + 1)
            if config.testing
            else (0, len(frames))
        )

        homography_matrix = np.asarray(
            params["transformation_matrix"], dtype=np.float32
        )

        for i, idx in enumerate(frames[start : end : config.frequency]):
            frame = data.get_frame(idx, config.rotation_index)
            dewarped = cv2.warpPerspective(
                frame, homography_matrix, (w_out, h_out), flags=cv2.INTER_LINEAR
            )
            dset.resize((h_out, w_out, i + 1))
            dset[:, :, i] = dewarped.astype(np.float32, copy=False)
            yield i

    experiment.h5_file = h5py.File(out_path, "r+")
    experiment.h5_path = out_path


def rotate_image_and_points(
    image: NDArray[np.float32] | NDArray[np.uint8],
    points: NDArray[np.float32],
    angle_degrees: float,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
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
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    rotated_img = cv2.warpAffine(image, rotation_matrix, (w_img, h_img))
    points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    rotated_pts = (rotation_matrix @ points_h.T).T.astype(np.float32)
    return rotated_img.astype(np.float32, copy=False), rotated_pts


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
#                 "%Y-%m-%d %homography:%M:%S"
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

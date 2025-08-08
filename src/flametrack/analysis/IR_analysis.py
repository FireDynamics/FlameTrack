import logging

import cv2
import numpy as np

from flametrack.analysis import dataset_handler


def compute_remap_from_homography(H, width, height):
    """
    Compute remap grids from a homography matrix H.

    Args:
        H (np.ndarray): Homography matrix.
        width (int): Width of the target image.
        height (int): Height of the target image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: src_x and src_y remap matrices.
    """
    map_x, map_y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    ones = np.ones_like(map_x)
    target_coords = np.stack([map_x, map_y, ones], axis=-1).reshape(-1, 3).T

    source_coords = H @ target_coords
    source_coords /= source_coords[2, :]  # normalize

    src_x = source_coords[0, :].reshape((height, width)).astype(np.float32)
    src_y = source_coords[1, :].reshape((height, width)).astype(np.float32)

    return src_x, src_y


def read_IR_data(filename: str) -> np.ndarray:
    """
    Read raw IR data from CSV-like ASCII export format.

    Args:
        filename (str): Path to the IR data file.

    Returns:
        np.ndarray: 2D array of IR values.
    """
    with open(filename, "r", encoding="latin-1") as f:
        line = f.readline()
        while line:
            if line.startswith("[Data]"):
                return np.genfromtxt(
                    (line.replace(",", ".")[:-2] for line in f.readlines()),
                    delimiter=";",
                )
            line = f.readline()

    raise ValueError("No data found in file, check file format!")


def get_dewarp_parameters(
    corners,
    target_pixels_width=None,
    target_pixels_height=None,
    target_ratio=None,
    plate_width_m=None,
    plate_height_m=None,
    pixels_per_millimeter=1,
) -> dict:
    """
    Calculate the transformation matrix and target geometry for dewarping.

    Args:
        corners (array-like): Four corner points.
        target_pixels_width (int): Desired width in pixels.
        target_pixels_height (int): Desired height in pixels.
        target_ratio (float): Desired width/height ratio.
        plate_width_m (float): Real-world plate width (optional).
        plate_height_m (float): Real-world plate height (optional).
        pixels_per_millimeter (int): Resolution scaling factor.

    Returns:
        dict: Dictionary with keys: transformation_matrix, target_pixels_width,
              target_pixels_height, target_ratio.
    """
    buffer = 1.1
    source_corners = np.array(corners, dtype=np.float32)

    if plate_width_m is not None and plate_height_m is not None:
        target_pixels_width = int(plate_width_m * pixels_per_millimeter)
        target_pixels_height = int(plate_height_m * pixels_per_millimeter)

    if target_pixels_width is None or target_pixels_height is None:
        if target_ratio is None:
            raise ValueError("Either plate dimensions or target ratio must be provided")

        max_width = max(
            source_corners[1][0] - source_corners[0][0],
            source_corners[2][0] - source_corners[3][0],
        )
        max_height = max(
            source_corners[2][1] - source_corners[1][1],
            source_corners[3][1] - source_corners[0][1],
        )
        target_pixels_height = int(max(max_height, max_width / target_ratio) * buffer)
        target_pixels_width = int(target_pixels_height * target_ratio)

    target_corners = np.array(
        [
            [0, 0],
            [target_pixels_width, 0],
            [target_pixels_width, target_pixels_height],
            [0, target_pixels_height],
        ],
        dtype=np.float32,
    )

    transformation_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)

    return {
        "transformation_matrix": transformation_matrix,
        "target_pixels_width": target_pixels_width,
        "target_pixels_height": target_pixels_height,
        "target_ratio": target_pixels_height / target_pixels_width,
    }


# ==============================================================================
# ARCHIVED / UNUSED FUNCTIONS
# ==============================================================================

# def dewarp_exp(...):
#     """Legacy function for remapping IR data (used pre-v1.0)."""
#     ...

# def dewarp_RCE_exp(...):
#     """Used for room corner experiments, pre-remap standardization."""
#     ...

# def dewarp_data(data, dewarp_params) -> np.ndarray:
#     """
#     Dewarp the data using the transformation matrix and geometry.
#     """
#     transformation_matrix = dewarp_params["transformation_matrix"]
#     target_pixels_width = dewarp_params["target_pixels_width"]
#     target_pixels_height = dewarp_params["target_pixels_height"]
#     return cv2.warpPerspective(
#         data, transformation_matrix, (target_pixels_width, target_pixels_height)
#     )

# def sort_corner_points(points) -> list:
#     """
#     Sort the points anti-clockwise starting from the top left corner.
#     Deprecated – use plotting_utils.sort_corner_points instead.
#     """
#     points = np.array(points)
#     origin = np.mean(points, axis=0)
#     sort_by_angle = lambda x: np.arctan2(x[1] - origin[1], x[0] - origin[0])
#     return sorted(points, key=sort_by_angle, reverse=True)
#
# def dewarp_exp(
#     exp_name, data, rotation_index, frequency=10, testing=False, renew=False
# ):
#     dewarped_grp = dataset_handler.get_file(exp_name, "a").get("dewarped_data", None)
#     # Should not be necessary (but just in case keep commented)
#     # if dewarped_grp is None:
#     #     dewarped_grp = dataset_handler.get_file(exp_name, 'a').create_group('dewarped_data')
#     #     dset = dewarped_grp.get('data', None)
#
#     dset = dewarped_grp["data"]
#     metadata = dewarped_grp.attrs
#     data_numbers = data.data_numbers
#     dewarp_params = {}
#     start, end = metadata["frame_range"]
#     dset_w = metadata["target_pixels_width"]
#     dset_h = metadata["target_pixels_height"]
#     dewarp_params["transformation_matrix"] = metadata["transformation_matrix"]
#     dewarp_params["target_pixels_width"] = metadata["target_pixels_width"]
#     dewarp_params["target_pixels_height"] = metadata["target_pixels_height"]
#     dewarp_params["target_ratio"] = metadata["target_ratio"]
#     dewarp_params["selected_points"] = metadata["selected_points"]
#     dewarp_params["frame_range"] = metadata["frame_range"]
#
#     map_x = np.arange(0, dset_w, 1)
#     map_y = np.arange(0, dset_h, 1)
#     map_x, map_y = np.meshgrid(map_x, map_y)
#     transformation_matrix = np.linalg.inv(dewarp_params["transformation_matrix"])
#     src_x = (
#         transformation_matrix[0, 0] * map_x
#         + transformation_matrix[0, 1] * map_y
#         + transformation_matrix[0, 2]
#     ) / (
#         transformation_matrix[2, 0] * map_x
#         + transformation_matrix[2, 1] * map_y
#         + transformation_matrix[2, 2]
#     )
#     src_y = (
#         transformation_matrix[1, 0] * map_x
#         + transformation_matrix[1, 1] * map_y
#         + transformation_matrix[1, 2]
#     ) / (
#         transformation_matrix[2, 0] * map_x
#         + transformation_matrix[2, 1] * map_y
#         + transformation_matrix[2, 2]
#     )
#     if dewarped_grp.get("src_x", None) is not None:
#         del dewarped_grp["src_x"]
#     if dewarped_grp.get("src_y", None) is not None:
#         del dewarped_grp["src_y"]
#     dewarped_grp.create_dataset("src_x", data=src_x)
#     dewarped_grp.create_dataset("src_y", data=src_y)
#     assumed_pixel_error = 0.5
#     dewarped_grp.attrs["assumed_pixel_error"] = assumed_pixel_error
#     dewarped_grp.attrs["error_unit"] = "pixels"
#     src_points = np.array([src_x.flatten(), src_y.flatten()]).reshape(*src_x.shape, -1)
#
#     # err_x = assumed_pixel_error/np.linalg.norm(np.diff(src_points,axis=0),axis=2)
#     # err_y = assumed_pixel_error/np.linalg.norm(np.diff(src_points,axis=1),axis=2)
#     #
#     # dewarped_grp.create_dataset('err_x', data=err_x)
#     # dewarped_grp.create_dataset('err_y', data=err_y)
#     src_x_map, src_y_map = cv2.convertMaps(
#         src_x.astype(np.float32), src_y.astype(np.float32), cv2.CV_16SC2
#     )
#
#     if testing:
#         start = len(data_numbers) // 2 - 10
#         end = len(data_numbers) // 2 + 10
#
#     bar = progressbar.ProgressBar(max_value=len(data_numbers[start:end:frequency]))
#     for i, idx in bar(enumerate(data_numbers[start:end:frequency])):
#         if not renew and dset.shape[2] > i + 2:
#             continue
#         img = data.get_frame(idx, rotation_index)
#         dewarped_data = cv2.remap(
#             img, src_x_map, src_y_map, interpolation=cv2.INTER_LINEAR
#         )
#         dset.resize((dset_h, dset_w, i + 1))
#         dset[:, :, i] = dewarped_data
#     dataset_handler.close_file()
#     return src_x, src_y
#
#
# def dewarp_RCE_exp(
#     experiment, rotation_index, frequency=1, testing=False, renew=False, data_type="IR"
# ):
#     # Access the HDF5 file and get the data handler
#     h5_file = experiment.h5_file
#     data = experiment.get_data(data_type)
#
#     # Retrieve the groups for left and right dewarped data
#     dewarped_grp_left = h5_file["dewarped_data_left"]
#     dewarped_grp_right = h5_file["dewarped_data_right"]
#
#     # Prepare storage for the remap grids and metadata for both sides
#     src_x_maps = []
#     src_y_maps = []
#     dsets = []
#     dset_ws = []
#     dset_hs = []
#
#     for dewarped_grp in [dewarped_grp_left, dewarped_grp_right]:
#         # Retrieve the dataset and metadata
#         dset = dewarped_grp["data"]
#         metadata = dewarped_grp.attrs
#         data_numbers = data.data_numbers
#
#         # Get target dimensions and transformation matrix from metadata
#         dewarp_params = {}
#         start, end = metadata["frame_range"]
#         dset_w = metadata["target_pixels_width"]
#         dset_h = metadata["target_pixels_height"]
#         dewarp_params["transformation_matrix"] = metadata["transformation_matrix"]
#         dewarp_params["target_pixels_width"] = metadata["target_pixels_width"]
#         dewarp_params["target_pixels_height"] = metadata["target_pixels_height"]
#         dewarp_params["target_ratio"] = metadata["target_ratio"]
#         dewarp_params["selected_points"] = metadata["selected_points"]
#         dewarp_params["frame_range"] = metadata["frame_range"]
#
#         # Generate mapping grids for each pixel in the output image
#         map_x = np.arange(0, dset_w, 1)
#         map_y = np.arange(0, dset_h, 1)
#         map_x, map_y = np.meshgrid(map_x, map_y)
#
#         # Compute remap grid without inverting the matrix (to avoid flipping)
#         H = np.linalg.inv(dewarp_params["transformation_matrix"])
#         # H = dewarp_params["transformation_matrix"]
#         src_x, src_y = compute_remap_from_homography(H, dset_w, dset_h)
#
#         # Remove old remap data if present
#         if dewarped_grp.get("src_x", None) is not None:
#             del dewarped_grp["src_x"]
#         if dewarped_grp.get("src_y", None) is not None:
#             del dewarped_grp["src_y"]
#
#         # Store new remap maps in the HDF5 file
#         dewarped_grp.create_dataset("src_x", data=src_x)
#         dewarped_grp.create_dataset("src_y", data=src_y)
#
#         # Store additional metadata
#         assumed_pixel_error = 0.5
#         dewarped_grp.attrs["assumed_pixel_error"] = assumed_pixel_error
#         dewarped_grp.attrs["error_unit"] = "pixels"
#
#         # Convert float maps to OpenCV-compatible remap format
#         src_x_map, src_y_map = cv2.convertMaps(
#             src_x.astype(np.float32), src_y.astype(np.float32), cv2.CV_16SC2
#         )
#
#         # Store the mappings and metadata for use in the remapping loop
#         src_x_maps.append(src_x_map)
#         src_y_maps.append(src_y_map)
#         dsets.append(dset)
#         dset_ws.append(dset_w)
#         dset_hs.append(dset_h)
#
#     # For test mode, process only a central subset of frames
#     if testing:
#         start = len(data_numbers) // 2 - 50
#         end = len(data_numbers) // 2 + 50
#     print(f"[DEBUG] Frames to process: {data_numbers[start:end:frequency]}")
#     print(
#         f"[DEBUG] data_numbers: {data_numbers}, start: {start}, end: {end}, frequency: {frequency}"
#     )
#
#     # Loop over all selected frames and apply remapping
#     for i, idx in enumerate(data_numbers[start:end:frequency]):
#         for dset, src_x_map, src_y_map, dset_w, dset_h in zip(
#             dsets, src_x_maps, src_y_maps, dset_ws, dset_hs
#         ):
#
#             # Skip already existing frames unless forced with renew=True
#             if not renew and dset.shape[2] > i + 2:
#                 logging.debug(f"[DEBUG] Frame {i} already processed, skipping.")
#                 continue
#
#             # Get the original frame and apply remapping
#             img = data.get_frame(idx, rotation_index)
#             logging.debug(
#                 f"[DEBUG] Remap {idx}: img min/max = {np.min(img)}, {np.max(img)}"
#             )
#             logging.debug(
#                 "[DEBUG] Remap {idx}: src_x_map shape = {src_x_map.shape}, src_y_map shape = {src_y_map.shape}"
#             )
#
#             dewarped_data = cv2.remap(
#                 img, src_x_map, src_y_map, interpolation=cv2.INTER_LINEAR
#             )
#             logging.debug(
#                 "[DEBUG] Remap {idx}: dewarped min/max = {np.min(dewarped_data)}, {np.max(dewarped_data)}"
#             )
#
#             # Resize and store dewarped frame
#             dset.resize((dset_h, dset_w, i + 1))
#             dset[:, :, i] = dewarped_data
#             fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#             axs[0].imshow(img, cmap="gray")
#             axs[0].set_title(f"Input Frame {idx} (nach Rotation)")
#             axs[1].imshow(dewarped_data, cmap="gray")
#             axs[1].set_title(f"Dewarped Frame {idx}")
#             for ax in axs:
#                 ax.axis("off")
#             plt.tight_layout()
#             plt.show()
#
#         # Yield current frame index to update progress bar
#         yield idx
#
#
# def dewarp_data(data, dewarp_params) -> np.ndarray:
#     """
#     Dewarp the data using the corners and the target pixels width and height
#     :param data: data to dewarp
#     :param dewarp_params: dewarp parameters from get_dewarp_parameters
#     :return: dewarped data as numpy array
#     """
#     transformation_matrix = dewarp_params["transformation_matrix"]
#     target_pixels_width = dewarp_params["target_pixels_width"]
#     target_pixels_height = dewarp_params["target_pixels_height"]
#     dewarped_data = cv2.warpPerspective(
#         data, transformation_matrix, (target_pixels_width, target_pixels_height)
#     )
#     return dewarped_data
#
#
# # This method should be retired and will be succedded by the one in
# # "plotting_utils"
# def sort_corner_points(points) -> list:
#     """
#     Sort the points anti-clockwise starting from the top left corner
#     :param points: list of points
#     :return: sorted points
#     """
#     points = np.array(points)
#
#     # Get origin
#     origin = np.mean(points, axis=0)
#
#     sort_by_angle = lambda x: np.arctan2(x[1] - origin[1], x[0] - origin[0])
#     points = sorted(points, key=sort_by_angle, reverse=True)
#     return points

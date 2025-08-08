import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from scipy.signal import find_peaks, medfilt
from scipy.stats import skewnorm

from . import dataset_handler as dst_handler
from .dataset_handler import get_dewarped_data, save_edge_results

# =====================================================================================
# CORE FUNCTIONS: EDGE DETECTION LOGIC
# =====================================================================================


def find_peaks_in_gradient(y, min_distance=10, min_height=2, min_width=2):
    """
    Find peaks in the negative gradient of a 1D signal.

    Args:
        y (np.ndarray): 1D signal.
        min_distance (int): Minimum distance between peaks.
        min_height (float): Minimum peak height.
        min_width (float): Minimum peak width.

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    gradient = -np.gradient(y)
    peaks, _ = find_peaks(
        gradient, height=min_height, distance=min_distance, width=min_width
    )
    return peaks


def right_most_point_over_threshold(y, threshold=0, params=None):
    """
    Find the last point in the signal above the given threshold.

    Args:
        y (np.ndarray): 1D signal.
        threshold (float): Threshold value.
        params: Unused; kept for compatibility.

    Returns:
        int: Index of the last point above threshold, or 0.
    """
    peaks = np.where(y > threshold)[0]
    return peaks[-1] if len(peaks) else 0


def left_most_point_over_threshold(y, threshold=0, params=None):
    """
    Find the first point in the signal above the given threshold.

    Args:
        y (np.ndarray): 1D signal.
        threshold (float): Threshold value.
        params: Unused; kept for compatibility.

    Returns:
        int: Index of the first point above threshold, or len(y).
    """
    peaks = np.where(y > threshold)[0]
    return peaks[0] if len(peaks) else len(y)


def right_most_peak(y, min_distance=10, min_height=2, min_width=2):
    """
    Return the right-most peak in the gradient of the signal.

    Returns:
        int: Index of last detected peak, or 0.
    """
    peaks = find_peaks_in_gradient(y, min_distance, min_height, min_width)
    return peaks[-1] if len(peaks) else 0


def highest_peak(y, min_distance=10, min_height=2, min_width=2):
    """
    Return the index of the peak with the highest gradient.

    Returns:
        int: Index of the highest peak, or 0.
    """
    gradient = -np.gradient(y)
    peaks = find_peaks_in_gradient(y, min_distance, min_height, min_width)
    return peaks[np.argmax(gradient[peaks])] if len(peaks) else 0


def highest_peak_to_lowest_value(
    y,
    min_distance=10,
    min_height=2,
    min_width=2,
    ambient_weighting=2,
    high_val=0,
    low_val=1e10,
    direction_weighting=0.0,
    previous_peak=None,
    previous_velocity=0,
):
    """
    Find the most plausible flame front peak using gradient + ambient suppression + direction.

    Returns:
        int: Index of selected edge point.
    """
    gradient = -np.gradient(y)
    peaks = find_peaks_in_gradient(y, min_distance, min_height, min_width)

    y_len = len(y) - 1
    peaks = [
        peak
        for peak in peaks
        if y[max(peak - 10, 0)] >= high_val and y[min(peak + 10, y_len)] <= low_val
    ]
    if len(peaks) == 0:
        return 0

    peak_values = gradient[peaks]
    ambient_values = y[peaks]
    rv = skewnorm(3)
    mean, _, _ = skewnorm.stats(3, moments="mvs")

    if previous_peak is not None and previous_peak > 0:
        if previous_velocity > 5:
            direction_factor = rv.pdf(
                (
                    (np.array(peaks) - previous_peak + previous_velocity)
                    / previous_velocity
                )
                / 10
                + mean * (1 - 1 / 10)
            )
            direction_factor[direction_factor == 0] = 1
            return peaks[
                np.argmax(
                    peak_values
                    / ambient_values**ambient_weighting
                    * direction_factor**direction_weighting
                )
            ]

    return peaks[np.argmax(peak_values / ambient_values**ambient_weighting)]


def calculate_edge_data(data, find_edge_point, custom_filter=lambda x: x):
    """
    Calculates the edge position for each row of each frame.

    Args:
        data (np.ndarray): 3D array of shape (H, W, T).
        find_edge_point (Callable): Method to find edge in 1D data.
        custom_filter (Callable): Optional filter to apply to each frame.

    Returns:
        list[list[int]]: Edge coordinates per frame.
    """
    result = []
    # bar = progressbar.ProgressBar()
    # for n in bar(range(data.shape[-1])):
    for n in range(data.shape[-1]):
        logging.debug(f"[DEBUG] Processing frame {n + 1}/{data.shape[-1]}")
        frame = data[:, :, n]
        background_frame = data[:, :, max(n - 1, 0)]

        filtered_frame = custom_filter(frame.copy())
        frame = filtered_frame - custom_filter(background_frame)

        cv2_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, thresh = cv2.threshold(
            cv2_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh = cv2.dilate(thresh, None, iterations=10)

        frame_result = []
        for i in range(frame.shape[0]):
            start, end = 0, -1
            if n < 150:
                try:
                    start, end = np.where(thresh[i, :] > 0)[0][[0, -1]]
                    end += 10
                except IndexError:
                    pass

            y = filtered_frame[i, start:end]
            params = {}
            if len(result) > 1:
                params["previous_peak"] = result[-1][i]
                params["previous_velocity"] = result[-1][i] - result[-2][i]

            peak = find_edge_point(y, params=params)
            if peak > 0:
                peak += start
            frame_result.append(peak)
        result.append(frame_result)

    return result


def calculate_edge_results_for_exp_name(
    exp_name, left=False, dewarped_data=None, save=True
):
    """
    Run full edge detection pipeline for a given experiment name.

    Args:
        exp_name (str): Experiment identifier.
        left (bool): Whether to process left side.
        dewarped_data (np.ndarray): Optional preloaded data.
        save (bool): Whether to write result to HDF5.

    Returns:
        np.ndarray: Edge data (if save=False).
    """
    if dewarped_data is None:
        dewarped_data = get_dewarped_data(exp_name)

    if "CANON" in exp_name:
        peak_method = lambda x, params=None: right_most_point_over_threshold(
            x, threshold=125
        )
        custom_filter = lambda x: x
    elif "RCE" in exp_name:
        peak_method = lambda x, params=None: right_most_point_over_threshold(
            x, threshold=280
        )
        custom_filter = lambda x: band_filter(x, low=100, high=380)
    else:
        peak_method = lambda x, params=None: highest_peak_to_lowest_value(
            x,
            min_distance=10,
            min_height=1,
            min_width=2,
            ambient_weighting=2,
            high_val=320,
            low_val=380,
            **params,
        )
        custom_filter = lambda x: band_filter(x, low=150, high=450)

    results = calculate_edge_data(
        dewarped_data, peak_method, custom_filter=custom_filter
    )
    dst_handler.close_file()
    if not save:
        return results
    save_edge_results(exp_name, np.array(results))


# =====================================================================================
# OPTIONAL: PLOTTING / VISUALIZATION (GUI or Debug only)
# =====================================================================================


def band_filter(frame, low=None, high=None):
    """
    Clip intensity values between low and high threshold.

    Args:
        frame (np.ndarray): Input image.
        low (float): Lower clipping limit.
        high (float): Upper clipping limit.

    Returns:
        np.ndarray: Filtered image.
    """
    frame = frame.copy()
    if low is None:
        low = frame.min()
    if high is None:
        high = frame.max()
    frame[frame > high] = high
    frame[frame < low] = low
    return frame


def plot_edge(frame, find_edge_point=right_most_peak):
    """
    Plot detected edge for each line in the frame.

    Args:
        frame (np.ndarray): 2D thermal frame.
        find_edge_point (Callable): Edge detection function.
    """
    plt.imshow(frame, cmap="hot")
    for slice in range(frame.shape[0]):
        y = frame[slice, :]
        peak = find_edge_point(y)
        plt.scatter(peak, slice, c="purple")


def show_flame_spread(edge_results, y_coord):
    """
    Plot flame front x-coordinate over time at a given y-line.

    Args:
        edge_results (np.ndarray): Edge matrix.
        y_coord (int): Line index from bottom.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    y_coord = -y_coord - 1
    fig, ax = plt.subplots()
    ax.plot(edge_results.T[y_coord])
    ax.set_title("Flame spread at y = {}".format(y_coord))
    ax.set_xlabel("Frame")
    ax.set_ylabel("X coordinate")
    return fig, ax


def show_flame_contour(data, edge_results, frame):
    """
    Overlay detected edge on thermal image for a given frame.

    Args:
        data (np.ndarray): 3D image data.
        edge_results (np.ndarray): Edge matrix.
        frame (int): Frame index.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, frame], cmap="hot")
    ax.plot(edge_results[frame][::-1], range(len(edge_results[frame]) - 1, -1, -1))
    ax.set_title(f"Flame contour at frame {frame}")
    ax.invert_yaxis()
    return fig, ax


def show_flame_spread_velocity(edge_results, y_coord, rolling_window=3):
    """
    Plot local velocity of flame front at a fixed y-line.

    Args:
        edge_results (np.ndarray): Edge data.
        y_coord (int): Line index.
        rolling_window (int): Smoothing window size.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    fig, ax = plt.subplots()
    data = edge_results.T[y_coord]
    data = np.convolve(
        np.diff(medfilt(data, rolling_window)),
        np.ones(rolling_window) / rolling_window,
        mode="same",
    )
    ax.plot(data)
    ax.set_title("Flame spread velocity at y = {}".format(y_coord))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Velocity [px/frame]")
    return fig, ax


# =====================================================================================
# ARCHIVED / NOT RECOMMENDED FOR CURRENT USE
# =====================================================================================

# def plot_3D(frame):
#     """
#     Plot 3D data
#     :param frame: 3D data to plot
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     x = np.arange(0, frame.shape[1], 1)
#     y = np.arange(0, frame.shape[0], 1)
#     X, Y = np.meshgrid(x, y)
#     ax.plot_surface(X, Y, frame, cmap="hot")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     plt.show()
#
# def plot_imshow(frame):
#     """
#     Plot 2D data
#     :param frame: 2D data to plot
#     """
#     plt.imshow(frame, cmap="hot")
#     plt.show()
#
# def plot_1D(frame, slice):
#     """
#     Plot 1D data
#     :param frame: 1D data to plot
#     """
#     fig, ax = plt.subplots()
#     y = frame[slice, :]
#     x = np.arange(0, frame.shape[1], 1)
#     ax.plot(x, y)
#     ax.set_title("Temp at y = {}".format(slice))
#     ax.set_ylabel("Temperature")
#     ax.set_xlabel("X")
#
#     return fig, ax
#
# def plot_gradient(frame, slice):
#     """
#     Plot gradient of 1D data
#     :param frame: 1D data to plot
#     """
#     y = frame[slice, :]
#     x = np.arange(0, frame.shape[1], 1)
#     gradient = np.gradient(y)
#     fig, ax = plt.subplots()
#     ax.plot(x, gradient)
#     ax.set_title("Gradient at y = {}".format(slice))
#     ax.set_xlabel("X")
#     ax.set_ylabel("Gradient")
#     return fig, ax
#
# def get_frame(data, frame_number):
#     return data[:, :, frame_number]
#
# def show_frame(data, frame_number):
#     fig, ax = plt.subplots()
#     ax.imshow(get_frame(data, frame_number), cmap="hot")
#     ax.set_title(f"Frame {frame_number}")
#     # ax.invert_yaxis()
#     return fig, ax
#
# def show_flame_spread_plotly(edge_results, y_coord):
#     y_coord = -y_coord - 1
#     fig = px.line(x=range(len(edge_results)), y=edge_results.T[y_coord])
#     fig.update_layout(
#         title="Flame spread at y = {}".format(y_coord),
#         xaxis_title="Frame",
#         yaxis_title="X coordinate",
#     )
#     return fig
#
# def show_flame_contour_plotly(data, edge_results, frame):
#     fig = go.Figure()
#     fig.add_trace(go.Heatmap(z=data[:, :, frame], colorscale="hot", showscale=False))
#     fig.add_trace(
#         go.Scatter(
#             x=edge_results[frame][::-1],
#             y=list(range(len(edge_results[frame]) - 1, -1, -1)),
#             mode="lines",
#         )
#     )
#     fig.update_layout(
#         title=f"Flame contour at frame {frame}", yaxis=dict(autorange="reversed")
#     )
#     return fig

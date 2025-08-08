from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

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


class EdgeFn(Protocol):
    def __call__(self, y: np.ndarray, params: Optional[Dict] = ...) -> int: ...


def find_peaks_in_gradient(
    y: np.ndarray, min_distance: int = 10, min_height: float = 2, min_width: int = 2
) -> np.ndarray:
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

    # Scipy-Stubs wollen hier eine Sequence/Array – Liste mit float passt:
    g = -np.gradient(y)
    kwargs: dict[str, Any] = {}
    if min_height is not None:
        kwargs["height"] = float(min_height)
    if min_distance is not None:
        kwargs["distance"] = int(min_distance)
    if min_width is not None:
        kwargs["width"] = float(min_width)
    peaks, _ = find_peaks(g, **kwargs)
    return peaks


def right_most_point_over_threshold(
    y: np.ndarray, threshold: float = 0, params: Optional[Dict] = None
) -> int:
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


def left_most_point_over_threshold(
    y: np.ndarray, threshold: float = 0, params: Optional[dict] = None
) -> int:
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


def right_most_peak(
    y: np.ndarray, min_distance: int = 10, min_height: float = 2, min_width: int = 2
) -> int:
    """
    Return the right-most peak in the gradient of the signal.

    Returns:
        int: Index of last detected peak, or 0.
    """
    peaks = find_peaks_in_gradient(y, min_distance, min_height, min_width)
    return peaks[-1] if len(peaks) else 0


def highest_peak(
    y: np.ndarray, min_distance: int = 10, min_height: float = 2, min_width: int = 2
) -> int:
    """
    Return the index of the peak with the highest gradient.

    Returns:
        int: Index of the highest peak, or 0.
    """
    gradient = -np.gradient(y)
    peaks = find_peaks_in_gradient(y, min_distance, min_height, min_width)
    return peaks[np.argmax(gradient[peaks])] if len(peaks) else 0


def highest_peak_to_lowest_value(
    y: np.ndarray,
    min_distance: int = 10,
    min_height: float = 2,
    min_width: int = 2,
    ambient_weighting: float = 2,
    high_val: float = 0,
    low_val: float = 1e10,
    direction_weighting: float = 0.0,
    previous_peak: Optional[int] = None,
    previous_velocity: float = 0,
) -> int:
    """
    Find the most plausible flame front peak using gradient + ambient suppression + direction.

    Returns:
        int: Index of selected edge point.
    """
    gradient = -np.gradient(y)
    peaks = find_peaks_in_gradient(y, min_distance, min_height, min_width)

    y_len = len(y) - 1
    # ❗ nicht zu einer list casten (mypy meckert über Typwechsel).
    # Danach wieder ndarray machen:
    peaks = np.array(
        [
            int(peak)
            for peak in peaks
            if y[max(peak - 10, 0)] >= high_val and y[min(peak + 10, y_len)] <= low_val
        ],
        dtype=int,
    )
    if peaks.size == 0:
        return 0

    peak_values = gradient[peaks]
    ambient_values = y[peaks]
    rv = skewnorm(3)
    mean, _, _ = rv.stats(moments="mvs")

    if previous_peak is not None and previous_peak > 0 and previous_velocity > 5:
        direction_factor = rv.pdf(
            ((peaks - previous_peak + previous_velocity) / previous_velocity) / 10
            + mean * (1 - 1 / 10)
        )
        direction_factor[direction_factor == 0] = 1
        return int(
            peaks[
                np.argmax(
                    peak_values
                    / (ambient_values**ambient_weighting)
                    * (direction_factor**direction_weighting)
                )
            ]
        )

    return int(peaks[np.argmax(peak_values / (ambient_values**ambient_weighting))])


def calculate_edge_data(
    data: np.ndarray,
    find_edge_point: EdgeFn,
    custom_filter: Callable[[np.ndarray], np.ndarray] = lambda x: x,
) -> list[list[int]]:
    """
    Calculates the edge position for each row of each frame.

    Args:
        data (np.ndarray): 3D array of shape (H, W, T).
        find_edge_point (Callable): Method to find edge in 1D data.
        custom_filter (Callable): Optional filter to apply to each frame.

    Returns:
        list[list[int]]: Edge coordinates per frame.
    """
    result: list[list[int]] = []
    # bar = progressbar.ProgressBar()
    # for n in bar(range(data.shape[-1])):
    for n in range(data.shape[-1]):
        logging.debug(f"[DEBUG] Processing frame {n + 1}/{data.shape[-1]}")
        frame = data[:, :, n].astype(np.float32)  # float32 behalten
        background_frame = data[:, :, max(n - 1, 0)].astype(np.float32)

        filtered_frame = custom_filter(frame.copy())
        frame = filtered_frame - custom_filter(background_frame)

        # Nur für Binärmaske → 0..1 float → u8
        minv = float(filtered_frame.min())
        maxv = float(filtered_frame.max())
        if maxv > minv:
            normed = (filtered_frame - minv) / (maxv - minv)
        else:
            normed = np.zeros_like(filtered_frame, dtype=np.float32)

        mask_u8 = (normed * 255.0).astype(np.uint8)

        # Otsu + Morphologie auf u8
        _, thresh = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), dtype=np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=10)

        frame_result: list[int] = []
        for i in range(frame.shape[0]):
            start, end = 0, -1
            if n < 150:
                idx = np.where(thresh[i, :] > 0)[0]
                if idx.size:
                    start, end = int(idx[0]), int(idx[-1]) + 10

            y = filtered_frame[i, start:end]
            params = {}
            if len(result) > 1:
                params["previous_peak"] = result[-1][i]
                params["previous_velocity"] = result[-1][i] - result[-2][i]

            # beide Signaturen erlauben
            peak = find_edge_point(y, params=params)

            if peak > 0:
                peak += start
            frame_result.append(int(peak))
        result.append(frame_result)

    return result


def calculate_edge_results_for_exp_name(
    exp_name: str,
    left: bool = False,
    dewarped_data: Optional[np.ndarray] = None,
    save: bool = True,
) -> Optional[np.ndarray]:
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
        return np.array(results)
    save_edge_results(exp_name, np.array(results))
    return None


# =====================================================================================
# OPTIONAL: PLOTTING / VISUALIZATION (GUI or Debug only)
# =====================================================================================


def band_filter(
    frame: np.ndarray, low: Optional[float] = None, high: Optional[float] = None
) -> np.ndarray:
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


def plot_edge(
    frame: np.ndarray, find_edge_point: Callable[[np.ndarray], int] = right_most_peak
) -> None:
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


def show_flame_spread(
    edge_results: np.ndarray, y_coord: int
) -> tuple[plt.Figure, plt.Axes]:
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


def show_flame_contour(
    data: np.ndarray, edge_results: np.ndarray, frame: int
) -> tuple[plt.Figure, plt.Axes]:
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


def show_flame_spread_velocity(
    edge_results: np.ndarray, y_coord: int, rolling_window: int = 3
) -> tuple[plt.Figure, plt.Axes]:
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

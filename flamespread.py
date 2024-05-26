import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import scipy
from scipy.stats import skewnorm

import user_config
import dataset_handler as dst_handler
from dataset_handler import get_dewarped_data, get_edge_results, save_edge_results
import plotly.express as px
from plotly import graph_objects as go


def plot_3D(frame):
    """
    Plot 3D data
    :param frame: 3D data to plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, frame.shape[1], 1)
    y = np.arange(0, frame.shape[0], 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, frame, cmap='hot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def find_peaks(y, min_distance, min_height, min_width):
    """
    Find peaks in 1D data
    :param x: x-axis of the data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, height=min_height, distance=min_distance, width=min_width)
    return peaks


def plot_imshow(frame):
    """
    Plot 2D data
    :param frame: 2D data to plot
    """
    plt.imshow(frame, cmap='hot')
    plt.show()


def plot_1D(frame, slice):
    """
    Plot 1D data
    :param frame: 1D data to plot
    """
    fig, ax = plt.subplots()
    y = frame[slice, :]
    x = np.arange(0, frame.shape[1], 1)
    ax.plot(x, y)
    ax.set_title('Temp at y = {}'.format(slice))
    ax.set_ylabel("Temperature")
    ax.set_xlabel("X")

    return fig, ax


def plot_gradient(frame, slice):
    """
    Plot gradient of 1D data
    :param frame: 1D data to plot
    """
    y = frame[slice, :]
    x = np.arange(0, frame.shape[1], 1)
    gradient = np.gradient(y)
    fig, ax = plt.subplots()
    ax.plot(x, gradient)
    ax.set_title('Gradient at y = {}'.format(slice))
    ax.set_xlabel('X')
    ax.set_ylabel('Gradient')
    return fig, ax


def right_most_peak(y, min_distance=10, min_height=2, min_width=2):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    """
    gradient = -np.gradient(y)
    peaks = find_peaks(gradient, min_distance=min_distance, min_height=min_height, min_width=min_width)
    if len(peaks) == 0:
        return 0
    return peaks[-1]


def highest_peak(y, min_distance=10, min_height=2, min_width=2):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    """
    gradient = -np.gradient(y)
    peaks = find_peaks(gradient, min_distance=min_distance, min_height=min_height, min_width=min_width)
    if len(peaks) == 0:
        return 0
    return peaks[np.argmax(gradient[peaks])]


rv = skewnorm(3)
mean, var, skew = skewnorm.stats(3, moments='mvs')
def highest_peak_to_lowest_value(y, min_distance=10, min_height=2, min_width=2, ambient_weighting=2, high_val=0,
                                 low_val=10e10, direction_weighting=0.0, previous_peak=None,previous_velocity=0):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    :param ambient_weighting: weight of the ambient value
    :param high_val: high value of the ambient value
    :param low_val: low value of the ambient value
    :param direction_weighting: weight of the direction
    :param previous_peak: previous peak

    """
    gradient = -np.gradient(y)
    peaks = find_peaks(gradient, min_distance=min_distance, min_height=min_height, min_width=min_width)

    y_len = len(y) - 1
    peaks = [peak for peak in peaks if y[max(peak - 10, 0)] >= high_val and y[min(peak + 10, y_len)] <= low_val]
    if len(peaks) == 0:
        return 0
    peak_values = gradient[peaks]
    ambient_values = y[peaks]
    if previous_peak is not None and previous_peak > 0:
        # direction_factor = (1 + ((np.array(peaks) - previous_peak +previous_velocity) / len(y))) ** direction_weighting
        if previous_velocity >5:
            direction_factor = (rv.pdf(((np.array(peaks)-previous_peak+previous_velocity)/previous_velocity)/10 + mean*(1-1/10)))
            if all(direction_factor == 0):
                direction_factor = 1
            return peaks[np.argmax(peak_values / ambient_values ** ambient_weighting * direction_factor**direction_weighting)]

    return peaks[np.argmax(peak_values / ambient_values ** ambient_weighting)]


def plot_edge(frame, find_edge_point=right_most_peak):
    plt.imshow(frame, cmap='hot')
    for slice in range(frame.shape[0]):
        y = frame[slice, :]
        peak = find_edge_point(y)
        plt.scatter(peak, slice, c='purple')


def calculate_edge_data(data, find_edge_point, custom_filter=lambda x: x):
    # data = data[:,::-1]
    result = []
    bar = progressbar.ProgressBar()
    for n in bar(range(data.shape[-1])):
        background_frame = data[:, :, max(n - 1, 0)]
        frame = data[:, :, n]
        frame_result = []
        filtered_frame = custom_filter(frame.copy())
        frame = filtered_frame - custom_filter(background_frame)
        cv2_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ret, thresh = cv2.threshold(cv2_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.dilate(thresh, None, iterations=10)
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
                params['previous_peak'] = result[-1][i]
                params['previous_velocity'] = result[-1][i] - result[-2][i]
            peak = find_edge_point(y, params =params)
            if peak > 0:
                peak += start
            frame_result.append(peak)
        result.append(frame_result)
    return result


def show_flame_spread(edge_results, y_coord):
    y_coord = -y_coord - 1
    fig, ax = plt.subplots()
    ax.plot(edge_results.T[y_coord])
    ax.set_title('Flame spread at y = {}'.format(y_coord))
    ax.set_xlabel('Frame')
    ax.set_ylabel('X coordinate')
    return fig, ax


def show_flame_spread_plotly(edge_results, y_coord):
    y_coord = -y_coord - 1
    fig = px.line(x=range(len(edge_results)), y=edge_results.T[y_coord])
    fig.update_layout(title='Flame spread at y = {}'.format(y_coord),
                      xaxis_title='Frame',
                      yaxis_title='X coordinate')
    return fig


def get_frame(data, frame_number):
    return data[:, :, frame_number]


def show_frame(data, frame_number):
    fig, ax = plt.subplots()
    ax.imshow(get_frame(data, frame_number), cmap='hot')
    ax.set_title(f'Frame {frame_number}')
    # ax.invert_yaxis()
    return fig, ax


def band_filter(frame, low=None, high=None):
    frame = frame.copy()
    if low is None:
        low = frame.min()
    if high is None:
        high = frame.max()
    frame[frame > high] = high
    frame[frame < low] = low
    return frame


def show_flame_contour(data, edge_results, frame):
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, frame], cmap='hot')

    ax.plot(edge_results[frame][::-1], range(len(edge_results[frame]) - 1, -1, -1), )
    ax.set_title(f'Flame contour at frame {frame}')
    ax.invert_yaxis()
    # ax.set_xlabel('X coordinate')
    # ax.set_ylabel('Y coordinate')
    # ax.set_ylim(ax.get_ylim()[::-1])
    return fig, ax


def show_flame_spread_velocity(edge_results, y_coord, rolling_window=3):
    fig, ax = plt.subplots()
    data = edge_results.T[y_coord]
    data = scipy.signal.medfilt(data, rolling_window)
    data = np.diff(data)
    # rolling average
    data = np.convolve(data, np.ones(rolling_window) / rolling_window, mode='same')
    ax.plot(data)
    ax.set_title('Flame spread velocity at y = {}'.format(y_coord))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity [px/frame]')
    return fig, ax


def show_flame_contour_plotly(data, edge_results, frame):
    fig = go.Figure()
    fig.add_trace(go.Heatmap
                  (z=data[:, :, frame],
                   colorscale='hot',
                   showscale=False))
    fig.add_trace(
        go.Scatter(x=edge_results[frame][::-1], y=list(range(len(edge_results[frame]) - 1, -1, -1)), mode='lines'))
    fig.update_layout(title=f'Flame contour at frame {frame}',
                      yaxis=dict(autorange='reversed'))
    return fig


def right_most_point_over_threshold(y, threshold=0):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param threshold: threshold value
    """
    peaks = np.where(y > threshold)[-1]
    if len(peaks) == 0:
        return 0
    return peaks[-1]


def calculate_edge_results_for_exp_name(exp_name):
    print(f'Loading {exp_name}')
    dewarped_data = get_dewarped_data(exp_name)
    print('Finding edge')
    if 'CANON' in exp_name:
        peak_method = lambda x, params=None: right_most_point_over_threshold(x, threshold=125)
        custom_filter = lambda x: x
    else:
        peak_method = lambda x, params=None: highest_peak_to_lowest_value(x, min_distance=10, min_height=1, min_width=2,
                                                                      ambient_weighting=2, high_val=320, low_val=380,
                                                                      **params)
        custom_filter = lambda x: band_filter(x, low=150, high=450)
    # peak_method = highest_peak
    results = calculate_edge_data(dewarped_data, peak_method, custom_filter=custom_filter)
    dst_handler.close_file()
    save_edge_results(exp_name, np.array(results))


if __name__ == '__main__':
    # for exp_name in ['lfs_pmma_DE_6mm_tc_R3_0001',]:
    #     print(f'Loading {exp_name}')
    #     dewarped_data = get_dewarped_data(exp_name)
    #     print('Finding edge')
    #     peak_method = lambda x,params = None: highest_peak_to_lowest_value(x, min_distance=10, min_height=1, min_width=2,
    #                                                          ambient_weighting=2, high_val=320, low_val=380, **params)
    #     # peak_method = highest_peak
    #     results = calculate_edge_data(dewarped_data, peak_method, custom_filter=lambda x: band_filter(x, low=150, high=450))
    #     dst_handler.close_file()
    #     save_edge_results(exp_name, np.array(results))


    #Canon
    exp_name = 'lfs_pmma_DE_6mm_tc_R2_CANON'
    peak_method = lambda x,params={}: right_most_point_over_threshold(x, threshold=75)
    dewarped_data = get_dewarped_data(exp_name)
    results = calculate_edge_data(dewarped_data, peak_method)
    dst_handler.close_file()
    save_edge_results(exp_name, np.array(results))

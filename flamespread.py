import os

import numpy as np
import matplotlib.pyplot as plt
import progressbar

import user_config
import dataset_handler as dst_handler
from dataset_handler import get_dewarped_data,get_edge_results,save_edge_results



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


def find_peaks( y, min_distance, min_height,min_width):
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

def plot_1D(frame,slice):
    """
    Plot 1D data
    :param frame: 1D data to plot
    """
    y = frame[slice,:]
    x = np.arange(0,frame.shape[1],1)
    plt.plot(x,y)
    plt.show()

def plot_gradient(frame,slice):
    """
    Plot gradient of 1D data
    :param frame: 1D data to plot
    """
    y = frame[slice,:]
    x = np.arange(0,frame.shape[1],1)
    gradient = np.gradient(y)
    fig, ax = plt.subplots()
    ax.plot(x,gradient)
    ax.set_title('Gradient of slice {}'.format(slice))
    ax.set_xlabel('X')
    ax.set_ylabel('Gradient')
    return fig, ax



def right_most_peak(y,  min_distance=10, min_height=2,min_width=2):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    """
    gradient = -np.gradient(y)
    peaks = find_peaks(gradient,min_distance=min_distance,min_height=min_height,min_width=min_width)
    if len(peaks) == 0:
        return 0
    return peaks[-1]

def highest_peak(y,  min_distance=10, min_height=2,min_width=2):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    """
    gradient = -np.gradient(y)
    peaks = find_peaks(gradient,min_distance=min_distance,min_height=min_height,min_width=min_width)
    if len(peaks) == 0:
        return 0
    return peaks[np.argmax(gradient[peaks])]

def highest_peak_to_lowest_value(y,  min_distance=10, min_height=2,min_width=2,ambient_weighting=2):
    """
    Find edge point in 1D data
    :param y: y-axis of the data
    :param minwidth: minimum width of the peaks
    :param min_height: minimum height of the peaks
    :param ambient_weighting: weight of the ambient value
    """
    gradient = -np.gradient(y)
    peaks = find_peaks(gradient,min_distance=min_distance,min_height=min_height,min_width=min_width)
    if len(peaks) == 0:
        return 0
    peak_values = gradient[peaks]
    ambient_values = y[peaks]
    return peaks[np.argmax(peak_values/ambient_values**ambient_weighting)]


def plot_edge(frame,find_edge_point = right_most_peak):
    plt.imshow(frame,cmap='hot')
    for slice in range(frame.shape[0]):
        y = frame[slice,:]
        peak = find_edge_point(y)
        plt.scatter(peak,slice,c='purple')

def calculate_edge_data(data, find_edge_point,filter=None):
    # data = data[:,::-1]
    result = []
    bar = progressbar.ProgressBar()
    for n in bar(range(data.shape[-1])):
        frame = data[:,:,n]
        frame_result = []
        if filter is not None:
            frame = filter(frame)
        for i in range(frame.shape[0]):
            y = frame[i,:]
            peak = find_edge_point(y)
            frame_result.append(peak)
        result.append(frame_result)
    return result


def show_flame_spread(edge_results, y_coord):
    y_coord =-y_coord-1
    fig, ax = plt.subplots()
    ax.plot(edge_results.T[y_coord])
    ax.set_title('Flame spread at y = {}'.format(y_coord))
    ax.set_xlabel('Frame')
    ax.set_ylabel('X coordinate')
    return fig, ax


def get_frame(data, frame_number):
    return data[:, :, frame_number]

def show_frame(data, frame_number):
    fig, ax = plt.subplots()
    ax.imshow(get_frame(data,frame_number), cmap='hot')
    ax.set_title(f'Frame {frame_number}')
    # ax.invert_yaxis()
    return fig, ax


def band_filter(frame,low=None,high=None):
    if low is None:
        low = frame.min()
    if high is None:
        high = frame.max()
    frame[frame>high] = high
    frame[frame<low] = low
    return frame

def show_flame_contour(data, edge_results, frame):
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, frame], cmap='hot')

    ax.plot(edge_results[frame][::-1], range(len(edge_results[frame])-1, -1, -1),)
    ax.set_title(f'Flame contour at frame {frame}')
    ax.invert_yaxis()
    # ax.set_ylim(ax.get_ylim()[::-1])
    return fig, ax




if __name__ == '__main__':
    exp_name = 'lfs_pmma_DE_6mm_tc_R1_0001'
    print(f'Loading {exp_name}')
    dewarped_data = get_dewarped_data(exp_name)
    print('Finding edge')
    peak_method = lambda x: highest_peak_to_lowest_value(x,min_distance=10,min_height=2,min_width=2,ambient_weighting=1)
    # peak_method = highest_peak
    results = calculate_edge_data(dewarped_data, peak_method,filter=lambda x: band_filter(x,low=250,high=450))
    dst_handler.close_file()
    save_edge_results(exp_name, np.array(results))



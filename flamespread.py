import os

import numpy as np
import matplotlib.pyplot as plt
import progressbar

import user_config


def read_dewarped_data(filename: str) -> np.ndarray:
    """
    Read the dewarped data from the file. The data is expected to be in the [Data] section of the file, separated by ';'
    :param filename: filepath to the dewarped data file
    :return: dewarped data as numpy array
    """
    data = np.load(filename)
    return data[::-1]

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
    plt.plot(x,gradient)
    plt.axhline(0, color='black', lw=0.5)



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
    plt.show()

def write_out_edge_results(data,find_edge_point):
    # data = data[:,::-1]
    result = []
    bar = progressbar.ProgressBar()
    for n in bar(range(data.shape[-1])):
        frame = data[:,:,n]
        frame_result = []
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
    ax.legend()
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


def get_edge_results(name):
    edge_results_folder = user_config.get_path('edge_results_folder')
    edge_results = np.load(f'{edge_results_folder}/{name}_edge_results.npy')
    return edge_results

def get_dewarped_data(name):
    dewarped_data_folder = user_config.get_path('dewarped_data_folder')
    dewarped_data = np.load(f'{dewarped_data_folder}/{name}_dewarped.npy')
    return dewarped_data

def load_data(filename):
    data = np.load(filename)
    return data
def show_flame_contour(data, edge_results, frame):
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, frame], cmap='hot')

    ax.plot(edge_results[frame][::-1], range(len(edge_results[frame])-1, -1, -1),)
    ax.set_title(f'Flame contour at frame {frame}')
    ax.invert_yaxis()
    # ax.set_ylim(ax.get_ylim()[::-1])
    return fig, ax




if __name__ == '__main__':
    filename = 'lfs_pmma_DE_6mm_tc_R2_0001_dewarped.npy'
    dewarped_data_folder =user_config.get_path('dewarped_data_folder')
    edge_results_folder = user_config.get_path('edge_results_folder')
    print(f'Loading {filename}')
    data = load_data(f'{dewarped_data_folder}/{filename}')
    print('Finding edge')
    peak_method = lambda x: highest_peak_to_lowest_value(x,min_distance=10,min_height=2,min_width=2,ambient_weighting=1)
    results = write_out_edge_results(data,highest_peak_to_lowest_value)
    np.save(f'{edge_results_folder}/{filename.replace("dewarped","edge_results_highest")}',np.array(results))
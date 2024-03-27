import numpy as np
import matplotlib.pyplot as plt
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
    plt.show()


def find_edge_point(y, min_distance=10, min_height=2,min_width=5):
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
def plot_edge(frame):
    plt.imshow(frame,cmap='hot')
    for slice in range(frame.shape[0]):
        y = frame[slice,:]
        peak = find_edge_point(y)
        plt.scatter(peak,slice,c='purple')
    plt.show()


if __name__ == '__main__':
    data = read_dewarped_data('dewarped_data/Messung_01_0001_dewarped.npy')
    slice = 90
    result = []
    for i in range(data.shape[-1]):
        y = data[:,slice,i]
        peak = find_edge_point(y)
        result.append(peak)
    plt.plot(result)
    plt.show()
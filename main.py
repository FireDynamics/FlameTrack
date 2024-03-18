import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from IR_analysis import *

#Kommentar!!!

def select_points_mpl(data, cmap='hot') -> np.ndarray:
    """
    Select points from the data using the mouse. Press 'q' to quit and return the selected points
    :param data: data to select points from
    :param cmap: colormap to use for the plot
    :return: points selected by the user
    """
    points = deque(maxlen=4)

    def on_click(event):
        if event.button is MouseButton.LEFT:
            points.append([event.xdata, event.ydata])
        lines = plt.gca().lines
        if len(lines) > 0:
            lines[-1].remove()
        x, y = list(zip(*points))
        x = list(x)
        y = list(y)
        x += [x[0]]
        y += [y[0]]
        plt.plot(x, y, linestyle='--', marker='x', color='red')
        plt.draw()

    def on_key_press(event):
        if event.key == 'q':
            print('Quitting')
            plt.close()

    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.connect('button_press_event', on_click)
    plt.connect('key_press_event', on_key_press)
    plt.show()
    return sort_corner_points(points)



if __name__ == '__main__':
    #Parameters
    datapath = 'data/Messung_01_0001_0001.csv'
    target_pixels_width = 100
    target_pixels_height = 100


    data = read_IR_data(datapath)
    points = select_points_mpl(data)
    dewarped_data = dewarp_data(data, points, target_pixels_width, target_pixels_height)
    plt.figure()
    plt.imshow(dewarped_data, cmap='hot')
    plt.show()
    # For saving the dewarped data
    # np.savetxt('dewarped_data.csv', dewarped_data, delimiter=';')
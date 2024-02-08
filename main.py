from IR_analysis import *


if __name__ == '__main__':
    data = read_IR_data('data/Messung_01_0001_0797.csv')
    print(data)
    points = select_points_mpl(data)
    print(points)
    dewarped_data = dewarp_data(data, points, 100, 100)
    plt.figure()
    plt.imshow(dewarped_data, cmap='hot')
    plt.show()
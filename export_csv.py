import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import flamespread as fs
import dataset_handler as dh
import user_config


#reload


def get_edge_results(exp_name):
    edge_results = dh.get_edge_results(exp_name)[:]
    dh.close_file()
    return edge_results


def export_csv(exp_name,height_percentage):
    edge_results = get_edge_results(exp_name)
    width, height = dh.get_dewarped_data(exp_name).shape[1:]
    y_slice = int(height_percentage * len(edge_results.T))
    dh.close_file()
    width_factor = target_width/width
    height_factor = target_height/len(edge_results.T)
    edge_results =edge_results * width_factor
    data = edge_results.T[y_slice]
    y_height = y_slice * height_factor
    np.savetxt(f'{exp_name}_flamespread_{y_height}mm.csv',data,delimiter=',')




if __name__ == '__main__':
    for exp_name in os.listdir(os.path.join(user_config.get_path('data_prefix_path'),'saved_data_old')):
        target_width = 773
        target_height = 133

        for hp in [0.25,0.5,0.75]:
            export_csv(exp_name.replace('.h5',''),hp)


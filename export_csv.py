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


def export_csv(exp_name,height_percentage,start_frame = 0):
    edge_results = get_edge_results(exp_name)
    width, height = dh.get_dewarped_data(exp_name).shape[1:]
    y_slice = int(height_percentage * len(edge_results.T))
    dh.close_file()
    max_edge = int(np.median(np.argmax(edge_results.T,axis=1))-10)
    width_factor = target_width/width
    height_factor = target_height/len(edge_results.T)
    edge_results =edge_results * width_factor

    #convert max edge to frame
    end_frame = max_edge % edge_results.shape[0]
    data = edge_results.T[y_slice,start_frame:end_frame]
    y_height = y_slice * height_factor
    np.savetxt(os.path.join(user_config.get_path('csv_folder'),f'{exp_name}_flamespread_{y_height:.2f}mm.csv'),data,delimiter=',')




if __name__ == '__main__':
    for exp_name,start in zip(['R1','R2'],[,364]):
        target_width = 773
        target_height = 133
        for hp in [0.25,0.5,0.75]:
        #     export_csv(f'lfs_PMMA_DE_6mm_tc_{exp_name}_IR',hp,start_frame=start)

            export_csv(f'PMMA_DE_6mm_RCE_1m_{exp_name}_IR',hp,start_frame=start)





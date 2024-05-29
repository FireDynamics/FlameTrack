import numpy as np
import matplotlib.pyplot as plt
import scipy
import flamespread as fs
import dataset_handler as dh
#reload

exp_name = 'lfs_pmma_DE_6mm_tc_R1_0001'
target_width = 773
target_height = 133


def get_data(exp_name):
    dewarped_data= dh.get_dewarped_data(exp_name)[:]
    edge_results = dh.get_edge_results(exp_name)[:]
    dh.close_file()
    return dewarped_data, edge_results

def get_frame(exp_name, frame):
    dewarped_data_frame= dh.get_dewarped_data(exp_name)[:,:,frame]
    edge_results_frame = dh.get_edge_results(exp_name)[frame]
    dh.close_file()
    return dewarped_data_frame, edge_results_frame

def get_edge_results(exp_name):
    edge_results = dh.get_edge_results(exp_name)[:]
    dh.close_file()
    return edge_results


def get_flame_spread(exp_name,y_slice):
    edge_results = get_edge_results(exp_name)
    dh.close_file()
    width_factor = target_width/width
    height_factor = target_height/height
    edge_results =edge_results * width_factor
    data = edge_results.T[y_slice]




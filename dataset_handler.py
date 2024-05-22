import h5py
import numpy as np
import os
import user_config


HDF_FILE = None
LOADED_EXP_NAME = None

def create_h5_file(exp_name):
    global HDF_FILE
    global LOADED_EXP_NAME
    filename = os.path.join(user_config.get_path('saved_data'), exp_name + '.h5')
    f = h5py.File(filename, 'w')
    f.create_group('dewarped_data')
    f.create_group('edge_results')
    HDF_FILE = f
    LOADED_EXP_NAME = filename
    return f



def get_data(exp_name, group_name):
    f = get_file(exp_name)
    data = f[group_name]['data']
    return data


def get_edge_results(exp_name):
    return get_data(exp_name, 'edge_results')[:]

def get_dewarped_data(exp_name):
    return get_data(exp_name, 'dewarped_data')

def get_dewarped_metadata(exp_name):
    f = get_file(exp_name)
    return f['dewarped_data'].attrs
def get_file(exp_name,mode='r'):
    if mode =='w':
        raise ValueError('Use create_h5_file to create a new file')
    global HDF_FILE
    global LOADED_EXP_NAME

    if HDF_FILE is None:
        filename = os.path.join(user_config.get_path('saved_data'), exp_name + '.h5')
        HDF_FILE = h5py.File(filename, mode)
        LOADED_EXP_NAME = filename

    if LOADED_EXP_NAME != os.path.join(user_config.get_path('saved_data'), exp_name + '.h5'):
        close_file()
        return get_file(exp_name,mode)


    return HDF_FILE

def close_file():
    global HDF_FILE
    global LOADED_EXP_NAME
    if HDF_FILE is not None:
        HDF_FILE.close()
        HDF_FILE = None
        LOADED_EXP_NAME = None

def save_edge_results(exp_name,edge_results):
    with get_file(exp_name,'a') as f:
        grp = f['edge_results']
        if 'data' in grp:
            del grp['data']
        f['edge_results'].create_dataset('data', data=edge_results)

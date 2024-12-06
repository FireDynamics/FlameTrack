import h5py
import os
import ir_reader.analysis.user_config


HDF_FILE = None
LOADED_EXP_NAME = None

def create_h5_file(exp_name=None,filename=None):
    global HDF_FILE
    global LOADED_EXP_NAME
    if filename is None:
        filename = get_h5_file_path(exp_name)
    foldername = os.path.dirname(filename)
    if not os.path.exists(foldername):
        os.mkdir(os.path.dirname(filename))
    f = h5py.File(filename, 'w')
    f.create_group('dewarped_data')
    f.create_group('edge_results')
    HDF_FILE = f
    LOADED_EXP_NAME = filename
    return f

def get_h5_file_path(exp_name,left=False):
    left_str = '_left' if left else ''
    return os.path.join(user_config.get_path(exp_name, 'processed_data'), exp_name + '_results'+ left_str+'.h5')

def get_data(exp_name, group_name,left=False):
    f = get_file(exp_name,left=left)
    data = f[group_name]['data']
    return data


def get_edge_results(exp_name, left=False):
    return get_data(exp_name, 'edge_results',left)[:]

def get_dewarped_data(exp_name, left=False):
    return get_data(exp_name, 'dewarped_data',left)

def get_dewarped_metadata(exp_name, left=False):
    f = get_file(exp_name,left=left)
    return f['dewarped_data'].attrs
def get_file(exp_name,mode='r',left=False):
    if mode =='w':
        raise ValueError('Use create_h5_file to create a new file')
    global HDF_FILE
    global LOADED_EXP_NAME

    if HDF_FILE is None:
        filename =get_h5_file_path(exp_name,left=left)
        HDF_FILE = h5py.File(filename, mode)
        LOADED_EXP_NAME = filename

    if LOADED_EXP_NAME != get_h5_file_path(exp_name,left=left):
        close_file()
        return get_file(exp_name,mode,left=left)


    return HDF_FILE

def close_file():
    global HDF_FILE
    global LOADED_EXP_NAME
    if HDF_FILE is not None:
        HDF_FILE.close()
        HDF_FILE = None
        LOADED_EXP_NAME = None

def save_edge_results(exp_name,edge_results,left=False):
    with get_file(exp_name,'a',left=left) as f:
        grp = f['edge_results']
        if 'data' in grp:
            del grp['data']
        f['edge_results'].create_dataset('data', data=edge_results)
    close_file()



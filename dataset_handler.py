import h5py
import numpy as np
import os


def create_h5_file(filename):
    f = h5py.File(filename, 'w')
    f.create_group('dewarped_data')
    f.create_group('edge_results')
    return f



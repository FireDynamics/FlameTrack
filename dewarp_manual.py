import os

import numpy as np

import dataset_handler
import user_config
from IR_analysis import dewarp_data,dewarp_exp
from DataTypes import IrData, ImageData
import progressbar
from flamespread import calculate_edge_results_for_exp_name




# print(IrData(os.path.join(user_config.get_path('data_folder'), 'lfs_pmma_DE_6mm_tc_R4_0001')).data_numbers)
# dewarp_exp('lfs_pmma_DE_6mm_tc_R4_0001',IrData('lfs_pmma_DE_6mm_tc_R4_0001'))
if __name__ == '__main__':
    exp_name ='PMMA_DE_6mm_RCE_1m_R1'
    dewarp_exp(exp_name,data =IrData(user_config.get_IR_path(exp_name)),frequency=1,renew=False)
    calculate_edge_results_for_exp_name(exp_name,left=False)
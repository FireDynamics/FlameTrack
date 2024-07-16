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
    # Change which experiments to dewarp here
    exp_names = [
        #IR
        'lfs_pmma_DE_6mm_tc_R1_IR',
        'lfs_pmma_DE_6mm_tc_R2_IR',
        'lfs_pmma_DE_6mm_tc_R3_IR',
        'lfs_pmma_DE_6mm_tc_R4_IR',
        #Canon
        # 'lfs_pmma_DE_6mm_tc_R1_CANON',
        # 'lfs_pmma_DE_6mm_tc_R2_CANON',
        # 'lfs_pmma_DE_6mm_tc_R3_CANON',
        # 'lfs_pmma_DE_6mm_tc_R4_CANON'
    ]

    # change to testing mode, testing mode only dewarps 20 frames in the middle of the dataset
    # testing = True
    #
    # for exp_name in exp_names:
    #     if 'CANON' in exp_name:
    #         data = ImageData(os.path.join(user_config.get_path('canon_folder'), exp_name.replace('_CANON', "")), 'JPG')
    #     else:
    #         data = IrData(os.path.join(user_config.get_path('data_folder'), exp_name.replace('IR', "0001")))
    #     dewarp_exp(exp_name, data,testing)
    #     calculate_edge_results_for_exp_name(exp_name)
    exp_name ='PMMA_DE_6mm_RCE_1m_R3_IR'
    dewarp_exp(exp_name,data =IrData(os.path.join(user_config.get_path('data_folder'), exp_name.replace('IR', "0001"))),frequency=1,renew=True)
    calculate_edge_results_for_exp_name(exp_name,mirror=True)
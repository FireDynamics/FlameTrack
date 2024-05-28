import os

import dataset_handler
import user_config
from IR_analysis import dewarp_data
from DataTypes import IrData, ImageData
import progressbar
from flamespread import calculate_edge_results_for_exp_name
def dewarp_exp(exp_name,data):
    dset = dataset_handler.get_file(exp_name,'a')['dewarped_data']['data']
    metadata = dataset_handler.get_dewarped_metadata(exp_name)
    data_numbers = data.data_numbers
    dewarp_params ={}
    # grp.attrs['transformation_matrix'] = dewarp_params['transformation_matrix']
    # grp.attrs['target_pixels_width'] = dewarp_params['target_pixels_width']
    # grp.attrs['target_pixels_height'] = dewarp_params['target_pixels_height']
    # grp.attrs['target_ratio'] = TARGET_RATIO
    # grp.attrs['selected_points'] = selected_points
    # grp.attrs['frame_range'] = (start, end)
    start, end = metadata['frame_range']
    dset_w = metadata['target_pixels_width']
    dset_h = metadata['target_pixels_height']
    dewarp_params['transformation_matrix'] = metadata['transformation_matrix']
    dewarp_params['target_pixels_width'] = metadata['target_pixels_width']
    dewarp_params['target_pixels_height'] = metadata['target_pixels_height']
    dewarp_params['target_ratio'] = metadata['target_ratio']
    dewarp_params['selected_points'] = metadata['selected_points']
    dewarp_params['frame_range'] = metadata['frame_range']
    bar = progressbar.ProgressBar(max_value=end-start)
    for i,idx in bar(enumerate(data_numbers[start:end])):
        img = data.get_frame(idx)
        dewarped_data = dewarp_data(img, dewarp_params)
        dset.resize((dset_h, dset_w, i + 1))
        dset[:, :, i] = dewarped_data

    dataset_handler.close_file()

# print(IrData(os.path.join(user_config.get_path('data_folder'), 'lfs_pmma_DE_6mm_tc_R4_0001')).data_numbers)
# dewarp_exp('lfs_pmma_DE_6mm_tc_R4_0001',IrData('lfs_pmma_DE_6mm_tc_R4_0001'))
if __name__ == '__main__':
    exp_names = ['lfs_pmma_DE_6mm_tc_R3_0001','lfs_pmma_DE_6mm_tc_R4_0001']
    exp_names += ['lfs_pmma_DE_6mm_tc_R2_CANON','lfs_pmma_DE_6mm_tc_R3_CANON','lfs_pmma_DE_6mm_tc_R4_CANON']

    for exp_name in exp_names:
        if 'CANON' in exp_name:
            data = ImageData(os.path.join(r'/Volumes/Tam Backup/OM/', exp_name.replace('_CANON',"")), 'JPG')
        else:
            data = IrData(os.path.join(user_config.get_path('data_folder'), exp_name))
        dewarp_exp(exp_name,data)
        calculate_edge_results_for_exp_name(exp_name)
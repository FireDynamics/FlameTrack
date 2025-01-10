import numpy as np
import cv2
import progressbar
from ir_reader.analysis.dataset_handler import *


def read_IR_data(filename: str) -> np.ndarray:
    """
    Read the IR data from the file. The data is expected to be in the [Data] section of the file, separated by ';'
    :param filename: filepath to the IR data file
    :return: IR data as numpy array
    """
    with open(filename, 'r', encoding='latin-1') as f:
        line = f.readline()
        while line:
            if line.startswith('[Data]'):
                return np.genfromtxt((line.replace(',', '.')[:-2] for line in f.readlines()), delimiter=';')
            line = f.readline()

    raise ValueError('No data found in file, check file format!')


def get_dewarp_parameters(corners, target_pixels_width=None, target_pixels_height=None, target_ratio=None) -> dict:
    """
    Get the dewarp parameters using the corners and the target pixels width and height
    :param target_ratio: target ratio of the dewarped data
    :param corners: selected corners of the data
    :param target_pixels_width: target width of the dewarped data
    :param target_pixels_height: target height of the dewarped data
    :return: dewarp parameters as dictionary
    """
    buffer = 1.1
    if all(x is None for x in [target_pixels_width, target_pixels_height, target_ratio]):
        raise ValueError('Either target_pixels_width and target_pixels_height or target_ratio must be provided')

    # Calculate target size if not provided
    source_corners = np.array(corners, dtype=np.float32)
    if target_pixels_width is None and target_pixels_height is None:
        max_width = max(source_corners[1][0] - source_corners[0][0], source_corners[2][0] - source_corners[3][0])
        max_height = max(source_corners[2][1] - source_corners[1][1], source_corners[3][1] - source_corners[0][1])
        target_pixels_height = int(max(max_height, max_width * target_ratio) * buffer)
        target_pixels_width = int(target_pixels_height / target_ratio)
    target_corners = np.array(
        [[0, 0], [target_pixels_width, 0], [target_pixels_width, target_pixels_height], [0, target_pixels_height]],
        dtype=np.float32)

    # Use getPerspectiveTransform instead of findHomography. findHomography is useful for multiple points since it is
    # able to reject outliers, since only 4 points are used, getPerspectiveTransform is sufficient
    transformation_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)

    return {
        'transformation_matrix': transformation_matrix,
        'target_pixels_width': target_pixels_width,
        'target_pixels_height': target_pixels_height,
        'target_ratio': target_pixels_height / target_pixels_width
    }


def dewarp_exp(exp_name, data, rotationfactor, frequency=10, testing=False, renew=False):
    dewarped_grp = dataset_handler.get_file(exp_name, 'a').get('dewarped_data', None)
    # Should not be necessary (but just in case keep commented)
    # if dewarped_grp is None:
    #     dewarped_grp = dataset_handler.get_file(exp_name, 'a').create_group('dewarped_data')
    #     dset = dewarped_grp.get('data', None)

    dset = dewarped_grp['data']
    metadata = dewarped_grp.attrs
    data_numbers = data.data_numbers
    dewarp_params = {}
    start, end = metadata['frame_range']
    dset_w = metadata['target_pixels_width']
    dset_h = metadata['target_pixels_height']
    dewarp_params['transformation_matrix'] = metadata['transformation_matrix']
    dewarp_params['target_pixels_width'] = metadata['target_pixels_width']
    dewarp_params['target_pixels_height'] = metadata['target_pixels_height']
    dewarp_params['target_ratio'] = metadata['target_ratio']
    dewarp_params['selected_points'] = metadata['selected_points']
    dewarp_params['frame_range'] = metadata['frame_range']

    map_x = np.arange(0, dset_w, 1)
    map_y = np.arange(0, dset_h, 1)
    map_x, map_y = np.meshgrid(map_x, map_y)
    transformation_matrix = np.linalg.inv(dewarp_params['transformation_matrix'])
    src_x = (transformation_matrix[0, 0] * map_x + transformation_matrix[0, 1] * map_y + transformation_matrix[
        0, 2]) / (transformation_matrix[2, 0] * map_x + transformation_matrix[2, 1] * map_y + transformation_matrix[
        2, 2])
    src_y = (transformation_matrix[1, 0] * map_x + transformation_matrix[1, 1] * map_y + transformation_matrix[
        1, 2]) / (transformation_matrix[2, 0] * map_x + transformation_matrix[2, 1] * map_y + transformation_matrix[
        2, 2])
    if dewarped_grp.get('src_x', None) is not None:
        del dewarped_grp['src_x']
    if dewarped_grp.get('src_y', None) is not None:
        del dewarped_grp['src_y']
    dewarped_grp.create_dataset('src_x', data=src_x)
    dewarped_grp.create_dataset('src_y', data=src_y)
    assumed_pixel_error = 0.5
    dewarped_grp.attrs['assumed_pixel_error'] = assumed_pixel_error
    dewarped_grp.attrs['error_unit'] = 'pixels'
    src_points = np.array([src_x.flatten(), src_y.flatten()]).reshape(*src_x.shape, -1)

    # err_x = assumed_pixel_error/np.linalg.norm(np.diff(src_points,axis=0),axis=2)
    # err_y = assumed_pixel_error/np.linalg.norm(np.diff(src_points,axis=1),axis=2)
    #
    # dewarped_grp.create_dataset('err_x', data=err_x)
    # dewarped_grp.create_dataset('err_y', data=err_y)
    src_x_map, src_y_map = cv2.convertMaps(src_x.astype(np.float32), src_y.astype(np.float32), cv2.CV_16SC2)

    if testing:
        start = len(data_numbers) // 2 - 10
        end = len(data_numbers) // 2 + 10

    bar = progressbar.ProgressBar(max_value=len(data_numbers[start:end:frequency]))
    for i, idx in bar(enumerate(data_numbers[start:end:frequency])):
        if not renew and dset.shape[2] > i + 2:
            continue
        img = data.get_frame(idx, rotationfactor)
        dewarped_data = cv2.remap(img, src_x_map, src_y_map, interpolation=cv2.INTER_LINEAR)
        dset.resize((dset_h, dset_w, i + 1))
        dset[:, :, i] = dewarped_data
    dataset_handler.close_file()
    return src_x, src_y


def dewarp_RCE_exp(experiment, rotationfactor, frequency=1, testing=False, renew=False, data_type='IR'):
    h5_file = experiment.h5_file
    data = experiment.get_data(data_type)
    dewarped_grp_left = h5_file['dewarped_data_left']
    dewarped_grp_right = h5_file['dewarped_data_right']

    src_x_maps = []
    src_y_maps = []
    dsets = []
    dset_ws = []
    dset_hs = []

    for dewarped_grp in [dewarped_grp_left, dewarped_grp_right]:

        dset = dewarped_grp['data']
        metadata = dewarped_grp.attrs
        data_numbers = data.data_numbers
        dewarp_params = {}
        start, end = metadata['frame_range']
        dset_w = metadata['target_pixels_width']
        dset_h = metadata['target_pixels_height']
        dewarp_params['transformation_matrix'] = metadata['transformation_matrix']
        dewarp_params['target_pixels_width'] = metadata['target_pixels_width']
        dewarp_params['target_pixels_height'] = metadata['target_pixels_height']
        dewarp_params['target_ratio'] = metadata['target_ratio']
        dewarp_params['selected_points'] = metadata['selected_points']
        dewarp_params['frame_range'] = metadata['frame_range']

        map_x = np.arange(0, dset_w, 1)
        map_y = np.arange(0, dset_h, 1)
        map_x, map_y = np.meshgrid(map_x, map_y)
        transformation_matrix = np.linalg.inv(dewarp_params['transformation_matrix'])
        src_x = (transformation_matrix[0, 0] * map_x + transformation_matrix[0, 1] * map_y + transformation_matrix[
            0, 2]) / (transformation_matrix[2, 0] * map_x + transformation_matrix[2, 1] * map_y + transformation_matrix[
            2, 2])
        src_y = (transformation_matrix[1, 0] * map_x + transformation_matrix[1, 1] * map_y + transformation_matrix[
            1, 2]) / (transformation_matrix[2, 0] * map_x + transformation_matrix[2, 1] * map_y + transformation_matrix[
            2, 2])
        if dewarped_grp.get('src_x', None) is not None:
            del dewarped_grp['src_x']
        if dewarped_grp.get('src_y', None) is not None:
            del dewarped_grp['src_y']
        dewarped_grp.create_dataset('src_x', data=src_x)
        dewarped_grp.create_dataset('src_y', data=src_y)
        assumed_pixel_error = 0.5
        dewarped_grp.attrs['assumed_pixel_error'] = assumed_pixel_error
        dewarped_grp.attrs['error_unit'] = 'pixels'
        src_x_map, src_y_map = cv2.convertMaps(src_x.astype(np.float32), src_y.astype(np.float32), cv2.CV_16SC2)
        src_x_maps.append(src_x_map)
        src_y_maps.append(src_y_map)
        dsets.append(dset)
        dset_ws.append(dset_w)
        dset_hs.append(dset_h)

    if testing:
        start = len(data_numbers) // 2 - 50
        end = len(data_numbers) // 2 + 50

    for i, idx in enumerate(data_numbers[start:end:frequency]):
        for dset, src_x_map, src_y_map, dset_w, dset_h in zip(dsets, src_x_maps, src_y_maps, dset_ws, dset_hs):
            if not renew and dset.shape[2] > i + 2:
                continue
            img = data.get_frame(idx, rotationfactor)
            dewarped_data = cv2.remap(img, src_x_map, src_y_map, interpolation=cv2.INTER_LINEAR)
            dset.resize((dset_h, dset_w, i + 1))
            dset[:, :, i] = dewarped_data
        yield idx


def dewarp_data(data, dewarp_params) -> np.ndarray:
    """
    Dewarp the data using the corners and the target pixels width and height
    :param data: data to dewarp
    :param dewarp_params: dewarp parameters from get_dewarp_parameters
    :return: dewarped data as numpy array
    """
    transformation_matrix = dewarp_params['transformation_matrix']
    target_pixels_width = dewarp_params['target_pixels_width']
    target_pixels_height = dewarp_params['target_pixels_height']
    dewarped_data = cv2.warpPerspective(data, transformation_matrix, (target_pixels_width, target_pixels_height))
    return dewarped_data


def sort_corner_points(points) -> list:
    """
    Sort the points anti-clockwise starting from the top left corner
    :param points: list of points
    :return: sorted points
    """
    points = np.array(points)

    # Get origin
    origin = np.mean(points, axis=0)

    sort_by_angle = lambda x: np.arctan2(x[1] - origin[1], x[0] - origin[0])
    points = sorted(points, key=sort_by_angle, reverse=True)
    return points

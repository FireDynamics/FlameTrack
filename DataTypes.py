import cv2
import h5py
import numpy as np
from IR_analysis import read_IR_data
import glob
import os

class DataClass:
    def __init__(self):
        self.data_numbers =[]

    def get_frame(self, framenr) -> np.ndarray:
        pass

    def get_frame_count(self) -> int:
        pass

    def get_frame_size(self) -> tuple[int, int]:
        return self.get_frame(0).shape


class VideoData(DataClass):
    def __init__(self, videofile):
        super().__init__()
        self.data = []
        cap = cv2.VideoCapture(videofile)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                self.data.append(frame)
            else:
                break
        self.data_numbers = list(range(self.get_frame_count()))
    def get_frame(self, framenr) -> np.ndarray:
        return cv2.cvtColor(self.data[framenr], cv2.COLOR_BGR2GRAY)

    def get_frame_count(self) -> int:
        return len(self.data)


class ImageData(DataClass):
    def __init__(self, image_folder, image_extension='JPG'):
        super().__init__()
        self.files = glob.glob(f'{image_folder}/*.{image_extension}')
        self.files = sorted(self.files, key=lambda x: os.path.getmtime(x))
        self.data_numbers = list(range(len(self.files)))

    def get_frame(self, framenr) -> np.ndarray:
        frame = cv2.imread(self.files[framenr])
        frame = frame [:,:,1]
        #resize frame to 1500x1000
        return cv2.resize(frame, (1500, 1000))

    def get_org_frame(self, framenr) -> np.ndarray:
        frame = cv2.imread(self.files[framenr])
        return cv2.resize(frame, (1500, 1000))

    def get_frame_count(self) -> int:
        return len(self.files)


class IrData(DataClass):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.files = glob.glob(f'{self.data_folder}/*.csv')
        self.files = sorted(self.files)
        self.data_numbers = list(range(len(self.files)))
        #Todo: Add check for missing files

    def __sort_files(self):
        pass
    def get_frame(self, framenr) -> np.ndarray:
        file = self.files[framenr]
        return read_IR_data(file)[::-1]

    def get_frame_count(self) -> int:
        return max(self.data_numbers)


class RCE_Experiment:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.exp_name = os.path.basename(folder_path)
        self.IR_data = None
        self.Video_data = None
        self.Picture_data = None
        self._h5_file = None


    @property
    def h5_file(self):
        try:
            self._h5_file.close()
        except:
            pass
        self._h5_file = h5py.File(os.path.join(self.folder_path, 'processed_data', self.exp_name + '_results_RCE.h5'), 'a')
        return self._h5_file
    @h5_file.setter
    def h5_file(self, value):
        self._h5_file = value




    def get_IR_data(self):
        if not os.path.exists(os.path.join(self.folder_path, 'exported_data')):
            raise FileNotFoundError('No exported data found')
        if self.IR_data is None:
            self.IR_data = IrData(os.path.join(self.folder_path, 'exported_data'))
        return self.IR_data

    def get_Video_data(self):
        if not os.path.exists(os.path.join(self.folder_path, 'video')):
            raise FileNotFoundError('No video data found')
        file = glob.glob(os.path.join(self.folder_path, 'video', '*.mp4'))[0]
        return VideoData(file)

    def get_Picture_data(self):
        if not os.path.exists(os.path.join(self.folder_path, 'images')):
            raise FileNotFoundError('No image data found')
        return ImageData(os.path.join(self.folder_path, 'images'))

    def get_processed_data(self):
        if not os.path.exists(os.path.join(self.folder_path, 'processed_data')):
            raise FileNotFoundError('No processed data found')
        return IrData(os.path.join(self.folder_path, 'processed_data'))



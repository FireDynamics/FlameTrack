import cv2
import numpy as np
from IR_analysis import read_IR_data
import glob

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


class IrData(DataClass):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.files = glob.glob(f'{self.data_folder}/*.csv')
        self.data_numbers = [int(file[-8:-4]) for file in self.files]

    def get_frame(self, framenr) -> np.ndarray:
        file = glob.glob(f'{self.data_folder}/*{framenr:04d}.csv')
        if len(file) == 0:
            raise ValueError('File not found')
        return read_IR_data(file[0])[::-1]

    def get_frame_count(self) -> int:
        return max(self.data_numbers)


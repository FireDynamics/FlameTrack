import cv2
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
    def __init__(self, image_folder, image_extension='jpg'):
        super().__init__()
        self.files = glob.glob(f'{image_folder}/*.{image_extension}')
        self.files = sorted(self.files, key=lambda x: os.path.getmtime(x))
        self.data_numbers = list(range(len(self.files)))

    def get_frame(self, framenr) -> np.ndarray:
        frame = cv2.imread(self.files[framenr])
        frame = frame [:,:,1]
        #resize frame to 640x480
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


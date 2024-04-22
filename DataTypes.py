import cv2
import numpy as np
from IR_analysis import read_IR_data

class DataClass:
    def __init__(self):
        pass

    def get_frame(self, framenr) -> np.ndarray:
        pass

    def get_frame_count(self) -> int:
        pass

    def get_frame_size(self) -> Tuple[int, int]:
        return self.get_frame(0).shape


class VideoData(DataClass):
    def __init__(self, videofile):
        super().__init__()
        cap = cv2.VideoCapture(videofile)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                self.data.append(frame)
            else:
                break
    def get_frame(self, framenr) -> np.ndarray:
        return self.data[framenr]

    def get_frame_count(self) -> int:
        return len(self.data)


class IrData(DataClass):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def get_frame(self, framenr) -> np.ndarray:
        file = glob.glob(f'{self.data_folder}/*{framenr:04d}.csv')
        if len(file) == 0:
            raise ValueError('File not found')
        return read_IR_data(file[0])

    def get_frame_count(self) -> int:
        return len(glob.glob(f'{self.data_folder}/*.csv'))


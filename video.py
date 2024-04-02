import cv2
import numpy as np
import matplotlib.pyplot as plt


data = np.load('dewarped_data/2_mal_1_5_dewarped.npy')
edges = np.load('edge_results.npy')


def make_edge_video(data, edges):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('edge_video.avi', fourcc, 20.0, (data.shape[1], data.shape[0]))

    for i in range(data.shape[-1]):
        frame = data[ ::-1, ::,i]
        edge = edges[i]
        #apply colormap
        frame = frame - np.min(frame)
        frame = frame / np.max(frame) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        for j in range(len(edge)):
            cv2.circle(frame, (edge[j], j), 1, (0, 255, 0), -1)
        out.write(frame)
    out.release()

make_edge_video(data, edges)



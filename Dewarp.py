import sys
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

cwd = os.getcwd()
os.chdir('../../Documents/Lehre_Wuppertal/Probe_Datensatz_IR_CAM/230217_124859')
cwd = os.getcwd()
print (cwd)




filename = 'Stufe_01_Prozent_005_0700.csv'
df = pd.read_csv(filename, decimal=',', header = None, comment = '#', delimiter=';', encoding='latin1', skiprows = 14)
data = np.array(df)
data = data [:,0:-1]
data_filename = data
corners_filename = 'corners.dat'
out_basefilename = 'test'
out_file_extension = 'test_02'
#out_basefilename, out_file_extension = os.path.splitext(os.path.basename(data_filename))
out_filename = out_basefilename + '_warp.dat'

print('reading thermography file: {}'.format(data_filename))

#target_pixels = 250
target_pixels_width = 250
target_pixels_hight = 5000
extent = 100


thermo_data = data
thermo_min = np.min(data)
thermo_max = np.max(data)
#thermo_max_scale = thermo_min + (thermo_max - thermo_min)*0.1
thermo_data = (thermo_data - thermo_min) / (thermo_max - thermo_min) * 255
thermo_data[thermo_data > 255] = 255

frame0 = np.zeros((thermo_data.shape[0], thermo_data.shape[1], 3), dtype=np.uint8)
frame0[:,:,0] = thermo_data
frame0[:,:,1] = thermo_data
frame0[:,:,2] = thermo_data
frame0_raw = np.copy(frame0)
frame_to_be_dewarped = frame0

zoom_range = 20
frame_zoom = frame0_raw[thermo_data.shape[0]//2-zoom_range:thermo_data.shape[0]//2+zoom_range, thermo_data.shape[1]//2-zoom_range:thermo_data.shape[1]//2+zoom_range]

cv2.namedWindow('thermal image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('thermal image', frame0)

cv2.namedWindow('zoom', cv2.WINDOW_AUTOSIZE)
cv2.imshow('zoom', cv2.pyrUp(cv2.pyrUp(frame_zoom)))

source_corners = []

def mouse_four_corners(event, x, y, d1, d2):
    global source_corners, frame0, frame0_raw, zoom_range, frame_zoom

    if event == cv2.EVENT_MOUSEMOVE:
        print('mouse moved at ', x, y)

        if x<zoom_range: x = zoom_range
        if y<zoom_range: y = zoom_range
        if x>(frame0.shape[1] - zoom_range): x = frame0.shape[1] - zoom_range
        if y>(frame0.shape[0] - zoom_range): y = frame0.shape[0] - zoom_range

        frame_zoom = np.copy(frame0_raw[y - zoom_range:y + zoom_range,
                     x - zoom_range:x + zoom_range])

        frame_zoom[zoom_range, zoom_range, 2] = 255

    if event == cv2.EVENT_LBUTTONDOWN:
        print('mouse click at ', x, y)
        source_corners.append([x,y])
        cv2.circle(frame0, (x,y), 10, (255,0,0), 2)

cv2.setMouseCallback('thermal image', mouse_four_corners)

while True:
    key=cv2.waitKey(1)
    cv2.imshow('thermal image', frame0)
    cv2.imshow('zoom', cv2.pyrUp(cv2.pyrUp(frame_zoom)))

    # enter -> warp movie
    if key==13: break

    # esc -> restart
    if key==27:
        source_corners = []
        frame0 = np.copy(frame0_raw)

cv2.destroyWindow('thermal image')
cv2.destroyWindow('zoom')

np.savetxt('corners.dat', source_corners, fmt='%i')

corners_filename = 'corners.dat'

source_corners = np.loadtxt(corners_filename)

if source_corners.shape != (4,2):
    print('ERROR: wrong shape of corner data: {}'.format(source_corners.shape))
    print(source_corners)
    sys.exit(0)

#target_corners = [[10,10], [10, target_pixels-10], [target_pixels-10, target_pixels-10], [target_pixels-10, 10]]
#target_corners = [[0,0], [target_pixels_hight, 0], [target_pixels_hight,target_pixels_width], [0, target_pixels_width]]
#target_corners = [[0,0], [0, target_pixels_hight], [target_pixels_hight,target_pixels_width], [target_pixels_width,0]]
target_corners = [[0,target_pixels_width], [0, 0], [target_pixels_hight,0], [target_pixels_hight,target_pixels_width]]


target_corners = np.array(target_corners, np.float32)

source_corners = np.array(source_corners, np.float32)

#print ('source corners', source_corners[0])

#print ('source corners', source_corners[0])

#M = cv2.getPerspectiveTransform(source_corners,target_corners)

#h, status = cv2.findHomography(np.array(source_corners), np.array(target_corners))

#frame0_warp = cv2.warpPerspective(frame0_raw, M, (target_pixels_width, target_pixels_hight),flags=cv2.INTER_LINEAR)

h, status = cv2.findHomography(source_corners, target_corners)

frame0_warp = cv2.warpPerspective(frame0, h, (target_pixels_hight, target_pixels_width))

thermo_warp = frame0_warp[:,:,0]

thermo_warp_ = thermo_warp * (thermo_max - thermo_min) / 255 + thermo_min

#print (np.shape(thermo_warp))

#print (thermo_warp.shape)

plt.imshow(thermo_warp_, origin='lower')
#plt.imshow(thermo_warp, aspect='auto')
#plt.imshow(thermo_warp, extent=(0, target_pixels_width, 0, target_pixels_hight), aspect='auto')
#plt.imshow(thermo_warp)

#plt.show()
#plt.imshow(thermo_warp, origin = 'lower', cmap='rainbow')

#for i in range (25,200,25):
#plt.plot(thermo_warp_[i,:])
#plt.imshow(image_warp, origin = 'lower', cmap='rainbow')
#plt.colorbar()
plt.show()
#plt.savefig(out_basefilename + '_warp.pdf')
#plt.clf()
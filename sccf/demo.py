import os
from SCCF import SCCF
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_img_lists(img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list

# Parameters
n = 0.0 # Online learning rate parameter
l = 0.0001#0.001 # gamma^2/v^2 where v - Controlling the prior variance of filter weight, gamma - 
num_training = 1000
rotation = False
frame_list = get_img_lists(os.path.dirname(__file__) + '/../datasets/imgs/')
frame_list.sort()

# Load first frame
first_img_path = frame_list[0]
first_img = cv2.imread(first_img_path)
rect = cv2.selectROI('demo', first_img, False, False)

# Get center and size of target
center_x = int(rect[0] + rect[2] / 2)
center_y = int(rect[1] + rect[3] / 2)
width = rect[2]
height = rect[3]

# Create tracker
tracker = SCCF(center_x, center_y, width, height, n, l)
target_2D, target_2D_norm = tracker.initialize(first_img)

# Show image with selected tracked rectangle
copied_img = first_img.copy()
cv2.rectangle(copied_img, tracker.get_target_rectangle(), (255, 0, 0), 2)
cv2.rectangle(copied_img, tracker.get_search_rectangle(), (0, 255, 0), 2)
cv2.imshow('init', copied_img)
cv2.waitKey(100)

# Set filter
tracker.set_filter(target_2D_norm)
# Test
#tracker.test(target_2D_norm)
tracker.test(first_img)

# Train filter
tracker.train_filter(target_2D, target_2D_norm, num_training, rotation)

#tracker.test(target_2D_norm)
tracker.test(first_img)
debug = False
for i in range(len(frame_list)):    
     frame = cv2.imread(frame_list[i])
     tracker.run(frame, debug)
     # Draw rectangle
     frame_copy = frame.copy()
     cv2.rectangle(frame_copy, tracker.get_target_rectangle(), (255, 0, 0), 3)
     cv2.rectangle(frame_copy, tracker.get_search_rectangle(), (0, 255, 0), 3)
     cv2.imshow('run', frame_copy)
     if cv2.waitKey(1) & 0xFF == ord('d'):
        debug = True
     else:
        debug = False

# # Plot result
DX = np.array(tracker.DX)
t = np.linspace(0, DX.size, DX.size)
DY = np.array(tracker.DY)
PX = np.array(tracker.PX)
PY = np.array(tracker.PY)
_, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(t, DX, label = "DX")
ax1.plot(t, DY, label = "DY")
ax1.legend()

ax2.plot(t, PX, label= "PX")
ax2.plot(t, PY, label= "PY")
ax2.legend()
plt.show()
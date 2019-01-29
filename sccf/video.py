import numpy as np
import cv2
from SCCF import SCCF

# Parameters
n = 0 # Online learning rate parameter
l = 0.001 # gamma^2/v^2 where v - Controlling the prior variance of filter weight, gamma - 
num_training = 1024
rotation = False

cap = cv2.VideoCapture(0)
# Capture frame-by-frame
# Load first frame
ret, first_img = cap.read()
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
first_image_copy = first_img.copy()
cv2.rectangle(first_img, tracker.get_target_rectangle(), (255, 0, 0), 2)
cv2.rectangle(first_img, tracker.get_search_rectangle(), (0, 255, 0), 2)
cv2.imshow('init', first_img)
cv2.waitKey(100)

# Set filter
tracker.set_filter(target_2D_norm)
# Test
tracker.test(target_2D_norm)

# Train filter
tracker.train_filter(target_2D, target_2D_norm, num_training, rotation)

debug = False
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()     
    tracker.run(frame, debug)
    # Draw rectangle
    frame_copy = frame.copy()
    cv2.rectangle(frame, tracker.get_target_rectangle(), (255, 0, 0), 3)
    cv2.rectangle(frame, tracker.get_search_rectangle(), (0, 255, 0), 3)
    cv2.imshow('run', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(20) & 0xFF == ord('d'):
        debug = True
    else:
        debug = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
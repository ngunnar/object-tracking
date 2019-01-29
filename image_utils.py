import numpy as np
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def pre_process(img):
    # Get size of image
    h, w = img.shape
    # Transform pixel values using log function which helps with low contrast lightning
    img = np.log(img + 1)
    # The pixels are normalized to have a mean of 0.0 and norm of 1.
    img = (img - np.mean(img)) / np.std(img)
    # Finally the image is mulitplied by a consine window which gradually reduce the pixel values near the edge to zero.
    return img * window_func_2d(h, w)

# used for linear mapping...
def linear_mapping(images):
    max_value = images.max()
    min_value = images.min()

    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a

    image_after_mapping = parameter_a * images + parameter_b

    return image_after_mapping


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row
    return win

def random_warp(img):
    h, w = img.shape
    a = -180 / 18
    b = 180 / 18
    r = a + (b - a) * np.random.uniform()
    # rotate the image...    
    matrix_rot = cv2.getRotationMatrix2D((w/2, h/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (w, h))    
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot
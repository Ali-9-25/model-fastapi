'''
This module is responsible for extracting the LBP (Linear Binary Pattern) features from the images.
'''

import cv2
import numpy as np
from skimage import feature

def extract_lbp_features(image, num_points=24, radius=8, eps=1e-7):
    # Resize the image
    # image = cv2.resize(image, (128, 128))
    
    # Compute the LBP representation of the image
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
    
    # Compute the histogram of the LBP
    hist = cv2.calcHist([lbp.astype('float32')], [0], None, [num_points + 2], [0, num_points + 2])
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    
    return hist
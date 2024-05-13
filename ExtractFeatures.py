from GaborFilter.GaborFilter import gabor_filter
from EnclosingArea.EnclosingArea import enclosing_area
import numpy as np

def extract_features(image):

    area = enclosing_area(image)
    gabor = gabor_filter(image)
    return np.concatenate([[area] , gabor])

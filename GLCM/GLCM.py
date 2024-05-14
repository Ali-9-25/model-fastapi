'''
This module extracts features from an image using Gray Level Co-occurrence Matrix (GLCM).
'''

import cv2
import numpy as np
import skimage as ski

def glcm_features(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True):
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute GLCM
    glcm = ski.feature.graycomatrix(image, distances, angles, levels, symmetric=symmetric, normed=normed)

    # Compute texture features
    contrast = ski.feature.graycoprops(glcm, 'contrast')
    dissimilarity = ski.feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = ski.feature.graycoprops(glcm, 'homogeneity')
    energy = ski.feature.graycoprops(glcm, 'energy')
    correlation = ski.feature.graycoprops(glcm, 'correlation')

    return np.array([contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]])
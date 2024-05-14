from GaborFilter.GaborFilter import get_gabor_vect
from EnclosingArea.EnclosingArea import enclosing_area
from GLCM.GLCM import glcm_features
from LBP.LBP import extract_lbp_features
from sklearn.preprocessing import StandardScaler
import numpy as np
import skimage as ski


def extract_features(image, kernels):
    area = enclosing_area(image)
    # convert area to a 1D array
    area = area.flatten()
    gabor = get_gabor_vect(image,  kernels)
    glcm = glcm_features(image)
    lbp = extract_lbp_features(image)
    # convert lbp to a 1D array
    lbp = lbp.flatten()
    features = np.concatenate((area, gabor, glcm, lbp))
    return features


def generate_kernels():
    kernels = []
    # mentioned in the papers  0  , 45 , 90 , 135
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for theta in thetas:
        for frequency in [0.05, 0.01, 0.15, 0.3]:
            kernel = ski.filters.gabor_kernel(frequency=frequency, theta=theta)
            kernels.append(np.real(kernel))

    return kernels


def normalize(x_data):
    scaler = StandardScaler()
    scaler.fit(x_data)
    return scaler

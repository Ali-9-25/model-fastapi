import skimage as ski 
from skimage.feature import local_binary_pattern
from skimage import io, measure, color , util, morphology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# Texture Feature (Gabor Feature extraction ) (as mention in the paper )

#  generate a gabor filters with different orientation and scales
#  apply all the filters to each image in the dataset
#  get the output of each image for each feature , calculate the statistical features of the output of each filter ``` mean  , standard deviation , local Energy```
#  concatenate the statistical features of all the filters to get the final feature vector of the image
#  since the gabor filter , should be implemnted for grey images with the same size, we can resize the image to a fixed size and convert it to grey image. 
#  i need a reasonable width and resonable hight 
#  so i will go through the dataset and get the average width and hight of the images and resize all the images to the average width and hight.  

#resizeing
def resize_image(image , width , height):
    return ski.transform.resize(image, (width, height))

#after , calculating width and height , i will choose some freqeuncies as mentioned in the paper (f<= N/4) where N is width of image N xN 
def get_frequencies(width , height) : 
    max_freq = width / 4
    frequencies = np.linspace(1, max_freq, num=5)

    frequencies = frequencies[1:]
    # frequencies= frequencies/max_freq
    return frequencies

#converte binary image to grayscale because gabor filter only works on grayscale images 
def convert_to_grayscale(binary_image):
    from skimage import img_as_ubyte
    grayscale_image = img_as_ubyte(binary_image)
    return grayscale_image


def apply_gabor_filters(image , thetas, frequencies):
    features=[]
    for theta in thetas : 
        for frequency in frequencies : 
            filtered_real, filtered_imag = ski.filters.gabor(image , frequency=frequency , theta = theta )
            mean_real = np.mean(filtered_real)
            std_dev_real = np.std(filtered_real)
            local_energy_real = np.sum(filtered_real**2)
            features.append(mean_real)
            features.append(std_dev_real)
            features.append(local_energy_real)
    return features

# apply gabor filters to the image
def gabor_filter(image):
    avg_width = 604.47125
    avg_height = 961.69775
    thetas = [0 , np.pi/4 , np.pi/2 , 3*np.pi/4] # mentioned in the papers  0  , 45 , 90 , 135
    frequencies = [0.1, 0.2, 0.3, 0.4]
    img = resize_image(image , avg_width , avg_height)
    gray_image = convert_to_grayscale(img)
    feature_vect = apply_gabor_filters(gray_image , thetas , frequencies)
    return feature_vect

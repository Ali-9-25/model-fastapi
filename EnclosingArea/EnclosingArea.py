import skimage as ski 
from skimage.feature import local_binary_pattern
from skimage import io, measure, color , util, morphology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# 1- SUM(area of closed chars)/(area of the image)
# to make the feature invariant to the size of the image. we can use the ratio of the sum of the area of the closed characters to the area of the image.

# TODO: we need to find more accurate way
def get_area_ratio(image):
    # this area_threshold is a hyperparameter
    mask = morphology.remove_small_holes(image , area_threshold=10)
    masked = np.where(mask, image , 255 ) 
    num_black_pixels = np.sum(masked == 0)
    total_area = masked.shape[0] * masked.shape[1]
    ratio = num_black_pixels / total_area
    return ratio 

def test_plot():
    import matplotlib.pyplot as plt

    # Assuming 'features' is your DataFrame and 'area_ratio' is your 1-D feature
    # and 'font_type' is your class label

    # Get unique classes
    def plot_1D_feature(features):
        classes = features['font_type'].unique()

        # Create a color map
        colors = ['r', 'g', 'b', 'y']  # Add more colors if you have more classes

        for i, cls in enumerate(classes):
            # Extract the feature values for this class
            feature_values = features[features['font_type'] == cls]['area_ratio']

            # Create an array of the same length as feature_values for the x-axis
            x = np.arange(len(feature_values))

            # Plot the feature values
            plt.scatter(x, feature_values, color=colors[i], label=cls)

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()

# Main function
def enclosing_area(image):
    return get_area_ratio(image)

"""
This is used to create image grids, of the various data that is looked at.
The process of creating the images is formed under the function called : plotprogressNegativePositive().
The image grid of rgb.png images from the negative and positive data is formed under the function called: plotprogressNegativePositive().
To get the random rgb.png images from the negative and positive data is formed under the function called: makeRandomRGBArray().
The plotting of all the image grids is done under the function called: plotAndSaveRgbGrid()
"""

# Processing images into a grid, to view all images at the same time, to view the process taken.
# IMPORTS
import glob
import os
import random
import re

import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits

from imageProcessorUtils import plotprogressNegativePositive, makeRandomRGBArray, plotAndSaveRgbGrid, plotKnownLenses
from positiveSetUtils import getNegativeNumbers

train_positive_path = 'Training/PositiveAll'
train_negative_path = 'Training/Negative'

number_iterations = 9  #or 12
# ___________________________________________________________________________________________________________________________________________
# MAIN 
# Number of Images creating grids to view.
# plot progress and rgb images of negative and positive images
numbers_train_neg = plotprogressNegativePositive(number_iterations)

# Get Random RGB images from PositiveWithDESSky
file_path6 = '%s_randomRGB_ImageGrid.png' % train_positive_path
rgb_random, image_title_array = makeRandomRGBArray(train_positive_path, number_iterations, numbers_train_neg)
plotAndSaveRgbGrid(file_path6, rgb_random, image_title_array)

# Get Random RGB images from NegativeDES
file_path7 = '%s_randomRGB_ImageGrid.png' % train_negative_path
rgb_random, image_title_array = makeRandomRGBArray(train_negative_path, number_iterations, numbers_train_neg)
plotAndSaveRgbGrid(file_path7, rgb_random, image_title_array)


# plot KnownLenses rgb images
plotKnownLenses(number_iterations, known_path='UnseenData/KnownLenses')

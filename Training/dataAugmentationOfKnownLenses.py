# Data Augmentation of Known Lenses, to create more lenses of the known lenses, to use as a positive trainging lenses.

# imports

# get Unseen set
import os
import sys

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from pandas import np

# variable
max_num_training = 3000  # Set to sys.maxsize when running entire data set
max_num_testing = sys.maxsize  # Set to sys.maxsize when running entire data set
image_shape = (100, 100, 3)  # The shape of the images being learned & evaluated.


def getUnseenData(images_dir, max_num, input_shape):
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))

        unseen_images = np.zeros([num_of_images, 3, 100, 100])
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/g_norm.fits' % (os.path.join(root, folder)))
            r_img_path = get_pkg_data_filename('%s/r_norm.fits' % (os.path.join(root, folder)))
            i_img_path = get_pkg_data_filename('%s/i_norm.fits' % (os.path.join(root, folder)))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = [g_data, r_data, i_data]
            unseen_images[index] = img_data
            index += 1

            if index >= num_of_images:
                break
        return unseen_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2])






# ______________________________________________________________________________________________________
# MAIN
known_47 = getUnseenData('UnseenData/Known47', max_num_training, input_shape=image_shape)
known_84 = getUnseenData('UnseenData/Known84', max_num_training, input_shape=image_shape)

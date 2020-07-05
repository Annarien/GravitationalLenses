# Data Augmentation of Known Lenses, to create more lenses of the known lenses, to use as a positive trainging lenses.

# imports

# get Unseen set
import os
import sys
import tensorflow as tf
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from pandas import np
from numpy.random import randint
from keras.preprocessing.image import ImageDataGenerator

# variable
max_num_training = 3000  # Set to sys.maxsize when running entire data set
max_num_testing = sys.maxsize  # Set to sys.maxsize when running entire data set
image_shape = (100, 100, 3)  # The shape of the images being learned & evaluated.
augmentedImages = []

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


def viewOriginalImage(array_known, file_name):
    first_original_image = array_known[0]
    original_image = plt.figure()
    plt.imshow(first_original_image[0])
    plt.show()
    print(first_original_image.shape)
    original_image.savefig('Training/Original_Image/Original_%s' % file_name)


def augmentImages(array_known):
    print('Augmenting some stuff')
    for i in range(0, len(array_known)):
        flipped = tf.image.flip_left_right(array_known[0])
        augmentedImages.append(flipped)
        saturated = tf.image.adjust_saturation(array_known[0], 3)
        bright = tf.image.adjust_brightness(array_known[0], 0.2)
        rotated = tf.image.rot90(array_known[0])
        cropped = tf.image.central_crop(array_known[0], central_fraction=0.3)


# ______________________________________________________________________________________________________
# MAIN
known_47 = getUnseenData('UnseenData/Known47', max_num_training, input_shape=image_shape)
known_84 = getUnseenData('UnseenData/Known84', max_num_training, input_shape=image_shape)

viewOriginalImage(known_47, 'known_47')
viewOriginalImage(known_84, 'known_84')

random_47 = randint(0, 47, 20)
random_84 = randint(0, 84, 40)

random_known_47 = getUnseenData('UnseenData/Known47', random_47, input_shape=image_shape)
random_known_84 = getUnseenData('UnseenData/Known84', random_84, input_shape=image_shape)

augmentImages(random_47)
augmentImages(random_84)


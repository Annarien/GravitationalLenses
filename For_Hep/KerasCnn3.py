"""
This is file performs the convolutional neural network algorithm, in which the k fold is performed as well.
The results were saved in a csv file.
"""

import os
import sys
import random
from datetime import datetime
import numpy as np
import tensorflow
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.vis_utils import plot_model

# added Adam opt for learning rate
# from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from ExcelUtils import createExcelSheet, writeToFile

print(tensorflow.__version__)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
print(dt_string)
excel_headers = []
excel_dictionary = []
excel_headers.append("Date and Time")
excel_dictionary.append(dt_string)

# Globals
makeNewCSVFile = False
max_num = sys.maxsize  # Set to sys.maxsize when running entire data set
max_num_testing = sys.maxsize  # Set to sys.maxsize when running entire data set
max_num_prediction = sys.maxsize  # Set to sys.maxsize when running entire data set
validation_split = 0.2  # A float value between 0 and 1 that determines what percentage of the training
# data is used for validation.
k_fold_num = 2  # A number between 2 and 10 that determines how many times the k-fold classifier
# is trained.
epochs = 7  # A number that dictates how many iterations should be run to train the classifier
batch_size = 128  # The number of items batched together during training.
run_k_fold_validation = False  # Set this to True if you want to run K-Fold validation as well.
input_shape = (100, 100, 3)  # The shape of the images being learned & evaluated.
augmented_multiple = 2  # This uses data augmentation to generate x-many times as much data as there is on file.
use_augmented_data = True  # Determines whether to use data augmentation or not.
patience_num = 3  # Used in the early stopping to determine how quick/slow to react.
use_early_stopping = False  # Determines whether to use early stopping or not.
use_model_checkpoint = True  # Determines whether the classifiers keeps track of the most accurate iteration of itself.
monitor_early_stopping = 'val_loss'
monitor_model_checkpoint = 'val_acc'
use_shuffle = True
learning_rate = 0.001

training_positive_path = 'Training/PositiveAll'
training_negative_path = 'Training/Negative'
testing_positive_path = 'Testing/PositiveAll'
testing_negative_path = 'Testing/Negative'
unseen_known_file_path = 'UnseenData/SelectingSimilarLensesToPositiveSimulated'
# unseen_known_file_path = 'UnseenData/KnownLenses'


# Adding global parameters to excel
excel_headers.append("Max Training Num")
excel_dictionary.append(max_num)
excel_headers.append("Max Testing Num")
excel_dictionary.append(max_num_testing)
excel_headers.append("Max Prediction Num")
excel_dictionary.append(max_num_prediction)
excel_headers.append("Validation Split")
excel_dictionary.append(validation_split)
excel_headers.append("K fold Num")
excel_dictionary.append(k_fold_num)
excel_headers.append("Epochs")
excel_dictionary.append(epochs)
excel_headers.append("Batch Size")
excel_dictionary.append(batch_size)
excel_headers.append("Run K fold")
excel_dictionary.append(run_k_fold_validation)
excel_headers.append("Input Shape")
excel_dictionary.append(input_shape)
excel_headers.append("Augmented Multiple")
excel_dictionary.append(augmented_multiple)
excel_headers.append("Use Augmented Data")
excel_dictionary.append(use_augmented_data)
excel_headers.append("Patience")
excel_dictionary.append(patience_num)
excel_headers.append("Use Early Stopping")
excel_dictionary.append(use_early_stopping)
excel_headers.append("Use Model Checkpoint")
excel_dictionary.append(use_model_checkpoint)
excel_headers.append("Monitor Early Stopping")
excel_dictionary.append(monitor_early_stopping)
excel_headers.append("Monitor Model Checkpoint")
excel_dictionary.append(monitor_model_checkpoint)
excel_headers.append("Use Shuffle")
excel_dictionary.append(use_shuffle)
excel_headers.append("Learning Rate")
excel_dictionary.append(learning_rate)

if not os.path.exists('../Results/%s/' % dt_string):
    os.mkdir('../Results/%s/' % dt_string)


# Helper methods
def getPositiveImages(images_dir, max_num, input_shape):
    """
    This gets the positively simulated images in the g, r and  i bands.
    Args:
        images_dir(string): This is the file path address of the positively simulated images.
        max_num(integer):   This is the number of sources of the positively simulated images to be used.
        input_shape(tuple): This is the shape of the images.
    Returns:
        positive_images(numpy array):   This is the numpy array of the positively simulated images with the shape of
                                        (num of images, input_shape[0], input_shape[1], input_shape[2]) =
                                        (num_of_images, 100, 100, 3).
    """
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        positive_images = np.zeros([num_of_images, 3, 100, 100])
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/%s_g_norm.fits' % (os.path.join(root, folder), folder))
            r_img_path = get_pkg_data_filename('%s/%s_r_norm.fits' % (os.path.join(root, folder), folder))
            i_img_path = get_pkg_data_filename('%s/%s_i_norm.fits' % (os.path.join(root, folder), folder))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = [g_data, r_data, i_data]
            positive_images[index] = img_data
            index += 1

            if index >= num_of_images:
                break
        return positive_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2])


def getNegativeImages(images_dir, max_num, input_shape):
    """
    This gets the negative images in the g, r and  i bands.
    Args:
        images_dir(string): This is the file path address of the negative images.
        max_num(integer):   This is the number of sources of the negative images to be used.
        input_shape(tuple): This is the shape of the images.
    Returns:
        negative_images(numpy array):   This is the numpy array of the negative images with the shape of
                                        (num of images, input_shape[0], input_shape[1], input_shape[2]) =
                                        (num_of_images, 100, 100, 3).
    """
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        negative_images = np.zeros([num_of_images, 3, 100, 100])
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/g_norm.fits' % (os.path.join(root, folder)))
            r_img_path = get_pkg_data_filename('%s/r_norm.fits' % (os.path.join(root, folder)))
            i_img_path = get_pkg_data_filename('%s/i_norm.fits' % (os.path.join(root, folder)))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = [g_data, r_data, i_data]
            negative_images[index] = img_data
            index += 1

            if index >= num_of_images:
                break
        return negative_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2])


def getUnseenData(images_dir, max_num, input_shape):
    """
        This gets the unseen images in the g, r and  i bands containing the identified known lenses.
        Args:
            images_dir(string): This is the file path address of the unseen images.
            max_num(integer):   This is the number of sources of the unseen images to be used.
            input_shape(tuple): This is the shape of the images.
        Returns:
            des_tiles(dictionary):   This is the dictionary of the unseen images with the shape of
                                            (num of images, input_shape[0], input_shape[1], input_shape[2]) =
                                            (num_of_images, 100, 100, 3).
        """

    des_tiles = {}

    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/g_norm.fits' % (os.path.join(root, folder)))
            r_img_path = get_pkg_data_filename('%s/r_norm.fits' % (os.path.join(root, folder)))
            i_img_path = get_pkg_data_filename('%s/i_norm.fits' % (os.path.join(root, folder)))

            # print(g_img_path)
            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            # print(np.shape(g_data))
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = np.array([g_data, r_data, i_data]).reshape(input_shape[0], input_shape[1], input_shape[2])
            des_tiles.update({folder: img_data})
            index += 1
            if index >= num_of_images:
                break

        return des_tiles


def makeImageSet(positive_images, negative_images=None, tile_names=None, shuffle_needed=use_shuffle):
    """
    This is used to create data set of images and labels, in which the positive and negative images are all
    combined and shuffled.
    Args:
        positive_images(numpy array):   This is the numpy array of the positively simulated images.
        negative_images(numpy array):   This is the numpy array of the negative images, this is set to a
                                        default of None.
        tile_names(list):               This is the dictionary of the unseen known lenses, this is set to a
                                        default of None.
        shuffle_needed(boolean):        This is a boolean value to determine whether or not shuffling of the given data
                                        sets is required.
    Returns:
        image_set(numpy array):         This is the image data set of  numpy array of the combination positive
                                        and negative images.
        label_set(numpy array):         This is the label data set of  numpy array of the combination positive
                                        and negative label.
        des_names_set(numpy array):     This is the des name data set of the known lenses and negative images used.
    """

    image_set = []
    label_set = []
    tile_name_set = []

    if positive_images is not None:
        for index in range(0, len(positive_images)):
            image_set.append(positive_images[index])
            label_set.append(1)
            if tile_names is not None:
                tile_name_set.append(tile_names[index])

    if negative_images is not None:
        for index in range(0, len(negative_images)):
            image_set.append(negative_images[index])
            label_set.append(0)
            if tile_names is not None:
                tile_name_set.append(tile_names[index])

    # print("Label Set: " + str(label_set))
    if shuffle_needed:
        if tile_names is not None:
            image_set, label_set, tile_name_set = shuffle(image_set, label_set, tile_name_set)
        else:
            image_set, label_set = shuffle(image_set, label_set)
    # print("Shuffled Label Set: " + str(label_set))

    return np.array(image_set), np.array(label_set), np.array(tile_name_set)


def buildClassifier(input_shape=(100, 100, 3)):
    """
    This creates the CNN algorithm.
    Args:
        input_shape(tuple): This is the image shape of (100,100,3)
    Returns:
        classifier(sequential): This is the sequential model.
    """
    # Initialising the CNN
    opt = Adam(lr=learning_rate)  # lr = learning rate
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    classifier.add(Dropout(0.5))  # added extra Dropout layer
    classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    classifier.add(Dropout(0.5))  # added extra dropout layer
    classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # classifier.add(Dropout(0.2))  # antes era 0.25
    # classifier.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # classifier.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # classifier.add(Flatten())  # This is added before dense layer a flatten is needed
    # classifier.add(Dense(units=1024, activation='relu'))  # added new dense layer
    classifier.add(Dropout(0.2))  # antes era 0.25
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='relu'))  # added new dense layer
    classifier.add(Dense(units=256, activation='relu'))  # added new dense layer
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.summary()

    # Compiling the CNN
    classifier.compile(optimizer=opt,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    #plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return classifier


def visualiseActivations(img_tensor, base_dir):
    """
    This makes images of the activations, as the selected image passed through the model
    Args:
        img_tensor(numpy array):    This is the numpy array of the selected image
        base_dir(string):           This is the file path name
    Saves:
        This saves the activation images of the selected source.
    """
    global predicted_class, size
    # Run prediction on that image
    predicted_class = classifier.predict_classes(img_tensor, batch_size=10)
    print("Predicted class is: ", predicted_class)
    # Visualize activations
    layer_outputs = [layer.output for layer in classifier.layers[:12]]
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    layer_names = []
    for layer in classifier.layers[:12]:
        layer_names.append(layer.name)
    images_per_row = 3
    count = 0
    for layer_name, layer_activation in zip(layer_names, activations):
        number_of_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        number_of_columns = number_of_features // images_per_row
        display_grid = np.zeros((size * number_of_columns, images_per_row * size))
        for col in range(number_of_columns):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        activations_figure = plt.figure(figsize=(scale * display_grid.shape[1],
                                                 scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        activations_figure.savefig('%s/%s_Activation_%s.png' % (base_dir, count, layer_name))
        plt.close()

        count += 1


def usingCnnModel(training_data, training_labels, val_data, val_labels):
    """
    This is using the CNN model and setting it up.
    Args:
        training_data(numpy arrays):    This is the numpy array of the training data.
        training_labels(numpy arrays):  This is the numpy array of the training labels.
        val_data(numpy arrays):         This is the numpy array of the validation data.
        val_labels(numpy arrays):       This is the numpy array of the validation labels.
    Returns:
        history(history):               This is the history of the classifier.
        classifier(sequential):         This is the cnn model classifier fitted to the training data and labels.
    """
    model_checkpoint = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor=monitor_model_checkpoint,
                                       save_best_only=True)

    early_stopping = EarlyStopping(monitor=monitor_early_stopping, patience=patience_num)  # original patience =3

    classifier = buildClassifier()
    callbacks_array = []
    if use_early_stopping:
        callbacks_array.append(early_stopping)
    if use_model_checkpoint:
        callbacks_array.append(model_checkpoint)

    print(len(training_data))
    history = classifier.fit(training_data,
                             training_labels,
                             epochs=epochs,
                             validation_data=(val_data, val_labels),
                             callbacks=callbacks_array,
                             batch_size=batch_size
                             # steps_per_epoch=int(len(training_data) / batch_size),
                             )
    return history, classifier


def createAugmentedData(training_data, training_labels):
    """
    This is creates the augmented data.
    Args:
        training_data(numpy arrays):    This is the numpy array of the training data.
        training_labels(numpy arrays):  This is the numpy array of the training labels.
    Returns:
        complete_training_data_set(numpy array):    This is the numpy array of the total training data, which is has
                                                    undergone augmentation.
        complete_training_labels_set(numpy array):  This is the numpy array of the total training labels, which is has
                                                    undergone augmentation.
    """
    complete_training_data_set = []
    complete_training_labels_set = []

    for data in training_data:
        complete_training_data_set.append(data)
    print("Complete Training Data: " + str(len(complete_training_data_set)))

    for label in training_labels:
        complete_training_labels_set.append(label)
    print("Complete Training Label: " + str(len(complete_training_labels_set)))

    # create augmented data
    data_augmented = ImageDataGenerator(featurewise_center=True,
                                        featurewise_std_normalization=True,
                                        rotation_range=90,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True)

    # data_augmented = ImageDataGenerator(featurewise_center=False,
    #                                     featurewise_std_normalization=False,
    #                                     rotation_range=90,
    #                                     horizontal_flip=True,
    #                                     vertical_flip=True)
    data_augmented.fit(training_data)

    training_data_size = training_data.shape[0]
    aug_counter = 0
    while aug_counter < (augmented_multiple - 1):
        iterator = data_augmented.flow(training_data, training_labels, batch_size=training_data_size)
        # iterator = data_augmented.flow(training_data, training_labels, batch_size=batch_size)
        augmented_data = iterator.next()
        for data in augmented_data[0]:
            complete_training_data_set.append(data)
        for label in augmented_data[1]:
            complete_training_labels_set.append(label)
        aug_counter += 1

    print("Size of All Training Data: " + str(len(complete_training_data_set)))
    print("Size of All Training Labels: " + str(len(complete_training_labels_set)))

    array_training_data = np.array(complete_training_data_set)
    array_training_labels = np.array(complete_training_labels_set)

    print("Shape of complete training data: " + str(array_training_data.shape))
    print("Shape of complete training labels: " + str(array_training_labels.shape))

    return np.array(complete_training_data_set), np.array(complete_training_labels_set)


def savePredictedLenses(des_names_array, predicted_class_probabilities, predicted_lenses_filepath, text_file_path):
    """
    This saves the names of the predicted lenses in the respective textfiles.
    Args:
        des_names_array(numpy array): This is a list of the des names of the sources.
        predicted_class_probabilities(list):    This is a list of the probabilities in which lenses are predicted by
                                                the algorithm.
        predicted_lenses_filepath(string):      This is the string of the predicted lenses filepath, where this needs
                                                to be saved in the directory.
        text_file_path(string):                 This is the text file path address to which these images are saved.
    Saves:
        text_file(.txt file):                   This is the text file saved containing the predicted lenses DES names.
    """
    predicted_lenses = []
    predicted_no_lenses = []
    if not os.path.exists(predicted_lenses_filepath):
        os.mkdir('%s/' % predicted_lenses_filepath)
    text_file = open('%s' % text_file_path, "a+")
    text_file.write('\n')
    text_file.write('Predicted Lenses: \n')
    for lens_index in range(len(predicted_class_probabilities)):
        if predicted_class_probabilities[lens_index] == 1:
            text_file.write("%s \n " % des_names_array[lens_index])
            predicted_lenses.append(des_names_array[lens_index])

    text_file.write('\n')
    text_file.write('No Lenses Predicted: \n')
    for lens_index in range(len(predicted_class_probabilities)):
        if predicted_class_probabilities[lens_index] == 0:
            text_file.write("%s \n " % des_names_array[lens_index])
            predicted_no_lenses.append(des_names_array[lens_index])
    text_file.close()

    return predicted_lenses, predicted_no_lenses


def gettingTrueFalsePositiveNegatives(testing_data, testing_labels, text_file_path,
                                      predicted_lenses_filepath, kf_counter=0):
    """
    This is used to get the True/False Positive and Negative values gained from the CNN confusion matrix.
    Args:
        testing_data(numpy array):          This is the unseen testing data numpy array.
        testing_labels(numpy array):        This is the unseen testing label numpy array.
        text_file_path(string):             This is the file path name of the text file in which the confusion
                                            matrix is saved.
        predicted_lenses_filepath(string):  This is the file path in which the text file is saved.
    Saves:
        This saves a confusion matrix of the True/False Positive and Negative values.
    """
    if not os.path.exists(predicted_lenses_filepath):
        os.mkdir('%s/' % predicted_lenses_filepath)

    predicted_data = classifier.predict_classes(testing_data)
    rounded_predicted_data = predicted_data.round()
    conf_matrix = confusion_matrix(testing_labels, rounded_predicted_data, labels=[0, 1])
    print(str(conf_matrix) + ' \n ')
    true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()
    print("True Positive: %s \n" % true_positive)
    print("False Negative: %s \n" % false_negative)
    print("False Positive: %s \n" % false_positive)
    print("True Negative: %s \n" % true_negative)

    text_file = open('%s' % text_file_path, "a+")
    text_file.write('\n')
    text_file.write('KFold Number: %s \n' % str(kf_counter))
    text_file.write('Predicted vs True Matrix: \n')
    text_file.write(str(conf_matrix) + " \n ")
    text_file.write("True Negative: %s \n" % str(true_negative))
    text_file.write("False Positive: %s \n" % str(false_positive))
    text_file.write("False Negative: %s \n" % str(false_negative))
    text_file.write("True Positive: %s \n" % str(true_positive))
    text_file.write("\n")
    text_file.close()

    confusion_matrix_array = [true_negative, false_positive, false_negative, true_positive]
    return confusion_matrix_array


def gettingKFoldConfusionMatrix(test_data, test_labels, unseen_images, unseen_labels, kf_counter):
    test_confusion_matrix = gettingTrueFalsePositiveNegatives(test_data,
                                                              test_labels,
                                                              text_file_path='../Results/%s/TrainingTestingResults'
                                                                             '/KFold_PredictedMatrix.txt' % dt_string,
                                                              predicted_lenses_filepath='../Results/%s/TrainingTestingResults'
                                                                                        % dt_string,
                                                              kf_counter=kf_counter)
    unseen_confusion_matrix = gettingTrueFalsePositiveNegatives(unseen_images,
                                                                unseen_labels,
                                                                text_file_path='../Results/%s/UnseenKnownLenses/'
                                                                               'KFold_LensesPredicted.txt' % dt_string,
                                                                predicted_lenses_filepath='../Results/%s/UnseenKnownLenses/'
                                                                                          % dt_string,
                                                                kf_counter=kf_counter)
    return test_confusion_matrix, unseen_confusion_matrix


def gettingRandomUnseenImage(filepath):
    g_img_path = get_pkg_data_filename('%s/g_norm.fits' % filepath)
    r_img_path = get_pkg_data_filename('%s/r_norm.fits' % filepath)
    i_img_path = get_pkg_data_filename('%s/i_norm.fits' % filepath)

    g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
    r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
    i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

    img_data = np.array([g_data, r_data, i_data]).reshape(input_shape[0], input_shape[1], input_shape[2])
    return img_data


def executeKFoldValidation(train_data, train_labels, val_data, val_labels, testing_data, testing_labels,
                           known_images, known_labels, known_des_names):
    """
    This does the k fold cross validation which is tested against the unseen testing and known lenses.
    Args:
        train_data(numpy arrays):           This is the numpy array of the training data.
        train_labels(numpy arrays):         This is the numpy array of the training labels.
        val_data(numpy arrays):             This is the numpy array of the validation data.
        val_labels(numpy arrays):           This is the numpy array of the validation labels.
        testing_data(numpy array):          This is the numpy array of the unseen testing data.
        testing_labels(numpy array):        This is the numpy array of the unseen testing label.
        images_47(numpy array):             This is the numpy array of the unseen DES images data.
        labels_47(numpy array):             This is the numpy array of the unseen DES images labels.
        images_84(numpy array):             This is the numpy array of the unseen Jacobs images data.
        labels_84(numpy array):             This is the numpy array of the unseen Jacobs images labels.
        all_unseen_images(numpy array):     This is the numpy array of the unseen DES + Jacobs images data.
        all_unseen_labels(numpy array):     This is the numpy array of the unseen DES + Jacobs images labels.

    Saves:
        This saves the scores, mean and std. of the unseen data that is evaluated in the k fold cross validation.
    """
    if run_k_fold_validation:
        print("In executingKFoldValidation")

        # this is doing it manually:
        kfold = StratifiedKFold(n_splits=k_fold_num, shuffle=True)

        test_scores_list = []
        unseen_scores_list = []
        test_matrix_list = []
        unseen_matrix_list = []
        kf_counter = 0
        true_positives = {}
        false_negatives = {}

        for train, test in kfold.split(train_data, train_labels):
            kf_counter += 1
            print('KFold #:', kf_counter)

            model = buildClassifier()
            # fit the model
            model.fit(train_data[train],
                      train_labels[train],
                      epochs=epochs,
                      validation_data=(val_data, val_labels),
                      batch_size=batch_size)

            test_scores = model.evaluate(testing_data, testing_labels, batch_size=batch_size)
            test_scores_list.append(test_scores[1])
            print(test_scores_list)
            unseen_scores = model.evaluate(known_images, known_labels, batch_size=batch_size)
            unseen_scores_list.append(unseen_scores[1])
            print(unseen_scores_list)

            # show confusion matrix
            test_confusion_matrix, unseen_confusion_matrix = gettingKFoldConfusionMatrix(testing_data,
                                                                                         testing_labels, known_images,
                                                                                         known_labels, kf_counter)

            probabilities_known_lenses = classifier.predict_classes(known_images, batch_size=batch_size)
            predicted_lens = np.count_nonzero(probabilities_known_lenses == 1)
            predicted_no_lens = np.count_nonzero(probabilities_known_lenses == 0)
            print("%s/%s known lenses predicted" % (predicted_lens, len(known_images)))
            print("%s/%s  non known lenses predicted" % (predicted_no_lens, len(known_images)))

            predicted_lenses, predicted_no_lenses = savePredictedLenses(known_des_names,
                                                                        predicted_class_probabilities_known_lenses,
                                                                        text_file_path='../Results/%s'
                                                                                       '/UnseenKnownLenses/'
                                                                                       'KFold_LensesPredicted.txt'
                                                                                       % dt_string,
                                                                        predicted_lenses_filepath='../Results/%s/'
                                                                                                  'UnseenKnownLenses'
                                                                                                  % dt_string)

            randomTP = None
            imageTP = None
            if predicted_lenses:
                randomTP = random.choice(predicted_lenses)
                filepathTP = unseen_known_file_path + '/%s' % randomTP
                imageTP = gettingRandomUnseenImage(filepathTP)
            true_positives[kf_counter] = (randomTP, imageTP)

            randomFN = None
            imageFN = None
            if predicted_no_lenses:
                randomFN = random.choice(predicted_no_lenses)
                filepathFN = unseen_known_file_path + '/%s' % randomFN
                imageFN = gettingRandomUnseenImage(filepathFN)
            false_negatives[kf_counter] = (randomFN, imageFN)

            # print("Lenses Predicted: " + str(randomTP))
            # print("Lenses Not Predicted: " + str(randomFN))

            test_matrix_list.append(test_confusion_matrix)
            unseen_matrix_list.append(unseen_confusion_matrix)

        test_scores_mean = np.mean(test_scores_list)
        test_scores_std = np.std(test_scores_list)
        unseen_scores_mean = np.mean(unseen_scores_list)
        unseen_scores_std = np.std(unseen_scores_list)

        print("Test Scores: " + str(test_scores_list))
        print("Test Scores Mean: " + str(test_scores_mean))
        print("Test Scores Std: " + str(test_scores_std))
        print("Test Confusion Matrices: " + str(test_matrix_list))
        print("Unseen Scores: " + str(unseen_scores_list))
        print("Unseen Scores Mean: " + str(unseen_scores_mean))
        print("Unseen Scores Std: " + str(unseen_scores_std))
        print("Unseen Confusion Matrices: " + str(unseen_matrix_list))

        excel_headers.append("Test Scores Mean")
        excel_dictionary.append(test_scores_mean)
        excel_headers.append("Test Scores Std")
        excel_dictionary.append(test_scores_std)
        excel_headers.append("Unseen Known Lenses Mean")
        excel_dictionary.append(unseen_scores_mean)
        excel_headers.append("Unseen Known Lenses Std")
        excel_dictionary.append(unseen_scores_std)

        plt.plot(test_scores_list, color='red', label='Testing Scores')
        plt.plot(unseen_scores_list, color='blue', label='Unseen Known Lenses Scores')

        plt.xlabel('Folds')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.savefig('../Results/%s/KFoldAccuracyScores.png' % dt_string)

        plotKFold(true_positives, false_negatives)


def viewActivationLayers():
    # make positive and negative directory
    if not os.path.exists('../Results/%s/PositiveResults/' % dt_string):
        os.mkdir('../Results/%s/PositiveResults/' % dt_string)

    if not os.path.exists('../Results/%s/NegativeResults/' % dt_string):
        os.mkdir('../Results/%s/NegativeResults/' % dt_string)

    # Plot original positive image
    img_positive_tensor = getPositiveImages('Training/PositiveAll', 1, input_shape=input_shape)
    positive_train_figure = plt.figure()
    plt.imshow(img_positive_tensor[0])
    # plt.show()
    print(img_positive_tensor.shape)
    positive_train_figure.savefig('../Results/%s/PositiveResults/PositiveTrainingFigure.png' % dt_string)
    plt.close()

    # Visualise Activations of positive image
    visualiseActivations(img_positive_tensor, base_dir='../Results/%s/PositiveResults/' % dt_string)

    # Plot original negative image
    img_negative_tensor = getNegativeImages('Training/Negative', 1, input_shape=input_shape)
    negative_train_figure = plt.figure()
    plt.imshow(img_negative_tensor[0])
    # plt.show()
    print(img_negative_tensor.shape)
    negative_train_figure.savefig('../Results/%s/NegativeResults/NegativeTrainingFigure.png' % dt_string)
    plt.close()

    # Visualise Activations of negative image
    visualiseActivations(img_negative_tensor, base_dir='../Results/%s/NegativeResults/' % dt_string)


def plotKFold(true_positives, false_negatives):
    # print('True Positives: ' + str(true_positives))
    # print('False Negatives: ' + str(false_negatives))
    fig, axs = plt.subplots(k_fold_num, 2)
    fig.tight_layout(pad=3.0)

    cols = ['True Positive', 'False Negative']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    # for ax, col in zip(axs[0], cols):
    # for i in range(len(cols)):
    #     #     axs[0, i].text(x=0.5, y=12, s="", ha="center", fontsize=12)
    #     # axs[k_fold_num - 1, i].set_xlabel(cols[i])
    #     axs[0, i].set_title(cols[i])
    #     # ax.set_title(col)

    for i in range(0, k_fold_num):
        axs[i, 0].text(x=-0.8, y=5, s="", rotation=90, va="center")
        axs[i, 0].set_ylabel("k = %s" % (i + 1))

        true_positive_tuple = true_positives[k_fold_num]
        if not true_positive_tuple[0] is None:
            axs[i, 0].set_xlabel(true_positive_tuple[0], fontsize=8)
            # axs[i, 0].set_title(true_positive_tuple[0], fontsize=6)
            axs[i, 0].imshow(true_positive_tuple[1])
        axs[i, 0].set_xticks([], [])
        axs[i, 0].set_yticks([], [])

        false_negative_tuple = false_negatives[k_fold_num]
        if not false_negative_tuple[0] is None:
            axs[i, 1].set_xlabel(false_negative_tuple[0], fontsize=8)
            # axs[i, 1].set_title(false_negative_tuple[0], fontsize=6)
            axs[i, 1].imshow(false_negative_tuple[1])
        axs[i, 1].set_xticks([], [])
        axs[i, 1].set_yticks([], [])

    fig.tight_layout()
    plt.show()
    fig.savefig('../Results/%s/UnseenKnownLenses/KFoldImages.png' % dt_string)


# __________________________________________________________________________
# MAIN

# Get positive training data
train_pos = getPositiveImages(images_dir=training_positive_path, max_num=max_num, input_shape=input_shape)
print("Train Positive Shape: " + str(train_pos.shape))
excel_headers.append("Train_Positive_Shape")
excel_dictionary.append(train_pos.shape)

# Get negative training data
train_neg = getNegativeImages(images_dir=training_negative_path, max_num=max_num, input_shape=input_shape)
print("Train Negative Shape: " + str(train_neg.shape))
excel_headers.append("Train_Negative_Shape")
excel_dictionary.append(train_neg.shape)

all_training_data, all_training_labels, _ = makeImageSet(train_pos, train_neg, shuffle_needed=use_shuffle)
if use_augmented_data:
    all_training_data, all_training_labels = createAugmentedData(all_training_data, all_training_labels)

training_data, val_data, training_labels, val_labels = train_test_split(all_training_data,
                                                                        all_training_labels,
                                                                        test_size=validation_split,
                                                                        shuffle=True)
excel_headers.append("All_Training_Data_Shape")
excel_dictionary.append(all_training_labels.shape)
excel_headers.append("All_Training_Labels_Shape")
excel_dictionary.append(all_training_labels.shape)
excel_headers.append("Training_Data_Shape")
excel_dictionary.append(training_data.shape)
excel_headers.append("Validation_Data_Shape")
excel_dictionary.append(val_data.shape)
excel_headers.append("Training_Labels_Shape")
excel_dictionary.append(training_labels.shape)
excel_headers.append("Validation_Labels_Shape")
excel_dictionary.append(val_labels.shape)
excel_headers.append("Validation_Split")
excel_dictionary.append(validation_split)

history, classifier = usingCnnModel(training_data,
                                    training_labels,
                                    val_data,
                                    val_labels)

#classifier.load_weights('best_weights.hdf5')
#classifier.save_weights('galaxies_cnn.h5')

excel_headers.append("Epochs")
excel_dictionary.append(epochs)
excel_headers.append("Batch_size")
excel_dictionary.append(batch_size)

# Plot run metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
number_of_completed_epochs = range(1, len(acc) + 1)

# Accuracies
train_val_accuracy_figure = plt.figure()
plt.plot(number_of_completed_epochs, acc, label='Training acc')
plt.plot(number_of_completed_epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
train_val_accuracy_figure.savefig('../Results/%s/TrainingValidationAccuracy.png' % dt_string)
plt.close()

# Losses
train_val_loss_figure = plt.figure()
plt.plot(number_of_completed_epochs, loss, label='Training loss')
plt.plot(number_of_completed_epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
train_val_loss_figure.savefig('../Results/%s/TrainingValidationLoss.png' % dt_string)
plt.close()

# make positive and negative results and plotting the activations of positive and negative images
#viewActivationLayers()

# Classifier evaluation
test_pos = getPositiveImages(images_dir=testing_positive_path, max_num=max_num_testing, input_shape=input_shape)
test_neg = getNegativeImages(images_dir=testing_negative_path, max_num=max_num_testing, input_shape=input_shape)
testing_data, testing_labels, _ = makeImageSet(test_pos, test_neg, shuffle_needed=True)
print("Testing Data Shape:  " + str(testing_data.shape))
print("Testing Labels Shape: " + str(testing_labels.shape))
print("Got Unseen Testing data")

scores = classifier.evaluate(testing_data, testing_labels, batch_size=batch_size)
loss = scores[0]
accuracy = scores[1]
print("Test loss: %s" % loss)
print("Test accuracy: %s" % accuracy)

excel_headers.append("Test_Loss")
excel_dictionary.append(loss)
excel_headers.append("Test_Accuracy")
excel_dictionary.append(accuracy)

gettingTrueFalsePositiveNegatives(testing_data,
                                  testing_labels,
                                  text_file_path='../Results/%s/TrainingTestingResults/PredictedMatrixBeforeKFOLD.txt'
                                                 % dt_string,
                                  predicted_lenses_filepath='../Results/%s/TrainingTestingResults' % dt_string)

unseen_known_images = getUnseenData(images_dir=unseen_known_file_path,
                                    max_num=max_num_prediction,
                                    input_shape=input_shape)

known_images, known_labels, known_des_names = makeImageSet(positive_images=list(unseen_known_images.values()),
                                                           tile_names=list(unseen_known_images.keys()),
                                                           shuffle_needed=True)
print("Unseen Known Images Shape:  " + str(known_images.shape))
print("Unseen Known Labels Shape: " + str(known_labels.shape))
print("Got Unseen Known Lenses Data")

unseen_scores = classifier.evaluate(known_images, known_labels, batch_size=batch_size)
unseen_loss_score = unseen_scores[0]
unseen_accuracy_score = unseen_scores[1]
print("Unseen loss: %s" % unseen_loss_score)
print("Unseen accuracy: %s" % unseen_accuracy_score)

excel_headers.append("Unseen_Loss")
excel_dictionary.append(unseen_loss_score)
excel_headers.append("Unseen_Accuracy")
excel_dictionary.append(unseen_accuracy_score)

predicted_class_probabilities_known_lenses = classifier.predict_classes(known_images, batch_size=batch_size)
lens_predicted = np.count_nonzero(predicted_class_probabilities_known_lenses == 1)
non_lens_predicted = np.count_nonzero(predicted_class_probabilities_known_lenses == 0)
print("%s/%s known lenses predicted" % (lens_predicted, len(known_images)))
print("%s/%s  non known lenses predicted" % (non_lens_predicted, len(known_images)))

gettingTrueFalsePositiveNegatives(known_images, known_labels,
                                  text_file_path='../Results/%s/UnseenKnownLenses/PredictedMatrixBeforeKFOLD.txt' % dt_string,
                                  predicted_lenses_filepath='../Results/%s/UnseenKnownLenses' % dt_string)

predicted_lenses, predicted_no_lenses = savePredictedLenses(known_des_names,
                                                            predicted_class_probabilities_known_lenses,
                                                            text_file_path='../Results/%s/UnseenKnownLenses/'
                                                                           'PredictedMatrixBeforeKFOLD.txt' % dt_string,
                                                            predicted_lenses_filepath='../Results/%s/UnseenKnownLenses'
                                                                                      % dt_string)

excel_headers.append("Unseen_Known_Lenses_Predicted")
excel_dictionary.append(lens_predicted)
excel_headers.append("Unseen_Known_Lenses_No_Lens_Predicted")
excel_dictionary.append(non_lens_predicted)

# K fold for training data
executeKFoldValidation(training_data,
                       training_labels,
                       val_data,
                       val_labels,
                       testing_data,
                       testing_labels,
                       known_images,
                       known_labels,
                       known_des_names)

if makeNewCSVFile:
    createExcelSheet('../Results/Architecture_kerasCNN_Results.csv', excel_headers)
    writeToFile('../Results/Architecture_kerasCNN_Results.csv', excel_dictionary)
else:
    writeToFile('../Results/Architecture_kerasCNN_Results.csv', excel_dictionary)

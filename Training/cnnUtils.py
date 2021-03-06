import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import makeExcelTable
from tensorflow.python.keras.callbacks import EarlyStopping, History
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def getPositiveSimulatedTrain(base_dir='Training/Positive'):
    """
    This gets the g, r, and i of the 10 000 positively simulated images from the
    PositiveWithDESSky, as well as returning the positively simulate array.

    Args:
        base_dir (string):      This the root file path of the positively simulated images.
    Returns:
        data_pos(numpy array):   This is the array of positively simulated images.
    """

    folders = {}
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value
            if len(folders) >= 5000:
                break

    # number of Positive DataPoints
    num_data_target = len(folders)

    data_pos = np.zeros([num_data_target, 3, 100, 100])  # this is the original
    # data_pos = np.zeros([num_data_target, 100, 100, 3])

    # key is name of folder number
    # value is the number of the folder to be added to the file name

    counter = 0
    for key, value in folders.items():
        g_name = get_pkg_data_filename(value + '/' + str(key) + '_g_norm.fits')
        r_name = get_pkg_data_filename(value + '/' + str(key) + '_r_norm.fits')
        i_name = get_pkg_data_filename(value + '/' + str(key) + '_i_norm.fits')

        # g_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_g.fits')
        # r_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_r.fits')
        # i_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_i.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        data_pos[counter] = [g, r, i]
        counter += 1
    print("GOT POSITIVE TRAINING DATA")
    return data_pos


def getNegativeDESTrain(base_dir='Training/Negative'):
    """
    This gets the g, r, and i  10 000 negative images from the
    DES/DES_Processed folder, as well as returning the
    negative array,

    Args:
        base_dir (string):      This the root file path of the negative images.
    Returns:
        data_neg (numpy array):  This is the array of negative images.
    """
    folders_neg = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            folders_neg.append(os.path.join(root, folder))
            if len(folders_neg) >= 5000:
                break

    num_data_target = len(folders_neg)
    data_neg = np.zeros([num_data_target, 3, 100, 100])  # original
    # data_neg = np.zeros([num_data_target, 100, 100, 3])

    for var in range(len(folders_neg)):
        # g_name = get_pkg_data_filename(folders_neg[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(folders_neg[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(folders_neg[var]+'/i_WCSClipped.fits')

        g_name = get_pkg_data_filename(folders_neg[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(folders_neg[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(folders_neg[var] + '/i_norm.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        data_neg[var] = [g, r, i]
        # just to run, and use less things

    print("GOT NEGATIVE TRAINING DATA")
    return data_neg


def getPositiveSimulatedTest(base_dir='Testing/Positive'):
    folders = {}
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value
            if len(folders) >= 1000:
                break

    # number of Positive DataPoints
    num_data_target = len(folders)

    positive_test = np.zeros([num_data_target, 3, 100, 100])  # original
    # positive_test = np.zeros([num_data_target, 100, 100, 3])
    # key is name of folder number
    # value is the number of the folder to be added to the file name

    counter = 0
    for key, value in folders.items():
        g_name = get_pkg_data_filename(value + '/' + str(key) + '_g_norm.fits')
        r_name = get_pkg_data_filename(value + '/' + str(key) + '_r_norm.fits')
        i_name = get_pkg_data_filename(value + '/' + str(key) + '_i_norm.fits')

        # g_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_g.fits')
        # r_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_r.fits')
        # i_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_i.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        positive_test[counter] = [g, r, i]
        counter += 1

    print("GOT POSITIVE TESTING DATA")
    return positive_test


def getNegativeDESTest(base_dir='Testing/Negative'):
    """
    This gets the g, r, and i  10 000 negative images from the
    DES/DES_Processed folder, as well as returning the
    negative array,

    Args:
        base_dir (string):      This the root file path of the negative images.
    Returns:
        negative_test (numpy array):  This is the array of negative images.
    """
    folders_neg = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            folders_neg.append(os.path.join(root, folder))
            if len(folders_neg) >= 1000:
                break
    num_data_target = len(folders_neg)
    negative_test = np.zeros([num_data_target, 3, 100, 100])  # original
    # negative_test = np.zeros([num_data_target, 100, 100, 3])
    for var in range(len(folders_neg)):
        # g_name = get_pkg_data_filename(folders_neg[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(folders_neg[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(folders_neg[var]+'/i_WCSClipped.fits')

        g_name = get_pkg_data_filename(folders_neg[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(folders_neg[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(folders_neg[var] + '/i_norm.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        negative_test[var] = [g, r, i]

    print("GOT NEGATIVE TESTING DATA")
    return negative_test


def loadImage(positive_array, negative_array):
    """
    This loads the positive and negative arrays, and makes an image dataset
    numpy array that is made through adding the images of the positive and
    negative arrays. This also makes a label dataset, by adding the appropriate
    labels for the positive and negative arrays.

    Args:
        positive_array (numpy array):    This is the positively simulated array of gravitational lenses.
        negative_array (numpy array):    This is the negative array of from DES.
    Returns:
        image_train (numpy array):      This is the numpy array of the positive and negative arrays added
                                        together to make a single array.
        image_labels (numpy array):     This is the numpy array of the labels for the positive and negative
                                        arrays added together to make a single array.
    """

    image_train = []
    image_labels = []

    for num in range(0, len(positive_array)):
        image_train.append(positive_array[num])
        # label_pos = 'Gravitational Lensing'
        label_pos = 1  # assign 1 for gravitational lensing
        image_labels.append(label_pos)

    for num in range(0, len(negative_array)):
        image_train.append(negative_array[num])
        # label_neg = 'No Gravitational Lensing'
        label_neg = 0  # assign 0  for non gravitational lenses
        image_labels.append(label_neg)

    print("LOADED POSITIVE AND NEGATIVE")
    return np.array(image_train), np.array(image_labels)


def getUnseenNegative(num, base_dir='UnseenData'):
    """
    This gets the unseen g, r, and i unknown/negative images, according to the number specified.
    This is so that the correct number of unknown images, is retrieved, according to the
    unseen known lenses.

    Args:
        num (integer):              This is the number specified, which equates to how many unseen known
                                    gravitational lenses.
                                    This indicates how many unseen negative images is to be retrieved.
        base_dir (string):          This is the root directory file path of where the unknown lenses are situated in.
    Returns:
        data_unknown (numpy array):  This is the numpy array of the unknown dataset.
    """
    path_unknown = '%s' % base_dir
    if num == 47:
        path_unknown = '%s/Negative47' % path_unknown
    elif num == 84:
        path_unknown = '%s/Negative84' % path_unknown
    elif num == 131:
        path_unknown = '%s/Negative131' % path_unknown

    folders_unknown = []
    for root, dirs, files in os.walk(path_unknown):
        for folder in dirs:
            folders_unknown.append(os.path.join(root, folder))

    num_data_target = len(folders_unknown)
    data_unknown = np.zeros([num_data_target, 3, 100, 100])

    for var in range(len(folders_unknown)):
        # gName = get_pkg_data_filename(folders_unknown[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(folders_unknown[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(folders_unknown[var]+'/i_WCSClipped.fits')

        gName = get_pkg_data_filename(folders_unknown[var] + '/g_norm.fits')
        rName = get_pkg_data_filename(folders_unknown[var] + '/r_norm.fits')
        iName = get_pkg_data_filename(folders_unknown[var] + '/i_norm.fits')

        g = fits.open(gName)[0].data[0:100, 0:100]
        r = fits.open(rName)[0].data[0:100, 0:100]
        i = fits.open(iName)[0].data[0:100, 0:100]

        data_unknown[var] = [g, r, i]
    return data_unknown


def getTestSet():
    positive_test = getPositiveSimulatedTest()
    negative_test = getNegativeDESTest()
    images_test, labels_test = loadImage(positive_test, negative_test)
    return images_test, labels_test


def getUnseenDES2017(base_dir='UnseenData/Known47'):
    """
    This is used to get g, r, and i images of the DES2017 array, which contains 47 unseen known lenses.
    Args:
        base_dir (string):          This is the root directory of the DES2017 folder.
    Returns:
        data_known_des (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    folders_known_des2017 = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            folders_known_des2017.append(os.path.join(root, folder))

    num_data_target = len(folders_known_des2017)
    data_known_des = np.zeros([num_data_target, 3, 100, 100])

    for var in range(len(folders_known_des2017)):
        # g_name = get_pkg_data_filename(folders_known_des2017[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(folders_known_des2017[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(folders_known_des2017[var]+'/i_WCSClipped.fits')

        g_name = get_pkg_data_filename(folders_known_des2017[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(folders_known_des2017[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(folders_known_des2017[var] + '/i_norm.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        data_known_des[var] = [g, r, i]
    return data_known_des


def getUnseenJacobs(base_dir='UnseenData/Known84'):
    """
    This is used to get g, r, and i images of the known Jacobs dataset, which contains 84 unseen known lenses.

    Args:
        base_dir (string):          This is the root directory of the DES2017 folder.
    Returns:
        dataKnownDES (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    folders_known_jacobs = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            folders_known_jacobs.append(os.path.join(root, folder))
    num_data_target = len(folders_known_jacobs)
    data_known_jacobs = np.zeros([num_data_target, 3, 100, 100])

    for var in range(len(folders_known_jacobs)):
        # g_name = get_pkg_data_filename(folders_known_jacobs[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(folders_known_jacobs[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(folders_known_jacobs[var]+'/i_WCSClipped.fits')

        g_name = get_pkg_data_filename(folders_known_jacobs[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(folders_known_jacobs[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(folders_known_jacobs[var] + '/i_norm.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        data_known_jacobs[var] = [g, r, i]
    return data_known_jacobs


def testUnseenDES2017(model, neural_network, n_splits):  # from last time everything worked

    # def testUnseenDES2017(model, neural_network, n_splits, input_shape):
    """
    This tests the unseen DES2017 images and unknown 47 images, to get the accuracy rate
    of these unseen images that aren't used in training.

    Returns:
        known_des2017_array (numpy array):    This is the numpy array of the known DES2017 images.
        accuracy_score_47 (float):           This is the accuracy score of the 47 unseen unknown images and of the
                                            47 images from DES2017, being tested on the already learnt set.
        k_fold_accuracy_47(float):            This is the accuracy score of the 47 unseen unknown images and of the 47
                                            images from DES2017 after k fold cross validation, being
                                            tested on the already learnt set.

    """

    known_des2017_array = getUnseenDES2017()

    num = 47
    unknown_array = getUnseenNegative(num)

    image_test, labels_test = loadImage(known_des2017_array, unknown_array)
    print("Image test shape: ", str(image_test.shape))
    image_test = image_test.reshape(image_test.shape[0], input_shape[0], input_shape[1], input_shape[2])

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_test)

    y_pred = model.predict(image_test)

    y_test_index = np.round(y_pred)
    ones = np.count_nonzero(y_test_index == 1)
    zeroes = np.count_nonzero(y_test_index == 0)

    print("Ones: %s / 47" % ones)
    print("Zeroes: %s / 47" % zeroes)

    # Get Accuracy Score tests DES2017 on the mlpclassifier:
    _, acc = model.evaluate(image_test, y_image_labels, verbose=0)
    accuracy_score_47 = acc * 100
    print("Accuracy Score_47: " + str(accuracy_score_47))

    # get the k fold accuracy after k fold cross validation
    scores = cross_val_score(neural_network, image_test, y_image_labels, scoring='accuracy', cv=n_splits)
    scores_mean = scores.mean() * 100
    print("kFold47 Scores Mean: " + str(scores_mean))
    k_fold_std_47 = scores.std()
    print("kFold47 Scores Std: " + str(k_fold_std_47))
    k_fold_accuracy_47 = scores_mean

    return known_des2017_array, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47


def testUnseenJacobs(model, neural_network, n_splits, input_shape):
    """
    This tests the unseen Jacobs images and unknown 84 images, to get the accuracy rate
    of these unseen images that aren't used in training.

    Returns:
        known_jacobs_array (numpy array):     This is the numpy array of the known Jacobs images.
        accuracy_score_84 (float):           This is the accuracy score of the 84 unseen unknown images and of the
                                            84 images from Jacobs, being tested on the already learnt set.
        k_fold_accuracy_84(float):            This is the accuracy score of the 84 unseen unknown images and of the 84
                                            images from Jacobs after k fold cross validation, being
                                            tested on the already learnt set.

    """

    known_jacobs_array = getUnseenJacobs()

    num = 84
    unknown_array = getUnseenNegative(num)

    image_jacobs_test, labels_jacobs_test = loadImage(known_jacobs_array, unknown_array)
    print("Image Jacobs test shape: ", str(image_jacobs_test.shape))
    image_jacobs_test = image_jacobs_test.reshape(image_jacobs_test.shape[0], input_shape[0], input_shape[1],
                                                  input_shape[2])

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_jacobs_test)

    y_pred = model.predict(image_jacobs_test)

    y_test_index = np.round(y_pred)
    ones = np.count_nonzero(y_test_index == 1)
    zeroes = np.count_nonzero(y_test_index == 0)

    print("Ones: %s / 84" % ones)
    print("Zeroes: %s / 84" % zeroes)

    # Get Accuracy Score tests Jacobs on the mlp classifier:
    _, acc = model.evaluate(image_jacobs_test, y_image_labels, verbose=0)
    accuracy_score_84 = acc * 100
    print("Accuracy Score_84: " + str(accuracy_score_84))

    # get the k fold accuracy after k fold cross validation
    scores = cross_val_score(neural_network, image_jacobs_test, y_image_labels, scoring='accuracy', cv=n_splits)
    scores_mean = scores.mean() * 100
    print("kFold84 Scores Mean: " + str(scores_mean))
    k_fold_std_84 = scores.std()
    print("kFold84 Scores Std: " + str(k_fold_std_84))
    k_fold_accuracy_84 = scores_mean
    return known_jacobs_array, accuracy_score_84, k_fold_accuracy_84, k_fold_std_84


def testUnseenDES2017AndJacobs(known_des2017_array, known_jacobs_array, model, neural_network, n_splits, input_shape):
    """
    This tests the unseen DES2017 and Jacobs images together with the unknown 131 images, to get the accuracy rate
    of these unseen images that aren't used in training.

    Args:
        known_des2017_array (numpy array):    This is the dataset of the unseen known DES2017 images.
        known_jacobs_array (numpy array):     This is the dataset of the unseen known Jacobs images.
    Returns:
        accuracy_score_131 (float):          This is the accuracy score of the 131 unseen unknown images and of the
                                            131 images from DES2017 and Jacobs, being tested on the already learnt set.
        k_fold_accuracy_131(float):           This is the accuracy score of the 131 unseen unknown images and of the 131
                                            images from DES2017 and Jacobs after k fold cross validation, being
                                            tested on the already learnt set.
    """

    all_known_array = np.vstack((known_des2017_array, known_jacobs_array))

    num = 131
    unknown_array = getUnseenNegative(num)

    image_known_test, labels_known_test = loadImage(all_known_array, unknown_array)
    # print("Image known test shape: " + str(image_known_test))
    image_known_test = image_known_test.reshape(image_known_test.shape[0], input_shape[0], input_shape[1],
                                                input_shape[2])

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_known_test)

    y_pred = model.predict(image_known_test)

    y_test_index = np.round(y_pred)
    ones = np.count_nonzero(y_test_index == 1)
    zeroes = np.count_nonzero(y_test_index == 0)

    print("Ones: %s / 131" % ones)
    print("Zeroes: %s / 131" % zeroes)

    # Get Accuracy Score tests DES2017 on the mlp classifier:
    _, acc = model.evaluate(image_known_test, y_image_labels, verbose=0)
    accuracy_score_131 = acc * 100
    print("Accuracy Score _131: " + str(accuracy_score_131))

    # get the k fold accuracy after k fold cross validation
    scores = cross_val_score(neural_network, image_known_test, y_image_labels, scoring='accuracy', cv=n_splits)
    scores_mean = scores.mean() * 100
    print("kFold131 Scores Mean: " + str(scores_mean))
    k_fold_std_131 = scores.std()
    print("kFold131 Scores Std: " + str(k_fold_std_131))
    k_fold_accuracy_131 = scores_mean
    return accuracy_score_131, k_fold_accuracy_131, k_fold_std_131


def makeTrainTest(positive_train, negative_train, positive_test, negative_test):
    """
    This makes the training and testing data sets that are to be made. This creates
    a training image data set with the positive and negative images together. This
    also creates a training label data set with the positive and negative images together.

    Args:
        positive_train (numpy array):    This is the positively simulated dataset images.
        negative_train (numpy array):    This is the negative DES images.
    Returns:
        x_train (numpy array):          This is the array of the training set of the training images,
                                        which is 80% of the image training set.
        x_test (numpy array):           This is the array of the testing set of the training images, which
                                        is the 20% of the training images.
        y_train (numpy array):          This is the array of the labels of the training labels, which is 80%
                                        of the training labels.
        y_test (numpy array):           This is the array of the labels of the testing labels, which is 20%
                                        of the training labels.
        train_percent (float):          This is the percentage of data used for training (1- testPercent).
        testPercent (float):           This is the percentage of date used for testing.
        image_train_std (float):         This is the standard deviation of the entire training set, all 20000 images.
        image_train_mean (float):        This is the mean of the entire training set, all 20000 images.
        image_train_shape (list):        This is the shape of the entire training set, all 20000 images.
        labels_train_shape (list):       This is the shape of the entire training sets' labels, all 20000 labels.
        x_train_shape (list):            This is the shape of the training set of the images.
        x_test_shape (list):             This is the shape of the testing set of the images.
        y_train_shape (list):            This is the shape of the training set of the labels.
        y_test_shape (list):             This is the shape of the testing set of the labels.
    """

    image_train, labels_train = loadImage(positive_train, negative_train)
    image_test, labels_test = loadImage(positive_test, negative_test)

    print("Labels Train = 1: " + str(np.count_nonzero(labels_train == 1)))
    print("Labels Train = 0: " + str(np.count_nonzero(labels_train == 0)))

    print("Labels Test = 1: " + str(np.count_nonzero(labels_test == 1)))
    print("Labels Test = 0: " + str(np.count_nonzero(labels_test == 0)))

    # getting parameters of training data
    image_train_std = image_train.std()
    image_train_mean = image_train.mean()
    image_train_shape = image_train.shape

    print("Train Std: " + str(image_train_std))
    print("Train Mean: " + str(image_train_mean))
    print("Train Shape: " + str(image_train_shape))

    labels_train_shape = labels_train.shape
    print("Train Labels Shape: " + str(labels_train_shape))

    image_test_std = image_test.std()
    image_test_mean = image_test.mean()
    image_test_shape = image_test.shape

    print("Test Std: " + str(image_test_std))
    print("Test Mean: " + str(image_test_mean))
    print("Test Shape: " + str(image_test_shape))
    print("Test Labels Shape: " + str(labels_test.shape))

    # Encoding y now
    encoder = LabelEncoder()
    y_labels_train = encoder.fit_transform(labels_train)
    y_labels_test = encoder.fit_transform(labels_test)

    test_percent = 0

    x_train = shuffle(image_train)
    y_train = shuffle(y_labels_train)
    x_test = shuffle(image_test)
    y_test = shuffle(y_labels_test)

    x_train_shape = x_train.shape
    x_test_shape = x_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape

    print("x_train type: " + str(type(x_train)))
    print("x_train shape: " + str(x_train_shape))
    print("y_train shape: " + str(y_train_shape))
    print("x_test shape: " + str(x_test_shape))
    print("y_test shape: " + str(y_test_shape))

    train_percent = (1 - test_percent)
    print("MADE TRAIN TEST SET")
    return (x_train, x_test, y_train, y_test, train_percent, test_percent, image_train_std, image_train_mean,
            image_train_shape, labels_train_shape, x_train_shape, x_test_shape, y_train_shape, y_test_shape)


def makeKerasModel():
    # mlp classifier without cnn
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(3, 100, 100)))  # change this to have a 2d shape
    # model.add(Dense(100, activation='relu', input_shape=[100, 100, 3]))  # change this to have a 2d shape
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # THE KERAS WITHOUT ES PNG IMAGE, HAS SIGMOID
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def makeModelFromTutorial(input_shape=(3, 100, 100)):
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))  # antes era 0.25
    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))  # antes era 0.25
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))  # antes era 0.25
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))
    model.summary()

    # Compiling the CNN
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def makeKerasCNNModel(input_shape=(3, 100, 100)):
    # https://medium.com/@randerson112358/classify-images-using-convolutional-neural-networks-python-a89cecc8c679
    # https: // keras.io / guides / sequential_model /

    model = Sequential()
    #
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))  # antes era 0.25
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))  # antes era 0.25
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))

    # model.add(Flatten()) # makes multiple arrays into a single vector
    model.add(Dense(1))
    print("MODEL 1 SUMMARY" + str(model.summary()))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def useKerasModel(x_train, x_test, y_train, y_test):
    # def useKerasModel(x_train, x_test, y_train, y_test, input_shape):
    # Changing shape of x_train and y_train from num, channels, height, width (num, 3, 100, 100) to num, widht,
    # # height, channels (num, 100, 100, 3)
    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[1])
    # x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[1])

    # x_train_shape = x_train.shape
    # x_test_shape = x_test.shape
    # y_train_shape = y_train.shape
    # y_test_shape = y_test.shape

    # print("x_train Reshaped: " + str(x_train_shape))
    # print("y_train Reshaped: " + str(y_train_shape))
    # print("x_test shape Reshaped: " + str(x_test_shape))
    # print("y_test shape Reshaped: " + str(y_test_shape))

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=3)
    model = makeKerasCNNModel()
    # model = makeKerasModel()
    # model = makeKerasCNNModel(input_shape)
    # model = makeModelFromTutorial(input_shape)
    seq_model = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=225,  # power of 2 , 100, 121, etc... 225
        validation_data=(x_test, y_test),
        callbacks=[es])

    description = str(model)

    y_pred = model.predict(x_test)
    print("y_pred: " + str(y_pred))
    print("y_pred shape: " + str(y_pred.shape))
    print("y_pred(type): " + str(type(y_pred)))
    y_test_index = np.round(y_pred)
    print("y_test_index: " + str(y_test_index))
    ones = np.count_nonzero(y_test_index == 1)
    zeroes = np.count_nonzero(y_test_index == 0)

    print("Ones: %s / 1000" % ones)
    print("Zeroes: %s / 1000" % zeroes)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: " + str(loss) + " / Test accuracy: " + str(accuracy))
    accuracy_score = accuracy * 100
    print("Accuracy Score: " + str(accuracy_score))
    model.save_weights('kerasModel.h5')

    return seq_model, model, accuracy_score


def plotModel(seq_model):
    # plot training vs validation loss.
    History()
    train_loss = seq_model.history['loss']
    val_loss = seq_model.history['val_loss']
    train_accuracy = seq_model.history['acc']
    val_accuracy = seq_model.history['val_acc']

    # epochs = range(1,50)
    fig1 = plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig1.savefig('../Results/TrainingvsValidationLoss_Keras.png')

    fig2 = plt.figure()
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig2.savefig('../Results/TrainingvsValidationAccuracy_Keras.png')

    print("USED KERAS MODEL")


# def getKerasKFold(x_train, x_test, y_train, y_test, input_shape):
def getKerasKFold(x_train, x_test, y_train, y_test):
    # Stratified K fold Cross Validation
    # https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

    print("DOING KERAS K FOLD")
    neural_network = KerasClassifier(build_fn=makeKerasCNNModel,
                                     epochs=30,
                                     batch_size=200,
                                     verbose=0)
    n_splits = 10
    random_state = 0
    print("DONE 2")
    scores = cross_val_score(neural_network, x_test, y_test, scoring='accuracy', cv=n_splits)
    print("DONE 3")
    scores_mean = scores.mean() * 100
    print("kFold Scores Mean: " + str(scores_mean))
    k_fold_std = scores.std()
    print("kFold Scores Std: " + str(k_fold_std))
    print("DONE 4")

    fig3 = plt.figure()
    plt.plot(scores, label='Scores')
    plt.legend()
    fig3.savefig('../Results/KerasKFold_Scores.png')
    return n_splits, random_state, scores_mean, k_fold_std, neural_network
    # _____________________________________________________________________________________________________________________________


def displayActivation(activations):
    for num in range(len(activations)):
        activation = activations[num]
        if activation.ndim >= 3:
            plt.figure()
            plt.matshow(activation[0, :, :, 4])
            plt.show()
    first_layer_activation = activations[0]
    # print(first_layer_activation.shape)
    # plt.figure()
    # plt.title("First Layer Activation")
    # plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    # plt.show()


# def visualizeKeras(model, input_shape):
def visualizeKeras(model):
    #     # https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
    #     # https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras
    #     # -260b36d60d0
    #
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input,
                             outputs=layer_outputs)
    img_tensor = getPositiveSimulatedTest()[0]
    plt.figure()
    plt.title("Original image")
    plt.imshow(img_tensor[0])
    plt.show()

    input_shape = (3, 100, 100)
    img_tensor = img_tensor.reshape(1, input_shape[0], input_shape[1], input_shape[2])
    activations = activation_model.predict(img_tensor)
    displayActivation(activations)

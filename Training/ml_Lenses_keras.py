"""
This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the gravitational lenses.
"""
# IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import makeExcelTable


# from keras.models import Model

# TO EXTRACT FEATURES FROM CNN
# https://datascience.stackexchange.com/questions/17513/extracting-features-using-tensorflow-cnn

# FUNCTIONS
def getPositiveSimulated(base_dir='PositiveWithDESSky'):
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

    # number of Positive DataPoints
    num_data_target = len(folders)

    data_pos = np.zeros([num_data_target, 3, 100, 100])

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
        # just to run, and use less things
        # if counter > 1500:
        #     break
    return data_pos


def getNegativeDES(base_dir='DES/DES_Processed'):
    """
    This gets the g, r, and i  10 000 negative images from the 
    DES/DES_Processedfolder, as well as returning the 
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
    num_data_target = len(folders_neg)
    data_neg = np.zeros([num_data_target, 3, 100, 100])

    for var in range(len(folders_neg)):
        # if var > 1500:
        # break
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
        # if var > 1500:
        #     break
    return data_neg


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

    positive_data = []
    negative_data = []
    positive_label = []
    negative_label = []
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

    return np.array(image_train), np.array(image_labels)


def getDES2017(base_dir='KnownLenses/DES2017/'):
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


def getJacobs(base_dir='KnownLenses/Jacobs_KnownLenses/'):
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


def getPositiveSimulated1000(base_dir='NewLenses/PositiveWithDESSky'):
    folders = {}
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value

    # number of Positive DataPoints
    num_data_target = len(folders)

    data_pos_1000 = np.zeros([num_data_target, 3, 100, 100])

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

        data_pos_1000[counter] = [g, r, i]
        counter += 1
        # just to run, and use less things
        # if counter > 1500:
        #     break
    return data_pos_1000


def getUnknown(num, base_dir='KnownLenses'):
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

    if num == 47:
        path_unknown = '%s/Unknown_Processed_47' % (base_dir)
    elif num == 84:
        path_unknown = '%s/Unknown_Processed_84' % (base_dir)
    elif num == 131:
        path_unknown = '%s/Unknown_Processed_131' % (base_dir)
    elif num == 1000:
        path_unknown = '%s/Unknown_Processed_1000' % (base_dir)

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
    data_des_2017 = getDES2017()
    negative_47 = getUnknown(47)

    # data_jacobs = getJacobs()
    # data_known_131 = np.vstack((data_des_2017,data_jacobs))
    # negative_131 = getUnknown(131)
    #
    images, labels = loadImage(data_des_2017, negative_47)

    # data_pos_1000 = getPositiveSimulated1000()
    # unknown_1000 = getUnknown(1000)
    # images, labels = loadImage(data_pos_1000, unknown_1000)

    return images, labels


def testDES2017(model, neural_network, n_splits):
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

    known_des2017_array = getDES2017()

    num = 47
    unknown_array = getUnknown(num)

    image_test, labels_test = loadImage(known_des2017_array, unknown_array)
    x_image_test = image_test.reshape(image_test.shape[0], image_test.shape[1] * image_test.shape[2] * image_test.shape[
        3])  # batchsize, height*width*3channels

    # print('x_image_test length: '+str(len(x_image_test)))
    # print('image_test shape: '+str(image_test.shape()))
    # print('labels_test Shape: '+str(labels_test.shape()))

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_test)
    # print('y_image_labels Shape: '+str(y_image_labels.shape()))

    # Get Accuracy Score tests DES2017 on the mlpclassifier:
    y_pred = model.predict(image_test)
    # print('y_pred Shape: '+str(y_pred.shape()))
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


def testJacobs(model, neural_network, n_splits):
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

    known_jacobs_array = getJacobs()

    num = 84
    unknown_array = getUnknown(num)

    image_jacobs_test, labels_jacobs_test = loadImage(known_jacobs_array, unknown_array)
    x_image_test = image_jacobs_test.reshape(image_jacobs_test.shape[0],
                                             image_jacobs_test.shape[1] * image_jacobs_test.shape[2] *
                                             image_jacobs_test.shape[
                                                 3])  # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_jacobs_test)

    # Get Accuracy Score tests Jacobs on the mlp classifier:
    y_pred = model.predict(image_jacobs_test)
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


def testDES2017AndJacobs(known_des2017_array, known_jacobs_array, model, neural_network, n_splits):
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
    unknown_array = getUnknown(num)

    image_known_test, labels_known_test = loadImage(all_known_array, unknown_array)
    x_image_test = image_known_test.reshape(image_known_test.shape[0],
                                            image_known_test.shape[1] * image_known_test.shape[2] *
                                            image_known_test.shape[
                                                3])  # batch size, height*width*3channels

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_known_test)

    # Get Accuracy Score tests DES2017 on the mlp classifier:
    y_pred = model.predict(image_known_test)
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


def makeTrainTest(positive_array, negative_array):
    """
    This makes the training and testing data sets that are to be made. This creates 
    a training image data set with the positive and negative images together. This 
    also creates a training label data set with the positive and negative images together.

    Args:  
        positive_array (numpy array):    This is the positively simulated dataset images.
        negative_array (numpy array):    This is the negative DES images.
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
        image_labels_shape (list):       This is the shape of the entire training sets' labels, all 20000 labels.
        x_train_shape (list):            This is the shape of the training set of the images.
        x_test_shape (list):             This is the shape of the testing set of the images.
        y_train_shape (list):            This is the shape of the training set of the labels.
        y_test_shape (list):             This is the shape of the testing set of the labels.
    """

    image_train, image_labels = loadImage(positive_array, negative_array)
    image_train_std = image_train.std()
    image_train_mean = image_train.mean()
    image_train_shape = image_train.shape

    # X = image_train.reshape(image_train.shape[0], image_train.shape[1]*image_train.shape[2]*image_train.shape[3])
    # print("X shape: " + str(X.shape))

    image_labels_shape = image_labels.shape

    image_new_test, labels_new_test = getTestSet()

    # Encoding y now
    encoder = LabelEncoder()
    y_labels_train = encoder.fit_transform(image_labels)
    y_labels_new = encoder.fit_transform(labels_new_test)
    # print("y shape: " +str(y.shape))

    # Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
    test_percent = 0
    # x_train, y_train,  = train_test_split(image_train, y_labels_train, shuffle=True, test_size=None, random_state=1)
    # x_test, y_test = train_test_split(, y_labels_47, shuffle=True, train_size=None, random_state=1)

    x_train = shuffle(image_train)
    y_train = shuffle(y_labels_train)
    x_test = shuffle(image_new_test)
    y_test = shuffle(y_labels_new)

    x_train_shape = x_train.shape
    x_test_shape = x_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape

    print("x_train: " + str(x_train_shape))
    print("y_train: " + str(y_train.shape))
    print("x_test shape: " + str(x_test_shape))
    print("y_test shape: " + str(y_train.shape))

    train_percent = (1 - test_percent)

    return (x_train, x_test, y_train, y_test, train_percent, test_percent, image_train_std, image_train_mean,
            image_train_shape, image_labels_shape, x_train_shape, x_test_shape, y_train_shape, y_test_shape)


def makeKerasModel():
    # mlp classifier without cnn
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(3, 100, 100)))  # change this to have a 2d shape
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # THE KERAS WITHOUT ES PNG IMAGE, HAS SIGMOID
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def makeKerasCNNModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(3, 100, 100)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def useKerasModel(positive_array, negative_array):

    x_train, x_test, y_train, y_test, train_percent, test_percent, image_train_std, image_train_mean, \
    image_train_shape, image_labels_shape, x_train_shape, x_test_shape, y_train_shape, y_test_shape = makeTrainTest(
        positive_array, negative_array)

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=3)
    model = makeKerasModel()
    seq_model = model.fit(x_train, y_train, epochs=30, batch_size=200, validation_data=(x_test, y_test), callbacks=[es])
    description = str(model)

    # Accuracy Testing
    y_pred = model.predict(x_test)
    y_test_index = np.round(y_pred)
    Ones = np.count_nonzero(y_test_index == 1)
    Zeroes = np.count_nonzero(y_test_index == 0)

    print("Ones: %s / 1000" % (Ones))
    print("Zeroes: %s / 1000" % (Zeroes))

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    accuracy_score = acc * 100.0
    print("Accuracy Score: " + str(accuracy_score))
    # plot training vs validation loss. 
    History()
    train_loss = seq_model.history['loss']
    val_loss = seq_model.history['val_loss']
    train_accuracy = seq_model.history['accuracy']
    val_accuracy = seq_model.history['val_accuracy']

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

    # saving model weights in keras.
    model.save_weights('kerasModel.h5')

    return (model, x_train, x_test, y_train, y_test, description, train_percent, test_percent, image_train_std,
            image_train_mean, image_train_shape, image_labels_shape, x_train_shape, x_test_shape, y_train_shape,
            y_test_shape, accuracy_score)


def getKerasKFold(x_train, x_test, y_train, y_test):
    # Stratified K fold Cross Validation
    # https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

    neural_network = KerasClassifier(build_fn=makeKerasModel,
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


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1


def visualizeKeras(model, x_train):
    # https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras

    # topLayer= model.layers[0]
    # plt.show(topLayer.get_weights())

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x_train)  # 20000 images and 100X100 dimensions and 3 channels
    plt.imshow(x_train[10][:, :, 0])
    display_activation(activations, 100, 100, 3)


# _________________________________________________________________________________________________________________________
# MAIN

positive_array = getPositiveSimulated()
negative_array = getNegativeDES()

model, x_train, x_test, yTrain, y_test, description, train_percent, test_percent, image_train_std, image_train_mean, \
image_train_shape, image_labels_shape, x_train_shape, x_test_shape, y_train_shape, y_test_shape, accuracy_score = \
    useKerasModel(positive_array, negative_array)
print("DONE 1")
# visualizeKeras(model, x_train)
n_splits, random_state, k_fold_accuracy, k_fold_std, neural_network = getKerasKFold(x_train, x_test, yTrain, y_test)

# calculating the amount of things accuractely identified
# looking at Known131
# 1 = gravitational lens
# 0 = negative lens

# #______________________________________________________________________________________________________________________
known_des_2017, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47 = testDES2017(model, neural_network, n_splits)
known_jacobs, accuracy_score_84, k_fold_accuracy_84, k_fold_std_84 = testJacobs(model, neural_network, n_splits)
accuracy_score_131, k_fold_accuracy_131, k_fold_std_131 = testDES2017AndJacobs(known_des_2017, known_jacobs, model,
                                                                               neural_network, n_splits)

# write to ml_Lenses_results.xlsx
# makeExcelTable.makeInitialTable()
element_list = makeExcelTable.getElementList(description, image_train_std, image_train_mean, image_train_shape,
                                             image_labels_shape, train_percent, test_percent, x_train_shape,
                                             x_test_shape,
                                             y_train_shape, y_test_shape, n_splits, random_state, accuracy_score,
                                             k_fold_accuracy,
                                             k_fold_std, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47,
                                             accuracy_score_84,
                                             k_fold_accuracy_84, k_fold_std_84, accuracy_score_131, k_fold_accuracy_131,
                                             k_fold_std_131)
file_name = '../Results/ml_Lenses_results.csv'
makeExcelTable.appendRowAsList(file_name, element_list)

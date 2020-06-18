"""
This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the gravitational lenses.
"""
# IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import makeExcelTable
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.utils import shuffle


# FUNCTIONS
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

    # number of Positive DataPoints
    num_data_targets = len(folders)

    data_pos = np.zeros([num_data_targets, 3, 100, 100])

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
    num_data_targets = len(folders_neg)
    data_neg = np.zeros([num_data_targets, 3, 100, 100])

    for var in range(len(folders_neg)):
        # g_name = get_pkg_data_filename(folders_neg[var] + '/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(folders_neg[var] + '/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(folders_neg[var] + '/i_WCSClipped.fits')    

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
    print("DONE 2")
    return data_neg


def getPositiveSimulatedTest(base_dir='Testing/Positive'):
    folders = {}
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value

    # number of Positive DataPoints
    num_data_target = len(folders)

    positive_test = np.zeros([num_data_target, 3, 100, 100])

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
        # just to run, and use less things
        # if counter > 1500:
        #     break
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
    num_data_target = len(folders_neg)
    negative_test = np.zeros([num_data_target, 3, 100, 100])

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

        negative_test[var] = [g, r, i]
        # just to run, and use less things
        # if var >= 1000:
        #     break
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
        label_neg = 0  # assign 0 for no lensing
        image_labels.append(label_neg)
    print("DONE 3")
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


def getDES2017(base_dir='KnownLenses/DES2017/'):
    """
    This is used to get g, r, and i images of the DES2017 array, which contains 47 unseen known lenses.
    Args:
        base_dir (string):          This is the root directory of the DES2017 folder. 
    Returns:
        data_known_des (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    known_des2017 = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            known_des2017.append(os.path.join(root, folder))

    num_data_targets = len(known_des2017)
    data_known_des = np.zeros([num_data_targets, 3, 100, 100])

    for var in range(len(known_des2017)):
        # g_name = get_pkg_data_filename(known_des2017[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(known_des2017[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(known_des2017[var]+'/i_WCSClipped.fits')    

        g_name = get_pkg_data_filename(known_des2017[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(known_des2017[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(known_des2017[var] + '/i_norm.fits')

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

    known_jacobs = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            known_jacobs.append(os.path.join(root, folder))
    num_data_targets = len(known_jacobs)
    data_known_jacobs = np.zeros([num_data_targets, 3, 100, 100])

    for var in range(len(known_jacobs)):
        # g_name = get_pkg_data_filename(known_jacobs[var] + '/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(known_jacobs[var] + '/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(known_jacobs[var] + '/i_WCSClipped.fits')

        g_name = get_pkg_data_filename(known_jacobs[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(known_jacobs[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(known_jacobs[var] + '/i_norm.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        data_known_jacobs[var] = [g, r, i]
    return data_known_jacobs


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
        path_unknown = '%s/Unknown_Processed_47' % base_dir
    elif num == 84:
        path_unknown = '%s/Unknown_Processed_84' % base_dir
    elif num == 131:
        path_unknown = '%s/Unknown_Processed_131' % base_dir
    elif num == 1000:
        path_unknown = '%s/Unknown_Processed_1000' % base_dir

    folders_unknown = []
    for root, dirs, files in os.walk(path_unknown):
        for folder in dirs:
            folders_unknown.append(os.path.join(root, folder))

    num_data_targets = len(folders_unknown)
    data_unknown = np.zeros([num_data_targets, 3, 100, 100])

    for var in range(len(folders_unknown)):
        # g_name = get_pkg_data_filename(folders_unknown[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(folders_unknown[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(folders_unknown[var]+'/i_WCSClipped.fits')

        g_name = get_pkg_data_filename(folders_unknown[var] + '/g_norm.fits')
        r_name = get_pkg_data_filename(folders_unknown[var] + '/r_norm.fits')
        i_name = get_pkg_data_filename(folders_unknown[var] + '/i_norm.fits')

        g = fits.open(g_name)[0].data[0:100, 0:100]
        r = fits.open(r_name)[0].data[0:100, 0:100]
        i = fits.open(i_name)[0].data[0:100, 0:100]

        data_unknown[var] = [g, r, i]

    return data_unknown

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

def getTestSet():
    data_des_2017 = getDES2017()
    negative_47 = getUnknown(47)

    # data_jacobs = getJacobs()
    # data_known_131 = np.vstack((data_des_2017,data_jacobs))
    # negative_131 = getUnknown(131)

    images, labels = loadImage(data_des_2017, negative_47)

    # data_pos_1000 = getPositiveSimulated1000()
    # unknown_1000 = getUnknown(1000)
    # images, labels = loadImage(data_pos_1000, unknown_1000)

    return images, labels


def testDES2017():
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
        3])  # batch size, height*width*3channels

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_test)

    y_pred = clf_images.predict(x_image_test)
    accuracy_score_47 = (accuracy_score(y_image_labels, y_pred)) * 100.0

    results = model_selection.cross_val_score(clf_images, x_image_test, y_image_labels, cv=k_fold)
    k_fold_accuracy_47 = (results.mean()) * 100.0
    k_fold_std_47 = results.std()

    return known_des2017_array, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47


def testJacobs():
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

    y_pred = clf_images.predict(x_image_test)
    accuracy_score_84 = (accuracy_score(y_image_labels, y_pred)) * 100

    results = model_selection.cross_val_score(clf_images, x_image_test, y_image_labels, cv=k_fold)
    k_fold_accuracy_84 = (results.mean()) * 100
    k_fold_std_84 = results.std()

    return known_jacobs_array, accuracy_score_84, k_fold_accuracy_84, k_fold_std_84


def testDES2017AndJacobs(known_des2017_array, known_jacobs_array):
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
                                                3])  # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_image_labels = encoder.fit_transform(labels_known_test)

    y_pred = clf_images.predict(x_image_test)

    accuracy_score_131 = (accuracy_score(y_image_labels, y_pred)) * 100

    results = model_selection.cross_val_score(clf_images, x_image_test, y_image_labels, cv=k_fold)
    k_fold_accuracy_131 = (results.mean()) * 100
    k_fold_std_131 = results.std()

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
        train_percent (float):          This is the percentage of data used for training (1- test_percent).
        test_percent (float):           This is the percentage of date used for testing.
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
    print("image_train shape: " + str(image_train_shape))
    image_labels_shape = image_labels.shape

    image_new_test, labels_new_test = getTestSet()

    # reshape x
    image_train_reshaped = image_train.reshape(image_train.shape[0], image_train.shape[1] * image_train.shape[2] *
                                               image_train.shape[3])  # batch size, height*width*3channels
    # print("x shape: " + str(x.shape))

    image_new_test_reshaped = image_new_test.reshape(image_new_test.shape[0], image_new_test.shape[1] *
                                                     image_new_test.shape[2] * image_new_test.shape[3])
    # Encoding y now
    encoder = LabelEncoder()
    y_labels_train = encoder.fit_transform(image_labels)
    y_labels_new = encoder.fit_transform(labels_new_test)

    # Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
    test_percent = 0.2
    # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=test_percent)

    x_train = shuffle(image_train_reshaped)
    y_train = shuffle(y_labels_train)
    x_test = shuffle(image_new_test_reshaped)
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


# _____________________________________________________________________________________________________________________________
# MAIN

positive_array = getPositiveSimulatedTrain()
negative_array = getNegativeDES()

x_train, x_test, y_train, yTest, trainPercent, test_percent, image_train_std, image_train_mean, image_train_shape, \
image_labels_shape, x_train_shape, x_test_shape, y_train_shape, y_test_shape = makeTrainTest(
    positive_array, negative_array)

# Trianing the data with MLPClassifier, from scikit learn
clf_images = MLPClassifier(activation='relu',
                           hidden_layer_sizes=(100, 100, 100),  # 3 layers of 100 neurons each
                           solver='adam',
                           verbose=True,
                           max_iter=100,
                           batch_size=200,
                           early_stopping=True)  # batch size = 200 default

description = str(clf_images)

# Getting training loss
clf_images.fit(x_train, y_train)
train_loss = clf_images.loss_curve_

# Accuracy Testing
y_pred = clf_images.predict(x_test)
model_accuracy_score = (accuracy_score(yTest, y_pred)) * 100

# Testing number of ones and zeroes
y_test_index = np.round(y_pred)
Ones = np.count_nonzero(y_test_index == 1)
Zeroes = np.count_nonzero(y_test_index == 0)
print("Ones: %s / 1000" % (Ones))
print("Zeroes: %s / 1000" % (Zeroes))

# Getting validation loss
clf_images.fit(x_test, yTest)
val_loss = clf_images.loss_curve_

epochs = range(1, 30)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../Results/TrainingvsValidationLoss_Sklearn.png')

# Cross Validation
n_splits = 10
random_state = 100
k_fold = model_selection.KFold(n_splits=n_splits, random_state=random_state)
score = model_selection.cross_val_score(clf_images, x_test, yTest, scoring='accuracy', cv=k_fold)
k_fold_accuracy = (score.mean()) * 100
k_fold_std = score.std()

fig4 = plt.figure()
plt.plot(score, label='Scores')
plt.legend()
fig4.savefig('../Results/SkLearnKFold_Scores.png')

# ______________________________________________________________________________________________________________________
known_des2017, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47 = testDES2017()
known_jacobs, accuracy_score_84, k_fold_accuracy_84, k_fold_std_84 = testJacobs()
accuracy_score_131, k_fold_accuracy_131, k_fold_std_131 = testDES2017AndJacobs(known_des2017, known_jacobs)

print("Accuracy Score: " + str(model_accuracy_score))
print("Accuracy Type: " + str(type(model_accuracy_score)))
print("K Fold Accuracy: "+str(k_fold_accuracy))
print("K Fold Std: " + str(k_fold_std))
print("Accuracy_47: " + str(accuracy_score_47))
print("K Fold Accuracy_47: " + str(k_fold_accuracy_47))
print("K Fold Std_47: " + str(k_fold_std_47))
print("Accuracy_84: " + str(accuracy_score_84))
print("K Fold Accuracy_84: " + str(k_fold_accuracy_84))
print("K Fold Std_84: " + str(k_fold_std_84))
print("Accuracy_131: " + str(accuracy_score_131))
print("K Fold Accuracy_131: " + str(k_fold_accuracy_131))
print("K Fold Std_131: " + str(k_fold_std_131))

# write to ml_Lenses_results.xlsx
# makeExcelTable.makeInitialTable()
elementList = makeExcelTable.getElementList(description, image_train_std, image_train_mean, image_train_shape,
                                            image_labels_shape, trainPercent, test_percent, x_train_shape, x_test_shape,
                                            y_train_shape, y_test_shape, n_splits, random_state, model_accuracy_score,
                                            k_fold_accuracy,
                                            k_fold_std, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47,
                                            accuracy_score_84, k_fold_accuracy_84, k_fold_std_84, accuracy_score_131,
                                            k_fold_accuracy_131, k_fold_std_131)
filename = '../Results/ml_Lenses_results.csv'
makeExcelTable.appendRowAsList(filename, elementList)

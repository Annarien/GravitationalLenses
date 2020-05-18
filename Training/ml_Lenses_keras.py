"""
This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the gravitational lenses.
"""
# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import makeExcelTable
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import History 
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_classification

# FUNCTIONS
def getPositiveSimulated(base_dir = 'PositiveWithDESSky'):
    """
    This gets the g, r, and i of the 10 000 positively simulated images from the 
    PositiveWithDESSky, as well as returning the positively simulate array.

    Args:
        base_dir (string):      This the root file path of the positively simulated images.  
    Returns:
        DataPos(numpy array):   This is the array of positively simulated images.
    """

    folders = {}
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value

    #number of Positive DataPoints
    nDT = len(folders)

    DataPos = np.zeros([nDT, 3, 100, 100])

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
        
        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]
        
        DataPos[counter] = [g, r, i] 
        counter += 1
        # just to run, and use less things
        # if counter > 1500:
        #     break
    return (DataPos)

def getNegativeDES(base_dir = 'DES/DES_Processed'):
    """
    This gets the g, r, and i  10 000 negative images from the 
    DES/DES_Processedfolder, as well as returning the 
    negative array,

    Args:
        base_dir (string):      This the root file path of the negative images.  
    Returns:
        DataNeg (numpy array):  This is the array of negative images.
    """
    foldersNeg = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersNeg.append(os.path.join(root, folder))
    nDT = len(foldersNeg)
    DataNeg = np.zeros([nDT,3,100,100])

    for var in range(len(foldersNeg)):

        # g_name = get_pkg_data_filename(foldersNeg[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(foldersNeg[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(foldersNeg[var]+'/i_WCSClipped.fits')    

        g_name = get_pkg_data_filename(foldersNeg[var]+'/g_norm.fits')
        r_name = get_pkg_data_filename(foldersNeg[var]+'/r_norm.fits')
        i_name = get_pkg_data_filename(foldersNeg[var]+'/i_norm.fits')    

        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]    
        
        DataNeg[var] = [g, r, i]
        # just to run, and use less things
        # if var > 1500:
        #     break
    return (DataNeg)

def loadImage(positiveArray, negativeArray):
    """
    This loads the positive and negative arrays, and makes an image dataset 
    numpy array that is made through adding the images of the positive and 
    negative arrays. This also makes a label dataset, by adding the appropriate
    labels for the positive and negative arrays.

    Args:
        positiveArray (numpy array):    This is the positively simulated array of gravitational lenses.
        negativeArray (numpy array):    This is the negative array of from DES. 
    Returns:
        image_train (numpy array):      This is the numpy array of the positive and negative arrays added
                                        together to make a single array.
        image_labels (numpy array):     This is the numpy array of the labels for the positive and negative 
                                        arrays added together to make a single array.
    """

    positiveData = []
    negativeData = []
    positiveLabel = []
    negativeLabel = []
    image_train = []
    image_labels = []

    for num in range(0,len(positiveArray)):
        image_train.append(positiveArray[num])
        # labelPos = 'Gravitational Lensing'
        labelPos = 1 # assign 1 for gravitational lensing
        image_labels.append(labelPos)
    
    for num in range(0,len(negativeArray)):
        image_train.append(negativeArray[num])
        # labelNeg = 'No Gravitational Lensing'
        labelNeg = 0 # assign 0  for non gravitational lenses 
        image_labels.append(labelNeg)

    return (np.array(image_train), np.array(image_labels))

def getDES2017(base_dir = 'KnownLenses/DES2017/'):
    """
    This is used to get g, r, and i images of the DES2017 array, which contains 47 unseen known lenses.
    Args:
        base_dir (string):          This is the root directory of the DES2017 folder. 
    Returns:
        DataKnownDES (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    foldersKnownDES2017 = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersKnownDES2017.append(os.path.join(root, folder))

    nDT = len(foldersKnownDES2017)
    DataKnownDES = np.zeros([nDT,3,100,100])

    for var in range(len(foldersKnownDES2017)):

        # g_name = get_pkg_data_filename(foldersKnownDES2017[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(foldersKnownDES2017[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(foldersKnownDES2017[var]+'/i_WCSClipped.fits')    

        g_name = get_pkg_data_filename(foldersKnownDES2017[var]+'/g_norm.fits')
        r_name = get_pkg_data_filename(foldersKnownDES2017[var]+'/r_norm.fits')
        i_name = get_pkg_data_filename(foldersKnownDES2017[var]+'/i_norm.fits')    
    
        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]    
        
        DataKnownDES[var] = [g, r, i]

    return (DataKnownDES)

def getJacobs(base_dir = 'KnownLenses/Jacobs_KnownLenses/'):
    """
    This is used to get g, r, and i images of the known Jacobs dataset, which contains 84 unseen known lenses.

    Args:
        base_dir (string):          This is the root directory of the DES2017 folder. 
    Returns:
        DataKnownDES (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    foldersKnownJacobs = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersKnownJacobs.append(os.path.join(root, folder))
    nDT = len(foldersKnownJacobs)
    DataKnownJacobs = np.zeros([nDT,3,100,100])

    for var in range(len(foldersKnownJacobs)):

        # g_name = get_pkg_data_filename(foldersKnownJacobs[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(foldersKnownJacobs[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(foldersKnownJacobs[var]+'/i_WCSClipped.fits')    

        g_name = get_pkg_data_filename(foldersKnownJacobs[var]+'/g_norm.fits')
        r_name = get_pkg_data_filename(foldersKnownJacobs[var]+'/r_norm.fits')
        i_name = get_pkg_data_filename(foldersKnownJacobs[var]+'/i_norm.fits')    
    
        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]    
        
        DataKnownJacobs[var] = [g, r, i]
    return (DataKnownJacobs)

def getUnknown(num, base_dir = 'KnownLenses'):
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
        DataUnknown (numpy array):  This is the numpy array of the unknown dataset.
    """

    if num == 47:
        pathUnknown = '%s/Unknown_Processed_47' % (base_dir)
    elif num == 84:
        pathUnknown = '%s/Unknown_Processed_84' % (base_dir)
    elif num == 131:
        pathUnknown = '%s/Unknown_Processed_131' % (base_dir)

    foldersUnknown = []
    for root, dirs, files in os.walk(pathUnknown):
        for folder in dirs:
            foldersUnknown.append(os.path.join(root, folder))

    nDT = len(foldersUnknown)
    DataUnknown = np.zeros([nDT,3,100,100])

    for var in range(len(foldersUnknown)):
        # g_name = get_pkg_data_filename(foldersUnknown[var]+'/g_WCSClipped.fits')
        # r_name = get_pkg_data_filename(foldersUnknown[var]+'/r_WCSClipped.fits')
        # i_name = get_pkg_data_filename(foldersUnknown[var]+'/i_WCSClipped.fits')    

        g_name = get_pkg_data_filename(foldersUnknown[var]+'/g_norm.fits')
        r_name = get_pkg_data_filename(foldersUnknown[var]+'/r_norm.fits')
        i_name = get_pkg_data_filename(foldersUnknown[var]+'/i_norm.fits')    

        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]    
        
        DataUnknown[var] = [g, r, i]

    return (DataUnknown)

def testDES2017():
    """
    This tests the unseen DES2017 images and unknown 47 images, to get the accuracy rate 
    of these unseen images that aren't used in training. 

    Returns:
        knownDES2017Array (numpy array):    This is the numpy array of the known DES2017 images.
        AccuracyScore_47 (float):           This is the accuracy score of the 47 unseen unknown images and of the
                                            47 images from DES2017, being tested on the already learnt set.
        KFoldAccuracy_47(float):            This is the accuracy score of the 47 unseen unknown images and of the 47
                                            images from DES2017 after k fold cross validation, being 
                                            tested on the already learnt set. 
    
    """

    knownDES2017Array = getDES2017()

    num = 47
    unknownArray = getUnknown(num)

    imageTest, labelsTest = loadImage(knownDES2017Array, unknownArray)
    x_ImageTest = imageTest.reshape(imageTest.shape[0], imageTest.shape[1]*imageTest.shape[2]*imageTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_ImageLabels = encoder.fit_transform(labelsTest)

    y_pred = clf_image.predict(x_ImageTest)
    AccuracyScore_47 = (accuracy_score(y_ImageLabels, y_pred))*100

    results = model_selection.cross_val_score(clf_image, x_ImageTest, y_ImageLabels, cv = kfold)
    KFoldAccuracy_47 = (results.mean())*100

    return(knownDES2017Array, AccuracyScore_47, KFoldAccuracy_47)

def testJacobs():
    """
    This tests the unseen Jacobs images and unknown 84 images, to get the accuracy rate 
    of these unseen images that aren't used in training. 

    Returns:
        knownJacobsArray (numpy array):     This is the numpy array of the known Jacobs images.
        AccuracyScore_84 (float):           This is the accuracy score of the 84 unseen unknown images and of the
                                            84 images from Jacobs, being tested on the already learnt set.
        KFoldAccuracy_84(float):            This is the accuracy score of the 84 unseen unknown images and of the 84
                                            images from Jacobs after k fold cross validation, being 
                                            tested on the already learnt set. 
    
    """

    knownJacobsArray = getJacobs()

    num = 84
    unknownArray = getUnknown(num)

    imageJacobsTest, labelsJacobsTest = loadImage(knownJacobsArray, unknownArray)
    x_ImageTest = imageJacobsTest.reshape(imageJacobsTest.shape[0], imageJacobsTest.shape[1]*imageJacobsTest.shape[2]*imageJacobsTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_ImageLabels = encoder.fit_transform(labelsJacobsTest)

    y_pred = clf_image.predict(x_ImageTest)
    AccuracyScore_84 = (accuracy_score(y_ImageLabels, y_pred))*100

    results = model_selection.cross_val_score(clf_image, x_ImageTest, y_ImageLabels, cv = kfold)
    KFoldAccuracy_84 = (results.mean())*100

    return(knownJacobsArray, AccuracyScore_84, KFoldAccuracy_84)

def testDES2017AndJacobs(knownDES2017Array, knownJacobsArray):
    """
    This tests the unseen DES2017 and Jacobs images together with the unknown 131 images, to get the accuracy rate 
    of these unseen images that aren't used in training. 

    Args:
        knownDES2017Array (numpy array):    This is the dataset of the unseen known DES2017 images.
        knownJacobsArray (numpy array):     This is the dataset of the unseen known Jacobs images.
    Returns:
        AccuracyScore_131 (float):          This is the accuracy score of the 131 unseen unknown images and of the
                                            131 images from DES2017 and Jacobs, being tested on the already learnt set.
        KFoldAccuracy_131(float):           This is the accuracy score of the 131 unseen unknown images and of the 131
                                            images from DES2017 and Jacobs after k fold cross validation, being 
                                            tested on the already learnt set. 
    """
    
    allKnownArray = np.vstack((knownDES2017Array, knownJacobsArray))

    num = 131
    unknownArray = getUnknown(num)

    imageKnownTest, labelsKnownTest = loadImage(allKnownArray, unknownArray)
    x_ImageTest = imageKnownTest.reshape(imageKnownTest.shape[0], imageKnownTest.shape[1]*imageKnownTest.shape[2]*imageKnownTest.shape[3]) # batchsize, height*width*3channels
    print(" x ImageTest: " + str(x_ImageTest.shape))

    encoder = LabelEncoder()
    y_ImageLabels = encoder.fit_transform(labelsKnownTest)

    y_pred = clf_image.predict(x_ImageTest)
    AccuracyScore_131 = (accuracy_score(y_ImageLabels, y_pred))*100

    results = model_selection.cross_val_score(clf_image, x_ImageTest, y_ImageLabels, cv = kfold)
    KFoldAccuracy_131 = (results.mean())*100

    return(AccuracyScore_131, KFoldAccuracy_131)

def makeTrainTest(positiveArray, negativeArray):
    """
    This makes the training and testing data sets that are to be made. This creates 
    a training image data set with the positive and negative images together. This 
    also creates a training label data set with the positive and negative images together.

    Args:  
        positiveArray (numpy array):    This is the positively simulated dataset images.
        negativeArray (numpy array):    This is the negative DES images.
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
        imageTrain_std (float):         This is the standard deviation of the entire training set, all 20000 images. 
        imageTrain_mean (float):        This is the mean of the entire training set, all 20000 images. 
        imageTrain_shape (list):        This is the shape of the entire training set, all 20000 images.
        imageLabels_shape (list):       This is the shape of the entire training sets' labels, all 20000 labels.
        xTrain_shape (list):            This is the shape of the training set of the images.
        xTest_shape (list):             This is the shape of the testing set of the images.
        yTrain_shape (list):            This is the shape of the training set of the labels.
        yTest_shape (list):             This is the shape of the testing set of the labels. 
    """

    imageTrain, imageLabels = loadImage(positiveArray, negativeArray)
    imageTrain_std = imageTrain.std()
    imageTrain_mean = imageTrain.mean()
    imageTrain_shape = imageTrain.shape

    # X = imageTrain.reshape(imageTrain.shape[0], imageTrain.shape[1]*imageTrain.shape[2]*imageTrain.shape[3])
    # print("X shape: " + str(X.shape))

    imageLabels_shape = imageLabels.shape

    # Encoding Y now
    encoder = LabelEncoder()
    Y = encoder.fit_transform(imageLabels)
    # print("Y shape: " +str(Y.shape))

    # Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
    test_percent = 0.2
    x_train, x_test, y_train, y_test = train_test_split(imageTrain, Y, shuffle=True, test_size =test_percent, random_state = 1 )
    xTrain_shape = x_train.shape
    xTest_shape = x_test.shape
    yTrain_shape = y_train.shape
    yTest_shape = y_test.shape

    # yTrain_shape = y_train.reshape(-1).shape
    # y_train=y_train.reshape(-1)
    # yTest_shape = y_test.reshape(-1).shape
    # y_test = y_test.reshape(-1)
    print("x_train: " +str(xTrain_shape))
    print("y_train: "+str(y_train.shape))
    print("x_test shape: "+str(xTest_shape))
    print("y_test shape: "+str(y_train.shape))

    train_percent = (1 - test_percent)

    return(x_train, x_test, y_train, y_test, train_percent, test_percent, imageTrain_std, imageTrain_mean, imageTrain_shape, imageLabels_shape, xTrain_shape, xTest_shape, yTrain_shape, yTest_shape)

def makeKerasModel():
    # mlp classifeir without cnn
    model = Sequential()
    model.add(Dense(1000, activation = 'relu', input_shape = (3, 100, 100)))
    model.add(Dense(100, activation = 'relu'))
    # model.add(Dense(100, activation = 'relu'))
    model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) # THE KERAS WITHOUT ES PNG IMAGE, HAS SIGMOID
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model = Sequential()
    # model.add(Conv2D(4, kernel_size = (3, 3), activation='relu', input_shape=(3, 100, 100)))
    # model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return (model)

def useKerasModel(positiveArray, negativeArray):
    x_train, x_test, y_train, y_test, train_percent, test_percent, imageTrain_std, imageTrain_mean, imageTrain_shape, imageLabels_shape, xTrain_shape, xTest_shape, yTrain_shape, yTest_shape = makeTrainTest(positiveArray, negativeArray)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience = 3)

    model = makeKerasModel()
    seqModel = model.fit(x_train, y_train, epochs=30, batch_size=200, validation_data=(x_test, y_test), callbacks = [es])

    description = str(model)
    # Accuracy Testing
    y_pred = model.predict(x_test)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    accuracyscore =  acc * 100.0
    AccuracyScore = accuracyscore
    print("Accuracy Score: " +str(AccuracyScore))

    # plot training vs validation loss. 
    history = History()
    train_loss = seqModel.history['loss']
    val_loss = seqModel.history['val_loss']
    train_accuracy = seqModel.history['accuracy']
    val_accuracy = seqModel.history['val_accuracy']

    # epochs = range(1,50)
    fig1 = plt.figure()
    plt.plot(train_loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig1.savefig('../Results/TrainingvsValidationLoss_Keras.png')
    # fig1.close()

    fig2 = plt.figure()
    plt.plot(train_accuracy, label = 'Train Accuracy')
    plt.plot(val_accuracy, label = 'Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig2.savefig('../Results/TrainingvsValidationAccuracy_Keras.png')
    # fig2.close()

    return(x_train, x_test, y_train, y_test, model)
    
def getKerasKFold(x_train, x_test, y_train, y_test):
    # Stratified K fold Cross Validation
    neural_network = KerasClassifier(build_fn=makeKerasModel,
                                    epochs=10, 
                                    batch_size=100, 
                                    verbose=0)

    scores = cross_val_score(neural_network, x_test, y_test, cv=10)
    print('Scores Length: ' + str(len(scores)))
    print('Scores: ' +str(scores))
    scoresMean = scores.mean()
    scoreStd = scores.std()

    fig4 = plt.figure()
    # plt.plot(scores, label = 'Scores')
    plt.plot(scoresMean, label = 'Scores Mean')
    plt.plot(scoreStd, label = 'Scores Standard Deviation')
    plt.legend()
    fig4.savefig('../Results/KerasKFoldScores.png')

    fig5 = plt.figure()
    plt.plot(scores, label ='Scores')
    plt.plot(scoresMean, label = 'Scores Mean')
    plt.plot(scoreStd, label = 'Scores Standard Deviation')
    plt.legend()
    fig5.savefig('../Results/KerasKFold_AllScores.png')
    #_____________________________________________________________________________________________________________________________

# MAIN

positiveArray = getPositiveSimulated()
negativeArray = getNegativeDES()
x_train, x_test, y_train, y_test, model = useKerasModel(positiveArray, negativeArray)
getKerasKFold(x_train, x_test, y_train, y_test)
    
#______________________________________________________________________________________________________________________
# knownDES2017, AccuracyScore_47, KFoldAccuracy_47 = testDES2017()
# knownJacobs, AccuracyScore_84, KFoldAccuracy_84= testJacobs()
# AccuracyScore_131, KFoldAccuracy_131 =testDES2017AndJacobs(knownDES2017, knownJacobs)

# # write to ml_Lenses_results.xlsx
# elementList = makeExcelTable.getElementList(description, imageTrain_std, imageTrain_mean, imageTrain_shape, imageLabels_shape, train_percent, test_percent, xTrain_shape, xTest_shape, yTrain_shape, yTest_shape, n_splits, random_state, AccuracyScore, KFoldAccuracy, AccuracyScore_47, KFoldAccuracy_47, AccuracyScore_84, KFoldAccuracy_84, AccuracyScore_131, KFoldAccuracy_131)
# filename = '../Results/ml_Lenses_results.csv'
# makeExcelTable.appendRowAsList(filename, elementList)
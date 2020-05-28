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
from keras.models import Model

# TO EXTRACT FEATURES FROM CNN 
# https://datascience.stackexchange.com/questions/17513/extracting-features-using-tensorflow-cnn

# FUNCTIONS
def getPositiveSimulated(base_dir = 'NewLenses/LenseWithDES'):
    """
    This gets the g, r, and i of the 10 000 positively simulated images from the 
    PositiveWithDESSky, as well as returning the positively simulate array.

    Args:
        base_dir (string):      This the root file path of the positively simulated images.  
    Returns:
        dataPos(numpy array):   This is the array of positively simulated images.
    """

    folders = {}
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value

    #number of Positive DataPoints
    nDT = len(folders)

    dataPos = np.zeros([nDT, 3, 100, 100])

    # key is name of folder number
    # value is the number of the folder to be added to the file name

    counter = 0
    for key, value in folders.items():

        gName = get_pkg_data_filename(value + '/' + str(key) + '_g_norm.fits')
        rName = get_pkg_data_filename(value + '/' + str(key) + '_r_norm.fits')
        iName = get_pkg_data_filename(value + '/' + str(key) + '_i_norm.fits')

        # gName = get_pkg_data_filename(value + '/' + str(key) + '_posSky_g.fits')
        # rName = get_pkg_data_filename(value + '/' + str(key) + '_posSky_r.fits')
        # iName = get_pkg_data_filename(value + '/' + str(key) + '_posSky_i.fits')
        
        g = fits.open(gName)[0].data[0:100,0:100]
        r = fits.open(rName)[0].data[0:100,0:100]
        i = fits.open(iName)[0].data[0:100,0:100]
        
        dataPos[counter] = [g, r, i] 
        counter += 1
        # just to run, and use less things
        # if counter > 1500:
        #     break
    return (dataPos)

def getNegativeDES(base_dir = 'DES/DES_Processed'):
    """
    This gets the g, r, and i  10 000 negative images from the 
    DES/DES_Processedfolder, as well as returning the 
    negative array,

    Args:
        base_dir (string):      This the root file path of the negative images.  
    Returns:
        dataNeg (numpy array):  This is the array of negative images.
    """
    foldersNeg = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersNeg.append(os.path.join(root, folder))
    nDT = len(foldersNeg)
    dataNeg = np.zeros([nDT,3,100,100])

    for var in range(len(foldersNeg)):
        # if var > 1500:
        # break
        # gName = get_pkg_data_filename(foldersNeg[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersNeg[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersNeg[var]+'/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersNeg[var]+'/g_norm.fits')
        rName = get_pkg_data_filename(foldersNeg[var]+'/r_norm.fits')
        iName = get_pkg_data_filename(foldersNeg[var]+'/i_norm.fits')    

        g = fits.open(gName)[0].data[0:100,0:100]
        r = fits.open(rName)[0].data[0:100,0:100]
        i = fits.open(iName)[0].data[0:100,0:100]    
        
        dataNeg[var] = [g, r, i]
        # just to run, and use less things
        # if var > 1500:
        #     break
    return (dataNeg)

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
        imageTrain (numpy array):      This is the numpy array of the positive and negative arrays added
                                        together to make a single array.
        imageLabels (numpy array):     This is the numpy array of the labels for the positive and negative 
                                        arrays added together to make a single array.
    """

    positiveData = []
    negativeData = []
    positiveLabel = []
    negativeLabel = []
    imageTrain = []
    imageLabels = []

    for num in range(0,len(positiveArray)):
        imageTrain.append(positiveArray[num])
        # labelPos = 'Gravitational Lensing'
        labelPos = 1 # assign 1 for gravitational lensing
        imageLabels.append(labelPos)
    
    for num in range(0,len(negativeArray)):
        imageTrain.append(negativeArray[num])
        # labelNeg = 'No Gravitational Lensing'
        labelNeg = 0 # assign 0  for non gravitational lenses 
        imageLabels.append(labelNeg)

    return (np.array(imageTrain), np.array(imageLabels))

def getDES2017(base_dir = 'KnownLenses/DES2017/'):
    """
    This is used to get g, r, and i images of the DES2017 array, which contains 47 unseen known lenses.
    Args:
        base_dir (string):          This is the root directory of the DES2017 folder. 
    Returns:
        dataKnownDES (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    foldersKnownDES2017 = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersKnownDES2017.append(os.path.join(root, folder))

    nDT = len(foldersKnownDES2017)
    dataKnownDES = np.zeros([nDT,3,100,100])

    for var in range(len(foldersKnownDES2017)):

        # gName = get_pkg_data_filename(foldersKnownDES2017[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersKnownDES2017[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersKnownDES2017[var]+'/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersKnownDES2017[var]+'/g_norm.fits')
        rName = get_pkg_data_filename(foldersKnownDES2017[var]+'/r_norm.fits')
        iName = get_pkg_data_filename(foldersKnownDES2017[var]+'/i_norm.fits')    
    
        g = fits.open(gName)[0].data[0:100,0:100]
        r = fits.open(rName)[0].data[0:100,0:100]
        i = fits.open(iName)[0].data[0:100,0:100]    
        
        dataKnownDES[var] = [g, r, i]

    return (dataKnownDES)

def getJacobs(base_dir = 'KnownLenses/Jacobs_KnownLenses/'):
    """
    This is used to get g, r, and i images of the known Jacobs dataset, which contains 84 unseen known lenses.

    Args:
        base_dir (string):          This is the root directory of the DES2017 folder. 
    Returns:
        dataKnownDES (numpy array): This is the numpy array of the the DES2017 dataset.
    """

    foldersKnownJacobs = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersKnownJacobs.append(os.path.join(root, folder))
    nDT = len(foldersKnownJacobs)
    dataKnownJacobs = np.zeros([nDT,3,100,100])

    for var in range(len(foldersKnownJacobs)):

        # gName = get_pkg_data_filename(foldersKnownJacobs[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersKnownJacobs[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersKnownJacobs[var]+'/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersKnownJacobs[var]+'/g_norm.fits')
        rName = get_pkg_data_filename(foldersKnownJacobs[var]+'/r_norm.fits')
        iName = get_pkg_data_filename(foldersKnownJacobs[var]+'/i_norm.fits')    
    
        g = fits.open(gName)[0].data[0:100,0:100]
        r = fits.open(rName)[0].data[0:100,0:100]
        i = fits.open(iName)[0].data[0:100,0:100]    
        
        dataKnownJacobs[var] = [g, r, i]

    return (dataKnownJacobs)

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
        dataUnknown (numpy array):  This is the numpy array of the unknown dataset.
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
    dataUnknown = np.zeros([nDT,3,100,100])

    for var in range(len(foldersUnknown)):
        # gName = get_pkg_data_filename(foldersUnknown[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersUnknown[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersUnknown[var]+'/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersUnknown[var]+'/g_norm.fits')
        rName = get_pkg_data_filename(foldersUnknown[var]+'/r_norm.fits')
        iName = get_pkg_data_filename(foldersUnknown[var]+'/i_norm.fits')    

        g = fits.open(gName)[0].data[0:100,0:100]
        r = fits.open(rName)[0].data[0:100,0:100]
        i = fits.open(iName)[0].data[0:100,0:100]    
        
        dataUnknown[var] = [g, r, i]

    return (dataUnknown)

def testDES2017(model, neuralNetwork, nSplits):
    """
    This tests the unseen DES2017 images and unknown 47 images, to get the accuracy rate 
    of these unseen images that aren't used in training. 

    Returns:
        knownDES2017Array (numpy array):    This is the numpy array of the known DES2017 images.
        accuracyScore_47 (float):           This is the accuracy score of the 47 unseen unknown images and of the
                                            47 images from DES2017, being tested on the already learnt set.
        kFoldAccuracy_47(float):            This is the accuracy score of the 47 unseen unknown images and of the 47
                                            images from DES2017 after k fold cross validation, being 
                                            tested on the already learnt set. 
    
    """

    knownDES2017Array = getDES2017()

    num = 47
    unknownArray = getUnknown(num)

    imageTest, labelsTest = loadImage(knownDES2017Array, unknownArray)
    xImageTest = imageTest.reshape(imageTest.shape[0], imageTest.shape[1]*imageTest.shape[2]*imageTest.shape[3]) # batchsize, height*width*3channels
    # print('xImageTest length: '+str(len(xImageTest)))
    # print('imageTest shape: '+str(imageTest.shape()))
    # print('labelsTest Shape: '+str(labelsTest.shape()))

    encoder = LabelEncoder()
    yImageLabels = encoder.fit_transform(labelsTest)
    # print('yImageLabels Shape: '+str(yImageLabels.shape()))

    # Get Accuracy Score tests DES2017 on the mlpclassifier:
    yPred = model.predict(imageTest)
    # print('yPred Shape: '+str(yPred.shape()))
    _, acc = model.evaluate(imageTest, yImageLabels , verbose=0)
    accuracyScore_47 = acc * 100

    # get the k fold accuracy after k fold cross validation
    scores = cross_val_score(neuralNetwork, imageTest, yImageLabels, scoring = 'accuracy', cv=nSplits)
    scoresMean = scores.mean()*100
    print("kFold47 Scores Mean: " +str(scoresMean))
    kFoldStd_47 = scores.std()
    print("kFold47 Scores Std: " +str(kFoldStd_47))
    kFoldAccuracy_47 = scoresMean

    return(knownDES2017Array, accuracyScore_47, kFoldAccuracy_47,kFoldStd_47)

def testJacobs(model, neuralNetwork, nSplits):
    """
    This tests the unseen Jacobs images and unknown 84 images, to get the accuracy rate 
    of these unseen images that aren't used in training. 

    Returns:
        knownJacobsArray (numpy array):     This is the numpy array of the known Jacobs images.
        accuracyScore_84 (float):           This is the accuracy score of the 84 unseen unknown images and of the
                                            84 images from Jacobs, being tested on the already learnt set.
        kFoldAccuracy_84(float):            This is the accuracy score of the 84 unseen unknown images and of the 84
                                            images from Jacobs after k fold cross validation, being 
                                            tested on the already learnt set. 
    
    """

    knownJacobsArray = getJacobs()

    num = 84
    unknownArray = getUnknown(num)

    imageJacobsTest, labelsJacobsTest = loadImage(knownJacobsArray, unknownArray)
    xImageTest = imageJacobsTest.reshape(imageJacobsTest.shape[0], imageJacobsTest.shape[1]*imageJacobsTest.shape[2]*imageJacobsTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    yImageLabels = encoder.fit_transform(labelsJacobsTest)

    # Get Accuracy Score tests Jacobs on the mlpclassifier:
    yPred = model.predict(imageJacobsTest)
    _, acc = model.evaluate(imageJacobsTest, yImageLabels, verbose=0)
    accuracyScore_84 = acc * 100

    # get the k fold accuracy after k fold cross validation
    scores = cross_val_score(neuralNetwork, imageJacobsTest, yImageLabels, scoring = 'accuracy', cv=nSplits)
    scoresMean = scores.mean()*100
    print("kFold84 Scores Mean: " +str(scoresMean))
    kFoldStd_84 = scores.std()
    print("kFold84 Scores Std: " +str(kFoldStd_84))
    kFoldAccuracy_84 = scoresMean


    return(knownJacobsArray, accuracyScore_84, kFoldAccuracy_84, kFoldStd_84)

def testDES2017AndJacobs(knownDES2017Array, knownJacobsArray, model, neuralNetwork, nSplits):
    """
    This tests the unseen DES2017 and Jacobs images together with the unknown 131 images, to get the accuracy rate 
    of these unseen images that aren't used in training. 

    Args:
        knownDES2017Array (numpy array):    This is the dataset of the unseen known DES2017 images.
        knownJacobsArray (numpy array):     This is the dataset of the unseen known Jacobs images.
    Returns:
        accuracyScore_131 (float):          This is the accuracy score of the 131 unseen unknown images and of the
                                            131 images from DES2017 and Jacobs, being tested on the already learnt set.
        kFoldAccuracy_131(float):           This is the accuracy score of the 131 unseen unknown images and of the 131
                                            images from DES2017 and Jacobs after k fold cross validation, being 
                                            tested on the already learnt set. 
    """
    
    allKnownArray = np.vstack((knownDES2017Array, knownJacobsArray))

    num = 131
    unknownArray = getUnknown(num)

    imageKnownTest, labelsKnownTest = loadImage(allKnownArray, unknownArray)
    xImageTest = imageKnownTest.reshape(imageKnownTest.shape[0], imageKnownTest.shape[1]*imageKnownTest.shape[2]*imageKnownTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    yImageLabels = encoder.fit_transform(labelsKnownTest)

    # Get Accuracy Score tests DES2017 on the mlpclassifier:
    yPred = model.predict(imageKnownTest)
    _, acc = model.evaluate(imageKnownTest, yImageLabels, verbose=0)
    accuracyScore_131 = acc * 100

    # get the k fold accuracy after k fold cross validation
    scores = cross_val_score(neuralNetwork, imageKnownTest, yImageLabels, scoring = 'accuracy', cv=nSplits)
    scoresMean = scores.mean()*100
    print("kFold131 Scores Mean: " +str(scoresMean))
    kFoldStd_131 = scores.std()
    print("kFold131 Scores Std: " +str(kFoldStd_131))
    kFoldAccuracy_131 = scoresMean


    return(accuracyScore_131, kFoldAccuracy_131, kFoldStd_131)

def makeTrainTest(positiveArray, negativeArray):
    """
    This makes the training and testing data sets that are to be made. This creates 
    a training image data set with the positive and negative images together. This 
    also creates a training label data set with the positive and negative images together.

    Args:  
        positiveArray (numpy array):    This is the positively simulated dataset images.
        negativeArray (numpy array):    This is the negative DES images.
    Returns:
        xTrain (numpy array):          This is the array of the training set of the training images, 
                                        which is 80% of the image training set. 
        xTest (numpy array):           This is the array of the testing set of the training images, which 
                                        is the 20% of the training images. 
        yTrain (numpy array):          This is the array of the labels of the training labels, which is 80% 
                                        of the training labels.
        yTest (numpy array):           This is the array of the labels of the testing labels, which is 20% 
                                        of the training labels.
        trainPercent (float):          This is the percentage of data used for training (1- testPercent).
        testPercent (float):           This is the percentage of date used for testing.
        imageTrainStd (float):         This is the standard deviation of the entire training set, all 20000 images. 
        imageTrainMean (float):        This is the mean of the entire training set, all 20000 images. 
        imageTrainShape (list):        This is the shape of the entire training set, all 20000 images.
        imageLabelsShape (list):       This is the shape of the entire training sets' labels, all 20000 labels.
        xTrainShape (list):            This is the shape of the training set of the images.
        xTestShape (list):             This is the shape of the testing set of the images.
        yTrainShape (list):            This is the shape of the training set of the labels.
        yTestShape (list):             This is the shape of the testing set of the labels. 
    """

    imageTrain, imageLabels = loadImage(positiveArray, negativeArray)
    imageTrainStd = imageTrain.std()
    imageTrainMean = imageTrain.mean()
    imageTrainShape = imageTrain.shape

    # X = imageTrain.reshape(imageTrain.shape[0], imageTrain.shape[1]*imageTrain.shape[2]*imageTrain.shape[3])
    # print("X shape: " + str(X.shape))

    imageLabelsShape = imageLabels.shape

    # Encoding y now
    encoder = LabelEncoder()
    y = encoder.fit_transform(imageLabels)
    # print("y shape: " +str(y.shape))

    # Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
    testPercent = 0.2
    xTrain, xTest, yTrain, yTest = train_test_split(imageTrain, y, shuffle=True, test_size = testPercent, random_state = 1 )
    xTrainShape = xTrain.shape
    xTestShape = xTest.shape
    yTrainShape = yTrain.shape
    yTestShape = yTest.shape

    print("xTrain: " +str(xTrainShape))
    print("yTrain: "+str(yTrain.shape))
    print("xTest shape: "+str(xTestShape))
    print("yTest shape: "+str(yTrain.shape))

    trainPercent = (1 - testPercent)

    return(xTrain, xTest, yTrain, yTest, trainPercent, testPercent, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, xTrainShape, xTestShape, yTrainShape, yTestShape)

def makeKerasModel():
    # mlp classifeir without cnn
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_shape = (3, 100, 100))) # change this to have a 2d shape
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) # THE KERAS WITHOUT ES PNG IMAGE, HAS SIGMOID
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return (model)

def makeKerasCNNModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size = (3, 3), activation='relu', input_shape=(3, 100, 100)))
    model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return(model)

def useKerasModel(positiveArray, negativeArray):
    xTrain, xTest, yTrain, yTest, trainPercent, testPercent, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, xTrainShape, xTestShape, yTrainShape, yTestShape = makeTrainTest(positiveArray, negativeArray)
    es = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 3)
    model = makeKerasModel()
    seqModel = model.fit(xTrain, yTrain, epochs = 30, batch_size = 200, validation_data = (xTest, yTest), callbacks = [es])
    description = str(model)
    # Accuracy Testing
    yPred = model.predict(xTest)
    _, acc = model.evaluate(xTest,yTest, verbose=0)
    accuracyScore =  acc * 100.0
    print("Accuracy Score: " +str(accuracyScore))
    # plot training vs validation loss. 
    History()
    trainLoss = seqModel.history['loss']
    valLoss = seqModel.history['val_loss']
    trainAccuracy = seqModel.history['accuracy']
    valAccuracy = seqModel.history['val_accuracy']

    # epochs = range(1,50)
    fig1 = plt.figure()
    plt.plot(trainLoss, label = 'Training Loss')
    plt.plot(valLoss, label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig1.savefig('../Results/TrainingvsValidationLoss_Keras.png')

    fig2 = plt.figure()
    plt.plot(trainAccuracy, label = 'Train Accuracy')
    plt.plot(valAccuracy, label = 'Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig2.savefig('../Results/TrainingvsValidationAccuracy_Keras.png')

    #saving model weights in keras. 
    model.save_weights('kerasModel.h5')

    return(model, xTrain, xTest, yTrain, yTest, description,  trainPercent, testPercent,imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, xTrainShape, xTestShape, yTrainShape, yTestShape, accuracyScore)

def getKerasKFold(xTrain, xTest, yTrain, yTest):
    # Stratified K fold Cross Validation
    neuralNetwork = KerasClassifier(build_fn = makeKerasModel,  # https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
                                    epochs = 30,                 
                                    batch_size = 200, 
                                    verbose = 0)
    nSplits = 10    
    randomState = 0                            
    print("DONE 2")
    scores = cross_val_score(neuralNetwork, xTest, yTest, scoring = 'accuracy', cv = nSplits)
    print("DONE 3")
    scoresMean = scores.mean()*100
    print("kFold Scores Mean: " +str(scoresMean))
    kFoldStd = scores.std()
    print("kFold Scores Std: " +str(kFoldStd))
    print("DONE 4")

    fig3 = plt.figure()
    plt.plot(scores, label = 'Scores')
    plt.legend()
    fig3.savefig('../Results/KerasKFold_Scores.png')
    return(nSplits, randomState, scoresMean, kFoldStd, neuralNetwork)
    #_____________________________________________________________________________________________________________________________


def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

def visualizeKeras(model, xTrain):

    # https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras

    # topLayer= model.layers[0]
    # plt.show(topLayer.get_weights())

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(xTrain) # 20000 images and 100X100 dimensions and 3 channels
    plt.imshow(xTrain[10][:,:,0])
    display_activation(activations, 100, 100,3)
        
#_________________________________________________________________________________________________________________________
# MAIN

positiveArray = getPositiveSimulated()
negativeArray = getNegativeDES()
model, xTrain, xTest, yTrain, yTest, description, trainPercent, testPercent, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, xTrainShape, xTestShape, yTrainShape, yTestShape, accuracyScore = useKerasModel(positiveArray, negativeArray)
print("DONE 1")
visualizeKeras(model, xTrain)
# nSplits, randomState, kFoldAccuracy, kFoldStd, neuralNetwork = getKerasKFold(xTrain, xTest, yTrain, yTest)

    
# #______________________________________________________________________________________________________________________
# knownDES2017, accuracyScore_47, kFoldAccuracy_47,kFoldStd_47 = testDES2017(model, neuralNetwork, nSplits)
# knownJacobs, accuracyScore_84, kFoldAccuracy_84, kFoldStd_84= testJacobs(model, neuralNetwork, nSplits)
# accuracyScore_131, kFoldAccuracy_131, kFoldStd_131 =testDES2017AndJacobs(knownDES2017, knownJacobs,model, neuralNetwork, nSplits)

# # write to ml_Lenses_results.xlsx
# # makeExcelTable.makeInitialTable()
# elementList = makeExcelTable.getElementList(description, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, trainPercent, testPercent, xTrainShape, xTestShape, yTrainShape, yTestShape, nSplits, randomState, accuracyScore, kFoldAccuracy, kFoldStd, accuracyScore_47, kFoldAccuracy_47,kFoldStd_47, accuracyScore_84, kFoldAccuracy_84, kFoldStd_84, accuracyScore_131, kFoldAccuracy_131,kFoldStd_131)
# filename = '../Results/ml_Lenses_results.csv'
# makeExcelTable.appendRowAsList(filename, elementList)
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

# FUNCTIONS
def getPositiveSimulated(base_dir = 'PositiveWithDESSky'):
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
        
        g = fits.open(gName)[0].data[0:100, 0:100]
        r = fits.open(rName)[0].data[0:100, 0:100]
        i = fits.open(iName)[0].data[0:100, 0:100]
        
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
    dataNeg = np.zeros([nDT, 3, 100, 100])

    for var in range(len(foldersNeg)):

        # gName = get_pkg_data_filename(foldersNeg[var] + '/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersNeg[var] + '/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersNeg[var] + '/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersNeg[var] + '/g_norm.fits')
        rName = get_pkg_data_filename(foldersNeg[var] + '/r_norm.fits')
        iName = get_pkg_data_filename(foldersNeg[var] + '/i_norm.fits')    

        g = fits.open(gName)[0].data[0:100, 0:100]
        r = fits.open(rName)[0].data[0:100, 0:100]
        i = fits.open(iName)[0].data[0:100, 0:100]    
        
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

    for num in range(0, len(positiveArray)):
        imageTrain.append(positiveArray[num])
        labelPos = 'Gravitational Lensing'
        # labelPos = 1 # assign 1 for gravitational lensing
        imageLabels.append(labelPos)
    
    for num in range(0, len(negativeArray)):
        imageTrain.append(negativeArray[num])
        labelNeg = 'No Gravitational Lensing'
        # labelNeg = 1 # assign 1 for no lensing
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
    dataKnownDES = np.zeros([nDT, 3, 100, 100])

    for var in range(len(foldersKnownDES2017)):

        # gName = get_pkg_data_filename(foldersKnownDES2017[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersKnownDES2017[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersKnownDES2017[var]+'/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersKnownDES2017[var] + '/g_norm.fits')
        rName = get_pkg_data_filename(foldersKnownDES2017[var] + '/r_norm.fits')
        iName = get_pkg_data_filename(foldersKnownDES2017[var] + '/i_norm.fits')    
    
        g = fits.open(gName)[0].data[0:100, 0:100]
        r = fits.open(rName)[0].data[0:100, 0:100]
        i = fits.open(iName)[0].data[0:100, 0:100]    
        
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
    dataKnownJacobs = np.zeros([nDT, 3, 100, 100])

    for var in range(len(foldersKnownJacobs)):

        # gName = get_pkg_data_filename(foldersKnownJacobs[var] + '/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersKnownJacobs[var] + '/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersKnownJacobs[var] + '/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersKnownJacobs[var] + '/g_norm.fits')
        rName = get_pkg_data_filename(foldersKnownJacobs[var] + '/r_norm.fits')
        iName = get_pkg_data_filename(foldersKnownJacobs[var] + '/i_norm.fits')    
    
        g = fits.open(gName)[0].data[0:100, 0:100]
        r = fits.open(rName)[0].data[0:100, 0:100]
        i = fits.open(iName)[0].data[0:100, 0:100]    
        
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
    dataUnknown = np.zeros([nDT, 3, 100, 100])

    for var in range(len(foldersUnknown)):
        # gName = get_pkg_data_filename(foldersUnknown[var]+'/g_WCSClipped.fits')
        # rName = get_pkg_data_filename(foldersUnknown[var]+'/r_WCSClipped.fits')
        # iName = get_pkg_data_filename(foldersUnknown[var]+'/i_WCSClipped.fits')    

        gName = get_pkg_data_filename(foldersUnknown[var] + '/g_norm.fits')
        rName = get_pkg_data_filename(foldersUnknown[var] + '/r_norm.fits')
        iName = get_pkg_data_filename(foldersUnknown[var] + '/i_norm.fits')    

        g = fits.open(gName)[0].data[0:100, 0:100]
        r = fits.open(rName)[0].data[0:100, 0:100]
        i = fits.open(iName)[0].data[0:100, 0:100]    
        
        dataUnknown[var] = [g, r, i]

    return (dataUnknown)

def testDES2017():
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
    xImageTest = imageTest.reshape(imageTest.shape[0], imageTest.shape[1] * imageTest.shape[2] * imageTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    yImageLabels = encoder.fit_transform(labelsTest)

    yPred = clfImages.predict(xImageTest)
    accuracyScore_47 = (accuracy_score(yImageLabels, yPred))*100

    results = model_selection.cross_val_score(clfImages, xImageTest, yImageLabels, cv = kfold)
    kFoldAccuracy_47 = (results.mean())*100
    kFoldStd_47 = results.std

    return(knownDES2017Array, accuracyScore_47, kFoldAccuracy_47, kFoldStd_47)

def testJacobs():
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

    yPred = clfImages.predict(xImageTest)
    accuracyScore_84 = (accuracy_score(yImageLabels, yPred))*100

    results = model_selection.cross_val_score(clfImages, xImageTest, yImageLabels, cv = kfold)
    kFoldAccuracy_84 = (results.mean())*100
    kFoldStd_84 = results.std

    return(knownJacobsArray, accuracyScore_84, kFoldAccuracy_84, kFoldStd_84)

def testDES2017AndJacobs(knownDES2017Array, knownJacobsArray):
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
    xImageTest = imageKnownTest.reshape(imageKnownTest.shape[0], imageKnownTest.shape[1] * imageKnownTest.shape[2] * imageKnownTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    yImageLabels = encoder.fit_transform(labelsKnownTest)

    yPred = clfImages.predict(xImageTest)
    accuracyScore_131 = (accuracy_score(yImageLabels, yPred))*100

    results = model_selection.cross_val_score(clfImages, xImageTest, yImageLabels, cv = kfold)
    kFoldAccuracy_131 = (results.mean()) * 100
    kFoldStd_131 = results.std()

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
    print("imageTrain shape: " + str(imageTrainShape))
    imageLabelsShape = imageLabels.shape

    # reshape x
    x = imageTrain.reshape(imageTrain.shape[0], imageTrain.shape[1] * imageTrain.shape[2] * imageTrain.shape[3]) # batchsize, height*width*3channels
    print ("x shape: " + str(x.shape))

    # Encoding y now
    encoder = LabelEncoder()
    y = encoder.fit_transform(imageLabels)
    print (" y shape: " + str(y.shape))

    # Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
    testPercent = 0.2
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, shuffle = True, test_size = testPercent)
    xTrainShape = xTrain.shape
    xTestShape = xTest.shape
    yTrainShape = yTrain.shape
    yTestShape = yTest.shape
    print("xTrain: " + str(xTrainShape))
    print("yTrain: " + str(yTrain.shape))
    print("xTest shape: " + str(xTestShape))
    print("yTest shape: " + str(yTrain.shape))

    trainPercent = (1 - testPercent)

    return(xTrain, xTest, yTrain, yTest, trainPercent, testPercent, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, xTrainShape, xTestShape, yTrainShape, yTestShape)
#_____________________________________________________________________________________________________________________________
# MAIN

positiveArray = getPositiveSimulated()
negativeArray = getNegativeDES()

xTrain, xTest, yTrain, yTest, trainPercent, testPercent, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, xTrainShape, xTestShape, yTrainShape, yTestShape = makeTrainTest(positiveArray, negativeArray)

# Trianing the data with MLPClassifier, from scikit learn
clfImages = MLPClassifier(activation = 'relu',
                          hidden_layer_sizes = (100, 100, 100), # 3 layers of 100 neurons each
                          solver = 'sgd', 
                          verbose = True,
                          max_iter = 100,
                          batch_size = 200,
                          early_stopping = True) # batchsize = 200 default

description = str(clfImages)

# Getting training loss
clfImages.fit(xTrain, yTrain)
trainLoss = clfImages.loss_curve_ 

# Accuracy Testing
yPred = clfImages.predict(xTest)
accuracyScore = (accuracy_score(yTest, yPred))*100

# Getting validation loss
clfImages.fit(xTest, yTest)
valLoss = clfImages.loss_curve_

epochs = range(1, 30)
plt.plot(trainLoss, label = 'Training Loss')
plt.plot(valLoss, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../Results/TrainingvsValidationLoss_Sklearn.png')

# Cross Validation
nSplits = 10
randomState = 100
kfold = model_selection.KFold(nSplits = nSplits, random_state = randomState) 
kFoldAccuracy = model_selection.cross_val_score(clfImages, xTest, yTest, scoring='accuracy', cv = kfold)
kFoldMean = kFoldAccuracy.mean()*100
kFoldStd = kFoldAccuracy.std()
print("Accuracy Score: " + str(accuracyScore))
print("Accuracy Type: " + str(type(accuracyScore)))
print("Score Mean: " + str(kFoldMean))
print("Scores Std: " + str(kFoldStd))

fig4 = plt.figure()
plt.plot(kFoldAccuracy, label = 'Scores')
plt.legend()
fig4.savefig('../Results/SkLearnKFold_Scores.png')

fig5 = plt.figure()
plt.plot(trainLoss, label = 'Training Loss')
plt.plot(valLoss, label = 'Validation Loss')
# plt.plot(accuracyScore, label = 'Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../Results/LossAccuracy_Sklearn.png')

#______________________________________________________________________________________________________________________
knownDES2017, accuracyScore_47, kFoldAccuracy_47, kFoldStd_47 = testDES2017()
knownJacobs, accuracyScore_84, kFoldAccuracy_84, kFoldStd_84 = testJacobs()
accuracyScore_131, kFoldAccuracy_131, kFoldStd_131 = testDES2017AndJacobs(knownDES2017, knownJacobs)

# write to ml_Lenses_results.xlsx
makeExcelTable.makeInitialTable()
elementList = makeExcelTable.getElementList(description, imageTrainStd, imageTrainMean, imageTrainShape, imageLabelsShape, trainPercent, testPercent, xTrainShape, xTestShape, yTrainShape, yTestShape, nSplits, randomState, accuracyScore, kFoldAccuracy, kFoldStd, accuracyScore_47, kFoldAccuracy_47, kFoldStd_47, accuracyScore_84, kFoldAccuracy_84, kFoldStd_84, accuracyScore_131, kFoldAccuracy_131, kFoldStd_131)
filename = '../Results/ml_Lenses_results.csv'
makeExcelTable.appendRowAsList(filename, elementList)
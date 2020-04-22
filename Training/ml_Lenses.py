"""
This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the gravitational lenses.
"""
# IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# FUNCTIONS
def getPositiveSimulated(base_dir = 'PositiveWithDESSky'):

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

    dataSetName = 'Data Positively Simulated'
    return (DataPos, dataSetName)

def getNegativeDES(base_dir = 'DES/DES_Processed'):

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
    dataSetName = 'Data Negative From DES'
    return (DataNeg, dataSetName)

def checkParameters(dataSetName, arrayToCheck):
    print("Standard deviation of %s : %s " % (dataSetName, arrayToCheck.std()))
    print("Mean of %s : %s" % (dataSetName, arrayToCheck.mean()))
    print("Shape of %s : %s" %(dataSetName, arrayToCheck.shape))

def loadImage(positiveArray, negativeArray):
    positiveData = []
    negativeData = []
    positiveLabel = []
    negativeLabel = []
    image_train = []
    image_labels = []

    for num in range(0,len(positiveArray)):
        image_train.append(positiveArray[num])
        labelPos = 'Gravitational Lensing'
        image_labels.append(labelPos)
    
    for num in range(0,len(negativeArray)):
        image_train.append(negativeArray[num])
        labelNeg = 'No Gravitational Lensing'
        image_labels.append(labelNeg)

    return (np.array(image_train), np.array(image_labels))

def getDES2017(base_dir = 'KnownLenses/DES2017/'):

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

    dataSetName = 'Known DES2017 Lenses'
    return (DataKnownDES, dataSetName)

def getJacobs(base_dir = 'KnownLenses/Jacobs_KnownLenses/'):
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
    dataSetName = 'Known Jacobs Lenses'
    return (DataKnownJacobs, dataSetName)

def getUnknown(num, base_dir = 'KnownLenses'):
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

    dataSetName = 'Unknown Images'
    return (DataUnknown, dataSetName)

def testDES2017():
    knownDES2017Array,des2017Name = getDES2017()
    checkParameters(des2017Name, knownDES2017Array)

    num = 47
    unknownArray, unknownName = getUnknown(num)
    checkParameters(unknownName, unknownArray)

    imageTest, labelsTest = loadImage(knownDES2017Array, unknownArray)
    x_ImageTest = imageTest.reshape(imageTest.shape[0], imageTest.shape[1]*imageTest.shape[2]*imageTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_ImageLabels = encoder.fit_transform(labelsTest)

    y_pred = clf_image.predict(x_ImageTest)
    imageAccuracy = accuracy_score(y_ImageLabels, y_pred)
    print("Image DES2017 Accuracy: " + str(imageAccuracy))

    return(knownDES2017Array, des2017Name)

def testJacobs():
    knownJacobsArray, jacobsName = getJacobs()
    checkParameters(jacobsName, knownJacobsArray)

    num = 84
    unknownArray, unknownName = getUnknown(num)
    checkParameters(unknownName, unknownArray)

    imageJacobsTest, labelsJacobsTest = loadImage(knownJacobsArray, unknownArray)
    x_ImageTest = imageJacobsTest.reshape(imageJacobsTest.shape[0], imageJacobsTest.shape[1]*imageJacobsTest.shape[2]*imageJacobsTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_ImageLabels = encoder.fit_transform(labelsJacobsTest)

    y_pred = clf_image.predict(x_ImageTest)
    imageAccuracy = accuracy_score(y_ImageLabels, y_pred)
    print("Image Jacobs Accuracy: " + str(imageAccuracy))

    return(knownJacobsArray, jacobsName)

def testDES2017AndJacobs(knownDES2017Array, des2017Name, knownJacobsArray, jacobsName):
    
    allKnownArray = np.vstack((knownDES2017Array, knownJacobsArray))
    allKnownName = np.vstack((des2017Name, jacobsName))
    checkParameters(allKnownName, allKnownArray)

    num = 131
    unknownArray, unknownName = getUnknown(num)
    checkParameters(unknownName, unknownArray)

    imageKnownTest, labelsKnownTest = loadImage(allKnownArray, unknownArray)
    x_ImageTest = imageKnownTest.reshape(imageKnownTest.shape[0], imageKnownTest.shape[1]*imageKnownTest.shape[2]*imageKnownTest.shape[3]) # batchsize, height*width*3channels

    encoder = LabelEncoder()
    y_ImageLabels = encoder.fit_transform(labelsKnownTest)

    y_pred = clf_image.predict(x_ImageTest)
    imageAccuracy = accuracy_score(y_ImageLabels, y_pred)
    print("Image Jacobs Accuracy: " + str(imageAccuracy))

def makeTrainTest(positiveArray, negativeArray):
    imageTrain, imageLabels = loadImage(positiveArray, negativeArray)

    # check imageTrain shape
    checkParameters('ImageTrain' , imageTrain)

    # check shape of ImageLabels:
    print('Shape of ImageLabels: %s' %(imageLabels.shape))

    im2disp = imageTrain[10].transpose((1,2,0)) # changed 0,1,2,3 array to 0,1,2 for images(this is now from 10000,3, 100, 100, to 3,100,10000 )
    plt.imshow(im2disp)
    plt.show()
    print('Label: ' , imageLabels[10])

    # reshape X
    X = imageTrain.reshape(imageTrain.shape[0], imageTrain.shape[1]*imageTrain.shape[2]*imageTrain.shape[3]) # batchsize, height*width*3channels

    # Encoding Y now
    encoder = LabelEncoder()
    Y = encoder.fit_transform(imageLabels)

    # Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size = 0.2)

    # check the shapes of  x,y trains and x,y tests
    print("x_train shape: %s , and y_train shape: %s ." % (x_train.shape, y_train.shape))
    print("x_test.shape: %s , and y_test shape: %s ." %(x_test.shape, y_test.shape))
    return(x_train, x_test, y_train, y_test)
#_____________________________________________________________________________________________________________________________
# MAIN

positiveArray, posName = getPositiveSimulated()
negativeArray, negName = getNegativeDES()

#check parametes
checkParameters(posName, positiveArray)
checkParameters(negName, negativeArray)

x_train, x_test, y_train, y_test = makeTrainTest(positiveArray, negativeArray)

# Trianing the data with MLPClassifier, from scikit learn
clf_image = MLPClassifier(activation = 'relu',
                          hidden_layer_sizes = (100, 100, 100), # 3 layers of 100 neurons each
                          solver='adam', 
                          verbose=True,
                          max_iter=100)

clf_image.fit(x_train, y_train)

y_pred = clf_image.predict(x_test)
y_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy_Score: " +str(y_accuracy))

#______________________________________________________________________________________________________________________
knownDES2017, des2017Name = testDES2017()
knownJacobs, jacobsName = testJacobs()
testDES2017AndJacobs(knownDES2017, des2017Name, knownJacobs, jacobsName)
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
def getPositiveSimulated(base_dir = 'Training/PositiveWithDESSky'):

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

def getNegativeDES(base_dir = 'Training/DES/DES_Processed'):

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

def getDES2017(base_dir = 'Training/KnownLenses/DES2017/'):

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

def getJacobs(base_dir = 'Training/KnownLenses/Jacobs_KnownLenses/'):
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

def getUnknown(base_dir = 'Training/KnownLenses/Unknown_Processed_256/'):


    foldersUnknown = []
    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            foldersUnknown.append(os.path.join(root, folder))

    nDT = len(foldersUnknown)
    DataUnknown = np.zeros([nDT,3,100,100])

    for var in range(len(foldersUnknown)):
        g_name = get_pkg_data_filename(foldersUnknown[var]+'/g_WCSClipped.fits')
        r_name = get_pkg_data_filename(foldersUnknown[var]+'/r_WCSClipped.fits')
        i_name = get_pkg_data_filename(foldersUnknown[var]+'/i_WCSClipped.fits')    

        # g_name = get_pkg_data_filename(foldersUnknown[var]+'/g_norm.fits')
        # r_name = get_pkg_data_filename(foldersUnknown[var]+'/r_norm.fits')
        # i_name = get_pkg_data_filename(foldersUnknown[var]+'/i_norm.fits')    

        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]    
        
        DataUnknown[var] = [g, r, i]

    dataSetName = 'Unknown Images'
    return (DataUnknown, dataSetName)
#_____________________________________________________________________________________________________________________________
# MAIN

positiveArray, posName = getPositiveSimulated()
negativeArray, negName = getNegativeDES()

#check parametes
checkParameters(posName, positiveArray)
checkParameters(negName, negativeArray)

#load images into an image_train array and image_labels array
imageTrain, imageLabels = loadImage(positiveArray, negativeArray)

# check imageTrain shape
checkParameters('ImageTrain' , imageTrain)

# check shape of ImageLabels:
print('Shape of ImageLabels: %s' %(imageLabels.shape))

# add dimension to imageTrain
# print("10 in ImageTrain: %s" %(imageTrain[10]))

# checking by plotting image
# plt.imshow(imageTrain[10])
im2disp = imageTrain[10].transpose((1,2,0)) # changed 0,1,2,3 array to 0,1,2 for images(this is now from 10000,3, 100, 100, to 3,100,10000 )
plt.imshow(im2disp)
plt.show()
print('Label: ' , imageLabels[10])

# reshape X
X = imageTrain.reshape(imageTrain.shape[0], imageTrain.shape[1]*imageTrain.shape[2]*imageTrain.shape[3]) # batchsize, height*width*3channels
print("Shape of X: "+str(X.shape))

# Encoding Y now
encoder = LabelEncoder()
Y = encoder.fit_transform(imageLabels)

# Doing a train-test split with sklearn, to train the data, where 20% of the training data is used for the test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size = 0.2)

# check the shapes of  x,y trains and x,y tests
print("x_train shape: %s , and y_train shape: %s ." % (x_train.shape, y_train.shape))
print("x_test.shape: %s , and y_test shape: %s ." %(x_test.shape, y_test.shape))

# Trianing the data with MLPClassifier, from scikit learn
clf_image = MLPClassifier(activation = 'relu',
                          hidden_layer_sizes = (100, 100, 100), 
                          solver='adam', 
                          verbose=True,
                          max_iter=100)

clf_image.fit(x_train, y_train)

y_pred = clf_image.predict(x_test)
y_accuracy = accuracy_score(y_test, y_pred)

print("clf_image: "+str(clf_image))
print("Y_pred: " + str(y_pred))
print("Accuracy_Score: " +str(y_accuracy))

#______________________________________________________________________________________________________________________
knownDES2017Array,des2017Name = getDES2017()
knownJacobsArray, jacobsName = getJacobs()
unknownArray, unknownName = getUnknown()

checkParameters(des2017Name, knownDES2017Array)
checkParameters(jacobsName, knownJacobsArray)
checkParameters(unknownName, unknownArray)

imageTest, labelsTest = loadImage(knownDES2017Array, unknownArray)

checkParameters('Image DES2017 Test' , imageTest)
print('Shape of Image DES2017 Labels: %s' %(labelsTest.shape))

imageAccuracy = accuracy_score(imageTest, y_pred)
print("Image DES2017 Accuracy: " + str(imageAccuracy))

imageJacobsTest, labelsJacobsTest = loadImage(knownJacobsArray, unknownArray)

checkParameters('Image Jacobs Test: %s' % ( imageJacobsTest))
print('Shape of Image Jacobs Labels: %s' % (labelsJacobsTest.shape))

imageJacobsAccuracy = accuracy_score(imageTest, y_pred)
print("Image DES2017 Accuracy: " + str(imageJacobsAccuracy))

allKnownArray, allKnownName = np.vstack(knownDES2017Array, knownJacobsArray)
checkParameters(allKnownName, allKnownArray)

imageKnownTest, labelsKnownTest = loadImage(allKnownArray, unknownArray)
checkParameters('Image All Known Test: %s' % ( imageKnownTest))
print('Shape of Image All Known Labels: %s' % (labelsKnownTest.shape))

imageAllKnownAccuracy = accuracy_score(imageKnownTest, y_pred)
print("Image DES2017 Accuracy: " + str(imageAllKnownAccuracy))





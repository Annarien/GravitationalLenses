"""
This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the gravitational lenses.
"""
# IMPORTS
import os
import numpy as np

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits


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

        # g_name = get_pkg_data_filename(value + '/' + str(key) + '_g_norm.fits')
        # r_name = get_pkg_data_filename(value + '/' + str(key) + '_r_norm.fits')
        # i_name = get_pkg_data_filename(value + '/' + str(key) + '_i_norm.fits')

        g_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_g.fits')
        r_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_r.fits')
        i_name = get_pkg_data_filename(value + '/' + str(key) + '_posSky_i.fits')
        
        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]
        
        DataPos[counter] = [g, r, i] 
        counter += 1
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

        g_name = get_pkg_data_filename(foldersNeg[var]+'/g_WCSClipped.fits')
        r_name = get_pkg_data_filename(foldersNeg[var]+'/r_WCSClipped.fits')
        i_name = get_pkg_data_filename(foldersNeg[var]+'/i_WCSClipped.fits')    

        # g_name = get_pkg_data_filename(foldersNeg[var]+'/g_norm.fits')
        # r_name = get_pkg_data_filename(foldersNeg[var]+'/r_norm.fits')
        # i_name = get_pkg_data_filename(foldersNeg[var]+'/i_norm.fits')    

        g = fits.open(g_name)[0].data[0:100,0:100]
        r = fits.open(r_name)[0].data[0:100,0:100]
        i = fits.open(i_name)[0].data[0:100,0:100]    
        
        DataNeg[var] = [g, r, i]
    dataSetName = 'Data Negative From DES'
    return (DataNeg)

def checkParameters(dataSetName, arrayToCheck):
    print("Standard deviation of %s : %s " % (dataSetName, arrayToCheck.std()))
    print("Mean of %s : %s" % (dataSetName, arrayToCheck.mean()))
    print("Shape of %s : %s" %s(dataSetName, arrayToCheck.shape))



#_____________________________________________________________________________________________________________________________
# MAIN
positiveArray = []
negativeArray = []

positiveArray, posName = getPositiveSimulated()
negativeArray, negName = getNegativeDES()

#check parametes
checkParameters(posName, positiveArray)
checkParameters(negName, negativeArray)
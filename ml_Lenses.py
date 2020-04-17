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
    
    return (DataPos)
#_____________________________________________________________________________________________________________________________
# MAIN

positiveArray = getPositiveSimulated()
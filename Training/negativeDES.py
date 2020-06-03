"""
Downloading the original DES images (10000 * 10000 pixels).
These images are the into 100*100 pixels are cut, using random x, y coordinates, these images are 
known as background sky/noise. The original images are clipped using the World Coordinate System, 
and are 100*100 pixels in size around stellar/astronomical objects, and these images will be referred 
to as negativeDES images. These negativeDES images are normalised, as well as composite RGB images are created.
"""

import sys
import astropy.table as atpy
import numpy as np

from negativeDESUtils import getRandomIndices, loadDES, randomSkyClips, clipWCS, normaliseRGB

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
"""
Here we call the loadImages function which will declare the varaibles gDES, rDES, iDES.
Its is then clipped using the clipImages function.
And we write these images to a file in .fits format using writeClippedImagesToFile function.
"""

tableDES = atpy.Table().read("DES/DESGalaxies_18_I_22.fits")

# ensuring there is no none numbers in the gmag, rmag, and imag in the DES table. 
# ensuring that there is no Gmag with values of 99.
for key in ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I']:
    tableDES = tableDES[np.isnan(tableDES[key]) == False]

tableDES = tableDES[tableDES['MAG_AUTO_G'] < 24]
lenTabDES = len(tableDES)

training_size = 10000
testing_size = 1000
random_indices = []

training = getRandomIndices(training_size, random_indices, lenTabDES)
testing = getRandomIndices(training_size, random_indices, lenTabDES)

for i in range(training):
    num = training[i]

    tileName = tableDES['TILENAME'][num]
    print(type(tileName))
    gmag = tableDES['MAG_AUTO_G'][num]
    imag = tableDES['MAG_AUTO_I'][num]
    rmag = tableDES['MAG_AUTO_R'][num]
    ra = tableDES['RA'][num]
    dec = tableDES['DEC'][num]
    print('Gmag: ' + str(gmag))
    print('Imag: ' + str(imag))
    print('Rmag: ' + str(rmag))

    train_file = 'Training/Negative/'

    loadDES(tileName)
    randomSkyClips(num, tileName, ra, dec, gmag, rmag, imag)
    clipWCS(tileName, num, gmag, rmag, imag, ra, dec, train_file)
    normaliseRGB(num, tileName, train_file)

for i in range(testing):
    num = testing[i]

    tileName = tableDES['TILENAME'][num]
    print(type(tileName))
    gmag = tableDES['MAG_AUTO_G'][num]
    imag = tableDES['MAG_AUTO_I'][num]
    rmag = tableDES['MAG_AUTO_R'][num]
    ra = tableDES['RA'][num]
    dec = tableDES['DEC'][num]
    print('Gmag: ' + str(gmag))
    print('Imag: ' + str(imag))
    print('Rmag: ' + str(rmag))

    test_file = 'Testing/Negative/'

    loadDES(tileName)
    randomSkyClips(num, tileName, ra, dec, gmag, rmag, imag)
    clipWCS(tileName, num, gmag, rmag, imag, ra, dec, test_file)
    normaliseRGB(num, tileName, test_file)

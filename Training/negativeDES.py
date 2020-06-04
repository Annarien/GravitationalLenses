"""
Downloading the original DES images (10000 * 10000 pixels).
These images are the into 100*100 pixels are cut, using random x, y coordinates, these images are 
known as background sky/noise. The original images are clipped using the World Coordinate System, 
and are 100*100 pixels in size around stellar/astronomical objects, and these images will be referred 
to as negativeDES images. These negativeDES images are normalised, as well as composite RGB images are created.
"""

import sys
import os
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

table_des = atpy.Table().read("DES/DESGalaxies_18_I_22.fits")

# ensuring there is no none numbers in the gmag, rmag, and imag in the DES table. 
# ensuring that there is no Gmag with values of 99.
for key in ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I']:
    table_des = table_des[np.isnan(table_des[key]) == False]

table_des = table_des[table_des['MAG_AUTO_G'] < 24]
len_tab_des = len(table_des)

training_size = 10000
testing_size = 1000
random_indices = []

training = getRandomIndices(training_size, random_indices, len_tab_des)
testing = getRandomIndices(testing_size, random_indices, len_tab_des)

print("Training: "+str(training))
print("Testing: "+str(testing))

for i in range(0, len(training)):
    num = training[i]

    tile_name = table_des['TILENAME'][num]
    print(type(tile_name))
    g_mag = table_des['MAG_AUTO_G'][num]
    i_mag = table_des['MAG_AUTO_I'][num]
    r_mag = table_des['MAG_AUTO_R'][num]
    ra = table_des['RA'][num]
    dec = table_des['DEC'][num]
    print('Gmag: ' + str(g_mag))
    print('Imag: ' + str(i_mag))
    print('Rmag: ' + str(r_mag))

    train_file = 'Training/Negative/'
    train_dessky = 'Training/DESSky'

    if not os.path.exists('Training'):
        os.mkdir('Training')

    loadDES(tile_name)
    randomSkyClips(num, tile_name, ra, dec, g_mag, r_mag, i_mag, train_dessky)
    clipWCS(tile_name, num, g_mag, r_mag, i_mag, ra, dec, train_file)
    normaliseRGB(num, tile_name, train_file)

for i in range(0, len(testing)):
    num = testing[i]
    tile_name = table_des['TILENAME'][num]
    print(type(tile_name))
    g_mag = table_des['MAG_AUTO_G'][num]
    i_mag = table_des['MAG_AUTO_I'][num]
    r_mag = table_des['MAG_AUTO_R'][num]
    ra = table_des['RA'][num]
    dec = table_des['DEC'][num]
    print('Gmag: ' + str(g_mag))
    print('Imag: ' + str(i_mag))
    print('Rmag: ' + str(r_mag))

    test_file = 'Testing/Negative/'
    test_dessky = 'Testing/DESSky'

    if not os.path.exists('Testing'):
        os.mkdir('Testing')

    loadDES(tile_name)
    randomSkyClips(num, tile_name, ra, dec, g_mag, r_mag, i_mag, test_dessky)
    clipWCS(tile_name, num, g_mag, r_mag, i_mag, ra, dec, test_file)
    normaliseRGB(num, tile_name, test_file)

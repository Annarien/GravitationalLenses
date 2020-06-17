"""
This gets the UnknownLenses, that havent been seen before. This follows the same pattern as
the negativeDES.py code. This simply downloads code DES DR1 images and clips it using WCS, 
and then normalises it, and creates a rgb image. The amount of images downloaded depends on the
table requested in the knownLenses code, as Jacobs looks at 84 previously identified lenses, and
the DES2017 table looks at 47 previously identified lenses. The same amount of known lenses and 
unknown lenses are needed.
"""
# imports

import astropy.table as atpy
import matplotlib
import numpy as np

matplotlib.use('Agg')
from negativeDESUtils import getRandomIndices, normaliseRGB, loadDES, clipWCS
from positiveSetUtils import getNegativeNumbers

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
"""
Here we call the loadImages function which will declare the varaibles gDES, rDES, iDES.
Its is then clipped using the clipImages function.
And we write these images to a file in .fits format using writeClippedImagesToFile function.
"""

used_negative_train = getNegativeNumbers('Training/Negative')
used_negative_test = getNegativeNumbers('Testing/Negative')

used_list = used_negative_train + used_negative_test
print('Length of Used Negative List: ' + str(len(used_list)))

target_unseen_negative_size = 2000

table_des = atpy.Table().read("DES/DESGalaxies_18_I_22.fits")

# ensuring there is no none numbers in the gmag, rmag, and imag in the DES table. 
for key in ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I']:
    table_des = table_des[np.isnan(table_des[key]) == False]

table_des = table_des[table_des['MAG_AUTO_G'] < 24]
len_table_des = len(table_des)

negative_known = getRandomIndices(target_unseen_negative_size, used_list, len_table_des)

for i in range(0, len(negative_known)):
    num = negative_known[i]

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

    loadDES(tile_name)
    clipWCS(tile_name, num, g_mag, r_mag, i_mag, ra, dec,
            base_new='UnseenData/Negative/')  # takes DES images and clips it with RA, and DEC
    normaliseRGB(num, tile_name, base_dir='UnseenData/Negative/')

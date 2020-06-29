"""
This is to create the positively simulated images. 
By using the g, r, and i magnitudes from the COSMOS_Ilbert2009.fits, the magnitudes are realistic,
and are used when creating the simulated lenses. The images have DES background sky added to them, 
to create a more realistic positively simulated image, whereas without it, the images are too smooth.
These positive images which are now referred to as PositiveWithDESSky images. 
These images are normalised and also used to create a RGB composite image. 
"""

import random
import astropy.table as atpy
import numpy as np
from positiveSetUtils import cutCosmosTable, makeModelImage, addSky, normalise, getNegativeNumbers

# _______________________________________________________________________________________________________________
# MAIN

# Get the amount of data from the negative training and testing sets and use the same DESSky that was made by
# negativeDES.py, and add it to the positive simulated lenses.

numbers_train_neg = getNegativeNumbers('Training/Negative')
numbers_test_neg = getNegativeNumbers('Testing/Negative')

cosmos = atpy.Table().read("COSMOS_Ilbert2009.fits")
# to take out all nans in cosmos
for key in ['Rmag', 'Imag', 'Gmag']:
    cosmos = cosmos[np.isnan(cosmos[key]) == False]

g_ml = 0
r_ml = 0
i_ml = 0
g_ms = 0
r_ms = 0
i_ms = 0
rl = 0
ql = 0
b = 0
xs = 0
ys = 0
qs = 0
ps = 0
rs = 0

source_random_table, lens_random_table = cutCosmosTable(cosmos)

for i in range(0, len(numbers_train_neg)):
    num = numbers_train_neg[i]

    random_row = np.random.randint(0, len(lens_random_table))
    print('Random row number was %i' % random_row)
    g_ml = (lens_random_table['Gmag'][random_row])  # ml in g band, changed this from -2 to 0
    r_ml = (lens_random_table['Rmag'][random_row])  # ml in r band
    i_ml = (lens_random_table['Imag'][random_row])  # ml in i band

    random_row = np.random.randint(0, len(source_random_table))
    print('Random row number was %i' % random_row)
    g_ms = (source_random_table['Gmag'][random_row])  # ms in g band
    r_ms = (source_random_table['Rmag'][random_row])  # ms in r band
    i_ms = (source_random_table['Imag'][random_row])  # ms in i band

    ml = {'g_SDSS': g_ml,  # Mags for lens (dictionary of magnitudes by band)
          'r_SDSS': r_ml,
          'i_SDSS': i_ml}

    ms = {'g_SDSS': g_ms,  # Mags for source (dictionary of magnitudes by band)
          'r_SDSS': r_ms,
          'i_SDSS': i_ms}

    rl = float(random.uniform(1, 10))  # Half-light radius of the lens, in arcsec.
    ql = float(random.uniform(0.8, 1))  # Lens flattening (0 = circular, 1 = line)
    b = float(random.uniform(3, 5))  # Einstein radius in arcsec
    xs = float(random.uniform(1, 3))  # x-coord of source relative to lens centre in arcsec
    ys = float(random.uniform(1, 3))  # y-coord of source relative to lens centre in arcsec
    qs = float(random.uniform(1, 3))  # Source flattening (1 = circular, 0 = line)
    ps = float(random.uniform(0, 360))  # Position angle of source (in degrees)
    rs = float(random.uniform(1, 2))  # Half-light radius of the source, in arcsec.

    train_sky = 'Training/DESSky'
    train_positive_noiseless = 'Training/PositiveNoiseless3000'
    train_positive = 'Training/Positive3000'

    makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, train_positive_noiseless)
    addSky(num, train_positive_noiseless, train_sky, train_positive)
    normalise(num, train_positive)


# for i in range(0, len(numbers_test_neg)):
for i in range(0, 3000):
    num = numbers_test_neg[i]

    random_row = np.random.randint(0, len(lens_random_table))
    print('Random row number was %i' % random_row)
    g_ml = (lens_random_table['Gmag'][random_row])   # ml in g band
    r_ml = (lens_random_table['Rmag'][random_row])   # ml in r band
    i_ml = (lens_random_table['Imag'][random_row])   # ml in i band

    random_row = np.random.randint(0, len(source_random_table))
    print('Random row number was %i' % random_row)
    g_ms = (source_random_table['Gmag'][random_row])  # ms in g band
    r_ms = (source_random_table['Rmag'][random_row])  # ms in r band
    i_ms = (source_random_table['Imag'][random_row])  # ms in i band

    ml = {'g_SDSS': g_ml,  # Mags for lens (dictionary of magnitudes by band)
          'r_SDSS': r_ml,
          'i_SDSS': i_ml}

    ms = {'g_SDSS': g_ms,  # Mags for source (dictionary of magnitudes by band)
          'r_SDSS': r_ms,
          'i_SDSS': i_ms}

    rl = float(random.uniform(1, 10))  # Half-light radius of the lens, in arcsec.
    ql = float(random.uniform(0.8, 1))  # Lens flattening (0 = circular, 1 = line)
    b = float(random.uniform(3, 5))  # Einstein radius in arcsec
    xs = float(random.uniform(1, 3))  # x-coord of source relative to lens centre in arcsec
    ys = float(random.uniform(1, 3))  # y-coord of source relative to lens centre in arcsec
    qs = float(random.uniform(1, 3))  # Source flattening (1 = circular, 0 = line)
    ps = float(random.uniform(0, 360))  # Position angle of source (in degrees)
    rs = float(random.uniform(1, 2))  # Half-light radius of the source, in arcsec.

    test_sky = 'Testing/DESSky'
    test_positive_noiseless = 'Testing/PositiveNoiseless3000'
    test_positive = 'Testing/Positive3000'

    makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, test_positive_noiseless)
    addSky(num, test_positive_noiseless, test_sky, test_positive)
    normalise(num, test_positive)

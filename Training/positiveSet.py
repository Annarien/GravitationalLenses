"""
This is to create the positively simulated images. 
By using the g, r, and i magnitudes from the COSMOS_Ilbert2009.fits, the magnitudes are realistic,
and are used when creating the simulated lenses. The images have DES background sky added to them, 
to create a more realistic positively simulated image, whereas without it, the images are too smooth.
These positive images which are now referred to as PositiveWithDESSky images. 
These images are normalised and also used to create a RGB composite image. 
"""
import random
import sys

import astropy.table as atpy
import numpy as np

from positiveSetUtils import cutCosmosTable, makeModelImage, addSky, normalise, getNegativeNumbers

numbersTrainNeg = getNegativeNumbers('Training/Negative')
numbersTestNeg = getNegativeNumbers('Testing/Negative')

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

sourceRandomTable, lensRandomTable = cutCosmosTable(cosmos)

for i in range(0, len(numbersTrainNeg)):
    num = numbersTrainNeg[i]

    rndmRow = np.random.randint(0, len(lensRandomTable))
    print('Random row number was %i' % (rndmRow))
    g_ml = (lensRandomTable['Gmag'][rndmRow]) - 2  # ml in g band
    r_ml = (lensRandomTable['Rmag'][rndmRow]) - 2  # ml in r band
    i_ml = (lensRandomTable['Imag'][rndmRow]) - 2  # ml in i band

    rndmRow = np.random.randint(0, len(sourceRandomTable))
    print('Random row number was %i' % (rndmRow))
    g_ms = (sourceRandomTable['Gmag'][rndmRow])  # ms in g band
    r_ms = (sourceRandomTable['Rmag'][rndmRow])  # ms in r band
    i_ms = (sourceRandomTable['Imag'][rndmRow])  # ms in i band

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
    train_positive_noiseless = 'Training/PositiveNoiseless'
    train_positive = 'Training/Positive'

    makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, train_positive_noiseless)
    addSky(num, train_positive_noiseless, train_sky, train_positive)
    normalise(num, train_positive)

for i in range(0, len(numbersTestNeg)):
    num = numbersTestNeg[i]

    rndmRow = np.random.randint(0, len(lensRandomTable))
    print('Random row number was %i' % (rndmRow))
    g_ml = (lensRandomTable['Gmag'][rndmRow]) - 2  # ml in g band
    r_ml = (lensRandomTable['Rmag'][rndmRow]) - 2  # ml in r band
    i_ml = (lensRandomTable['Imag'][rndmRow]) - 2  # ml in i band

    rndmRow = np.random.randint(0, len(sourceRandomTable))
    print('Random row number was %i' % (rndmRow))
    g_ms = (sourceRandomTable['Gmag'][rndmRow])  # ms in g band
    r_ms = (sourceRandomTable['Rmag'][rndmRow])  # ms in r band
    i_ms = (sourceRandomTable['Imag'][rndmRow])  # ms in i band

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
    test_positive_noiseless = 'Testing/PositiveNoiseless'
    test_positive = 'Testing/Positive'

    makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, test_positive_noiseless)
    addSky(num, test_positive_noiseless, test_sky, test_positive)
    normalise(num, test_positive)


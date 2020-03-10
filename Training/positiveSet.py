"""

Matt's hacked version of the ModelAll.py code

"""
from __init__ import *
import glob
import os
import sys
import pylab as plt
import random
import numpy as np
import img_scale
import astropy.table  as atpy
from astropy.io import fits
from astLib import *
from astropy.visualization import make_lupton_rgb


cosmos = atpy.Table().read("COSMOS_Ilbert2009.fits")
# to take out all nans in cosmos
for key in ['Rmag','Imag','Gmag']:
    cosmos = cosmos[np.isnan(cosmos[key]) == False]
#------------------------------------------------------------------------------------------------------------
def cutCosmosTable(cosmos):
    """
    The cosmos table is used in order to get magnitudes,inorder to provide realistic magnitudes for our training set.  
    The cosmos table is restricted to all r magnitudes below 22 for sources.
    The cosmos table is restricted to all r magnitudes below 22 for lenses and between i magnitude for 18 and 22.
    This limits to the sources where the redshift (zpbest) is between 1 and 2.
    This limits to the sources where the redshift (zpbest) is between 0.1 and 0.3.
    For checking purposes the maximum and minimum, r and i magnitudes for the sources and lens tables, is printed as the code is run.

    Args:
        cosmos (table):          The table retrieved from the COSMOS_Ilbert2009.fits. 
        sourcesTable (table):    The redshift is limited  to between 1, and 2, for sources where the r magnitude is less than 22.
        lensTable (table):       Using the sourcesTable (rmag < 22), the redshift is limited between 0.1 and 0.3, 
                                 and the i magnitude is between 18 and 22.
        sourceMaxR (float):      The maximum value in the source table for the r magnitude.
        sourceMaxI (float):      The maximum value in the source table for the i magnitude.
        lensMaxR (float):        The maximum value in the lens table for the r magnitude.
        lensMaxI (float):        The maximum value in the lens table for the i magnitude.
        sourceMinR (float):      The minimum value in the source table for the r magnitude.
        sourceMinI (float):      The minimum value in the source table for the i magnitude. 
        lensMinR (float):        The minimum value in the lens table for the r magnitude.
        lensMinI (float):        The minimum value in the lens table for the i magnitude.
    Returns: 
        sourceTable, lensTable (table):     The sourcesTable and lensTable is created and returned.
    """ 
    tab = cosmos[cosmos['Rmag'] < 22]
    sourcesTable = tab[np.logical_and(tab['zpbest'] > 1, tab['zpbest'] < 2)]
    lensTable = tab[np.logical_and(tab['zpbest'] > 0.1, tab['zpbest'] < 0.3)]
    lensTable = lensTable[np.logical_and(lensTable['Imag'] > 18, lensTable['Imag'] < 22)]

    sourceMaxR = max(sourcesTable['Rmag'])
    sourceMaxI = max(sourcesTable['Imag'])
    print('SourceMaxR:' + str(sourceMaxR))
    print('SourceMaxI:' + str(sourceMaxI))
    lensMaxR = max(lensTable['Rmag'])
    lensMaxI = max(lensTable['Imag'])
    print('LensMaxR:' + str(lensMaxR))
    print('LensMaxI:' + str(lensMaxI))

    sourceMinR = min(sourcesTable['Rmag'])
    sourceMinI = min(sourcesTable['Imag'])
    print('SourceMinR:' + str(sourceMinR))
    print('SourceMinI:' + str(sourceMinI))
    lensMinR = min(lensTable['Rmag'])
    lensMinI = min(lensTable['Imag'])
    print('LensMinR:' + str(lensMinR))
    print('LensMinI:' + str(lensMinI))
    
    print('Row length of Sources Table ' + str(len(sourcesTable)) + '\n')
    print('Column length of Sources Table ' + str(len(sourcesTable[0])) + '\n')
    print (sourcesTable)
    print('Row length of Lens Table ' + str(len(lensTable)) + '\n')
    print('Column length of Lens Table '+ str(len(lensTable[0])) + '\n')
    print(lensTable)
    return(sourcesTable,lensTable)

def makeInitialTable(numObjects):
    """ 
    Writes a default of all zeroes table for g,r, i bands to include all the parameters that create the lenses and sources.

    Args:
        Folder (int):       Folder number of source.
        ml_g_SDSS (float):  The g band value of the lens magnitude.
        ml_r_SDSS (float):  The r band value of the lens magnitude.
        ml_i_SDSS (float):  The i band value of the lens magnitude.
        ms_g_SDSS (float):  The g band value of the source magnitude.
        ms_r_SDSS (float):  The r band value of the source magnitude.
        ms_i_SDSS (float):  The i band value of the source magnitude.
        rl (float):         Half-light radius of the lens, in arcsec.
        ql (float):         Flattening of the lens (1 = circular, 0 = line).
        b (float):          Einsten radius, in arcsec.
        ms (dictionary):    Apparent magnitude of the source, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        xs (float):         Horizontal coord of the source relative to the lens centre, in arcsec.
        ys (float):         Vertical coord of the source relative to the lens centre, in arcsec.
        qs (float):         Flattening of the source (1 = circular, 0 = line).
        ps (float):         Position angle of the source (degrees).
        rs (float):         Half-light radius of the source, in arcsec.

    Returns: 
        tab (table):        A table of default values of zero.
    """
    tab = atpy.Table()
    tab.add_column(atpy.Column( np.zeros(numObjects), "Folder"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ml_g_SDSS"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ml_r_SDSS"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ml_i_SDSS"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ms_g_SDSS"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ms_r_SDSS"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ms_i_SDSS"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "rl"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ql"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "b"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "xs"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ys"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "qs"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "ps"))
    tab.add_column(atpy.Column( np.zeros(numObjects), "rs"))
    return(tab)

def makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, survey = "DESc"):
    """ 
    Writes .fits images in g, r, i bands for lensed source.
    
    Args:
        ml (dictionary):  Apparent magnitude of the lens, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        rl (float):       Half-light radius of the lens, in arcsec.
        ql (float):       Flattening of the lens (1 = circular, 0 = line).
        b (float):        Einsten radius, in arcsec.
        ms (dictionary):  Apparent magnitude of the source, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        xs (float):       Horizontal coord of the source relative to the lens centre, in arcsec.
        ys (float):       Vertical coord of the source relative to the lens centre, in arcsec.
        qs (float):       Flattening of the source (1 = circular, 0 = line).
        ps (float):       Position angle of the source (degrees).
        rs (float):       Half-light radius of the source, in arcsec.
        label (str):      Label for this source (will be used as part of the file name)
        outDir (str):     Name of directory where images will be saved (format: outDir/label_band_image.fits).
        survey (str):     Name of survey (as defined by LensPop) - e.g., "DESc" corresponds to 
                          optimally stacked DES images.
    Returns: 
       Saved images and psf images in each band g, r, i for the source number as well as saving these fits images.
    """
    
    S = FastLensSim(survey, fractionofseeing = 1)
    S.bfac = float(2) # Not sure what these do - need to check
    S.rfac = float(2)
    
    sourcenumber = 1
    
    # Lens half-light radius in arcsec (weirdly, dictionary by band, all values the same, in arcsec)
    rlDict = {}
    for band in S.bands:
        rlDict[band] = rl
    
    S.setLensPars(ml, rlDict, ql, reset = True)
    S.setSourcePars(b, ms, xs, ys, qs, ps, rs, sourcenumber = sourcenumber)

    # Makes simulated image, convolving with PSF and adding noise
    model = S.makeLens(stochasticmode = "MP")
    SOdraw = numpy.array(S.SOdraw)
    S.loadModel(model)
    S.stochasticObserving(mode = "MP", SOdraw = SOdraw)
    print (num, band)
    S.ObserveLens()

    # Write FITS images
    outDir = "PositiveNoiseless"
    if os.path.exists(outDir) == False:
        os.makedirs(outDir)

    # For writing output
    folder = ('%s/%i' % (outDir, num))
    if os.path.exists(folder) == False:
        os.makedirs(folder)
    
    for band in S.bands:
        img = S.image[band]
        psf = S.psf[band]    
        fits.PrimaryHDU(img).writeto('%s/%s_image_%s.fits' % (folder, num, band), overwrite = True)  
        fits.PrimaryHDU(psf).writeto('%s/%s_psf_%s.fits' % (folder, num, band), overwrite = True)

def makeTable(tab, g_ml, r_ml, i_ml, g_ms, r_ms, i_ms, rl, ql, b, xs, ys, qs, ps, rs, num):
    """ 
    Inserting values into the initial table which now has values instead of zeroes for quick 
    access to look at the information easily for each source.

    Args:
        Folder (int):       Folder number of source.
        ml_g_SDSS (float):  The g band value of the lens magnitude.
        ml_r_SDSS (float):  The r band value of the lens magnitude.
        ml_i_SDSS (float):  The i band value of the lens magnitude.
        ms_g_SDSS (float):  The g band value of the source magnitude.
        ms_r_SDSS (float):  The r band value of the source magnitude.
        ms_i_SDSS (float):  The i band value of the source magnitude.
        rl (float):         Half-light radius of the lens, in arcsec.
        ql (float):         Flattening of the lens (1 = circular, 0 = line).
        b (float):          Einsten radius, in arcsec.
        ms (dictionary):    Apparent magnitude of the source, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        xs (float):         Horizontal coord of the source relative to the lens centre, in arcsec.
        ys (float):         Vertical coord of the source relative to the lens centre, in arcsec.
        qs (float):         Flattening of the source (1 = circular, 0 = line).
        ps (float):         Position angle of the source (degrees).
        rs (float):         Half-light radius of the source, in arcsec.

    Returns:
        tab (table):            A saved table in which the values of the parameters are assigned to for each source.    
    """

    tab["Folder"][num] = num
    tab["ml_g_SDSS"][num] = g_ml
    tab["ml_r_SDSS"][num] = r_ml
    tab["ml_i_SDSS"][num] = i_ml
    tab["ms_g_SDSS"][num] = g_ms
    tab["ms_r_SDSS"][num] = r_ms
    tab["ms_i_SDSS"][num] = i_ms
    tab["rl"][num] = rl
    tab["ql"][num] = ql
    tab["b"][num] = b
    tab["xs"][num] = xs
    tab["ys"][num] = ys
    tab["qs"][num] = qs
    tab["ps"][num] = ps
    tab["rs"][num] = rs
    tab.write('PositiveNoiseless.csv', overwrite = True)

def addSky(num):
    """
    Adds the DESSky images to the positive noiseless images made in this python file, to make them more realistic with noise from real DES images. 
    This is saved to 'PositiveWithDESsky/%s_%s_mock.fits'%(num,band).

    Args:
        band_skyImage (string):     The open .fits file for the DESSky for each number and band.
        band_mockImage (string):    The open .fits file for the simulated images for each number and band.
        WithSky (array):            The band_skyImage data array is added to the band_mockImage data array.
    
    Returns:
        Save images of the WithSky in the PositiveWithDESSky folders. These images are WithSky where the DESSky and the PositiveNoiseless images.
    """

    if os.path.exists('PositiveWithDESSky/%i' % (num)) == False:
        os.makedirs('PositiveWithDESSky/%i' % (num))

    for band in ['g','r','i']:
        bandSkyImage = fits.open('DESSky/%i_%s_sky.fits' % (num, band))
        bandPosNoiselessImage = fits.open('PositiveNoiseless/%s/%s_image_%s_SDSS.fits' % (num, num, band))
        withSky = bandSkyImage[0].data + bandPosNoiselessImage[0].data
        astImages.saveFITS('PositiveWithDESSky/%i/%i_posSky_%s.fits' % (num, num, band), withSky)
       
def normalise(num, base_dir = 'PositiveWithDESSky'):
    """
    The normalization follows a gaussian normalisation. The images are created in the directory path provided.

    Args:
        image (numpy array):        A numpy array of the g, r, i images which is made in the addSky function.
        im (numpy array):           A numpy array of the data of the image array.
        normImage(numpy array):     Normalised numpy array which is calculated as normImage = im/np.std(im) where 
                                    np.std is the standard deviation.  
    
    Returns:
        normImages (numpy array):   Images of the normalisation for the WCSClipped images are saved in the PositiveWithDESSky folder.
    """
    paths = {}
    paths['iImg'] = glob.glob('%s/%s/%s_posSky_i.fits' % (base_dir, num, num))[0]
    paths['rImg'] = glob.glob('%s/%s/%s_posSky_r.fits' % (base_dir, num, num))[0]
    paths['gImg'] = glob.glob('%s/%s/%s_posSky_g.fits' % (base_dir, num, num))[0]

    print (paths)

    rgbDict={}
    wcs=None
    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'Img']) as image:
            im = image[0].data
            normImage = (im-im.mean())/np.std(im)
            # astImages.saveFITS('%s/%s/%s_%s_norm.fits' % (base_dir, num, num, band), normImage, None)
            if wcs is None:
                wcs=astWCS.WCS(image[0].header, mode = 'pyfits')
            astImages.saveFITS('%s/%s/%s_%s_norm.fits' % (base_dir, num, num, band), normImage, None)
            rgbDict[band] = normImage

    minCut, maxCut=-1, 3
    cutLevels=[[minCut, maxCut], [minCut, maxCut], [minCut, maxCut]]
    plt.figure(figsize=(10, 10))
    astPlots.ImagePlot([rgbDict['i'], rgbDict['r'], rgbDict['g']],
                        wcs,
                        cutLevels = cutLevels,
                        axesLabels = None,
                        axesFontSize= 26.0,
                        axes = [0, 0, 1, 1])
    plt.savefig('%s/%i/%i_rgb.png' % ('PositiveWithDESSky', num, num))

#------------------------------------------------------------------------------------------------------------
# Main

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

numObjects = int(sys.argv[1]) # number of objects selected
sourceRandomTable, lensRandomTable = cutCosmosTable(cosmos)
tab = makeInitialTable(numObjects)

for num in range(numObjects):
    rndmRow = np.random.randint(0, len(lensRandomTable))
    print('Random row number was %i' % (rndmRow))
    g_ml = (lensRandomTable['Gmag'][rndmRow]) - 2 # ml in g band
    r_ml = (lensRandomTable['Rmag'][rndmRow]) - 2 # ml in r band
    i_ml = (lensRandomTable['Imag'][rndmRow]) - 2 # ml in i band

    rndmRow = np.random.randint(0, len(sourceRandomTable))
    print('Random row number was %i' % (rndmRow))
    g_ms = (sourceRandomTable['Gmag'][rndmRow])   # ms in g band
    r_ms = (sourceRandomTable['Rmag'][rndmRow])   # ms in r band
    i_ms = (sourceRandomTable['Imag'][rndmRow])   # ms in i band

    ml = {'g_SDSS' : g_ml,    # Mags for lens (dictionary of magnitudes by band)
          'r_SDSS' : r_ml,
          'i_SDSS' : i_ml}

    ms = {'g_SDSS': g_ms,     # Mags for source (dictionary of magnitudes by band)
          'r_SDSS': r_ms,     
          'i_SDSS': i_ms} 

    rl = float(random.uniform(1, 10))  # Half-light radius of the lens, in arcsec.
    ql = float(random.uniform(0.8, 1)) # Lens flattening (0 = circular, 1 = line)
    b = float(random.uniform(3, 5))    # Einstein radius in arcsec
    xs = float(random.uniform(1, 3))   # x-coord of source relative to lens centre in arcsec
    ys = float(random.uniform(1, 3))   # y-coord of source relative to lens centre in arcsec
    qs = float(random.uniform(1, 3))   # Source flattening (1 = circular, 0 = line)
    ps = float(random.uniform(0, 360))  # Position angle of source (in degrees)
    rs = float(random.uniform(1, 2))   # Half-light radius of the source, in arcsec.

    
    makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num)
    makeTable(tab, g_ml, r_ml, i_ml, g_ms, r_ms, i_ms, rl, ql, b, xs, ys, qs, ps, rs, num)
    addSky(num)
    normalise(num)
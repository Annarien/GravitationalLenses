"""
This is to create the positively simulated images. 
By using the g, r, and i magnitudes from the COSMOS_Ilbert2009.fits, the magnitudes are realistic,
and are used when creating the simulated lenses. The images have DES background sky added to them, 
to create a more realistic positively simulated image, whereas without it, the images are too smooth.
These positive images which are now referred to as PositiveWithDESSky images. 
These images are normalised and also used to create a RGB composite image. 
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
    The cosmos table is used in order to get magnitudes,inorder to provide realistic 
    magnitudes for our training set. This is used to create tables of magnitudes for 
    gravitational lenses and for the sources. This ensures that the magnitudes are 
    realistic in term of the g, r, and i magnitude bands, and those of the sources and lenses.

    Args:
        cosmos(table):          The table retrieved from the COSMOS_Ilbert2009.fits. 
    Returns: 
        sourceTable(table):     The sourcesTable containing objects with the revelant magnitudes of typical
                                strong galaxy-galaxy gravitational sources.
        lensTable(table):       The lensTable containing objects with the revelant magnitudes of typical 
                                strong galaxy-galaxy gravitational lenses.
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

def makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, survey = "DESc"):
    """ 
    Writes .fits images in g, r, and i bands to create the data set of positively simulated data, of strong 
    galaxy-galaxy gravitational lenses.   

    Args:
        ml(dictionary):     Apparent magnitude of the lens, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        rl(float):          Half-light radius of the lens, in arcsec.
        ql(float):          Flattening of the lens (1 = circular, 0 = line).
        b(float):           Einsten radius, in arcsec.
        ms(dictionary):     Apparent magnitude of the source, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        xs(float):          Horizontal coord of the source relative to the lens centre, in arcsec.
        ys(float):          Vertical coord of the source relative to the lens centre, in arcsec.
        qs(float):          Flattening of the source (1 = circular, 0 = line).
        ps(float):          Position angle of the source (degrees).
        rs(float):          Half-light radius of the source, in arcsec.
        num(integer):       The number of object that is made. This is the source number where working 
                            with the positive simulated data.
        survey(str):        Name of survey (as defined by LensPop), and is set to default of DESc. 
                            "DESc" corresponds to optimally stacked DES images.
    Returns: 
       Saved images and psf images in each band g, r, and i for the source number as well as saving these fits images.
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

def addSky(num):
    """
    Adds the DESSky images to the positive noiseless images made in this python file, 
    to make them more realistic with noise from real DES images. 
    This is saved to 'PositiveWithDESSky/%s/%s_posSky_%s.fits'%(num,num,band).

    Args:
        num(integer):   This is the source number of the positively simulated data.   
    Returns:
        Save images under 'PositiveWithDESSky/num/', with the _posSky_band.fits path names.
        This is to ensure that the simulated images have background noise, so that the positive 
        images are realistic. 
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
    This is to normalise the g, r, and i PositiveWithDESSky images that 
    were made by adding the background sky to the noiseless positively 
    simulated images. The g, r, and i normalised images are then used to create
    a RGB composite images. 
    
    Args:
        num(integer):   This is the source number of the positively simulated data.   
    Returns:
        Saves normalised images with the wcs as headers. 
        These normalised images are saved under 'PositiveWithDESSky/num/'.
        The rgb composite images are created and saved under 'PositiveWithDESSky/num/'.
    """
    paths = {}
    paths['iImg'] = glob.glob('%s/%s/%s_posSky_i.fits' % (base_dir, num, num))[0]
    paths['rImg'] = glob.glob('%s/%s/%s_posSky_r.fits' % (base_dir, num, num))[0]
    paths['gImg'] = glob.glob('%s/%s/%s_posSky_g.fits' % (base_dir, num, num))[0]

    rgbDict={}
    wcs=None
    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'Img']) as image:
            im = image[0].data
            normImage = (im-im.mean())/np.std(im)
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

numStart = int(sys.argv[1]) # number of objects selected
numEnd = int(sys.argv[2])
sourceRandomTable, lensRandomTable = cutCosmosTable(cosmos)

for num in range(numStart, numEnd):
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
    addSky(num)
    normalise(num)
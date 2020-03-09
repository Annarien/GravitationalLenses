"""

Example of how to make an RGB image

"""

import os
import sys
import astropy.io.fits as pyfits
import glob
from astLib import *
import numpy as np
import pylab as plt
import IPython

# Choose which set of images to use
# If the non-normed set is chosen, we normalise here on the fly
normed=True

if normed == False:
    filesList=glob.glob("3/3_posSky_?.fits")
    outFileName="norm.png"
else:
    filesList=glob.glob("3/*_norm.fits")
    outFileName="not-normed.png"
rgbDict={}
wcs=None
for f in filesList:
    with pyfits.open(f) as img:
        d=img[0].data
        # NOTE: If the header had the band as a keyword, we wouldn't need to get the band from the filename...
        if normed == True:
            band=os.path.split(f)[-1].split("posSky_")[1].split("_norm")[0]
        else:
            band=os.path.split(f)[-1].split(".fits")[0][-1]
            d=(d-d.mean())/np.std(d)    # Normalise on the fly
        if wcs is None:
            wcs=astWCS.WCS(img[0].header, mode = 'pyfits')
        rgbDict[band]=d

minCut, maxCut=-1, 3
cutLevels=[[minCut, maxCut],[minCut, maxCut],[minCut, maxCut]]
plt.figure(figsize=(10,10))
p=astPlots.ImagePlot([rgbDict['i'], rgbDict['r'], rgbDict['g']], wcs, 
                     cutLevels = cutLevels, axesLabels = None, 
                     axesFontSize=26.0, axes = [0, 0, 1, 1])
plt.savefig(outFileName)


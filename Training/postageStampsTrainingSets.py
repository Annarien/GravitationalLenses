# getting postage stamps for the Training Sets to the same size as the models

import os
import astropy.io.fits as pyfits
import random
import wget
import astropy.table as atpy 
import glob

from astLib import astImages
from bs4 import BeautifulSoup

def randomXY():
    """ 
    This function creates a random x,y, coordinates that is seen in the g, r, i images DES images.
    The x,y coordinates are the same for all bands of that source.
    This has to be within the image, and not outside.

    Args:
        x(integer) = random integer from 0 to 9900, since that is the width of a DES image is 10000.
        y(integer) = random integer from 0 to 9900, since that is the height of a DES image is 10000. 

    Return:
        x, y coordinates, that are random and will be used in the image. 
    """

    x = random.randint(0, 100)
    y = random.randint(0, 100)
    print("x: " + str(x))
    print("y: " + str(y))

    return (x, y)

def makingClips(num, x, y, base_dir = 'TrainingSet'):
    """
    This is the function which makes a folder containing clipped images for the sources that are used to check the simulation.

    This function opens the .fits file in the g, r, i bands of each source and then clips the .fits file 
    or image so that a catalogue of images is created. This is clipped for testing purposes in (x, y,clipSizePix)=(0,0,100).
    In general the clipSize of the pixels is 100*100 to all images simulated, and not is the same size.

    Args:
        gTrain(.fits file) = the g band .fits image for the Training Set file.
        rTrain(.fits file) = the r band .fits image for the Training Set file.
        iTrain(.fits file) = the i band .fits image for the Training Set file.
        gClipped(numpy array) = clipped image of gTrain
        rclipped(numpy array) = clipped image of rTrain
        iclipped(numpy array) = clipped image of iTrain

    Returns:
        gClipped,rclipped and iclipped
    """
    # opening g .fits images of the training set
    gTrainPath = '%s/%s/%s_image_g_SDSS.fits'%(base_dir,num,num)
    with pyfits.open(gTrainPath) as gTrain:
        gClipped = astImages.clipImageSectionPix(gTrain[0].data, x, y, [100,100])
        print(gTrain[0].data)
        print('gClipped loaded')
        astImages.saveFITS('%s.clipped.fits'%(gTrainPath), gClipped)
        #GCLIPPED HAS NO DATA, BUT gTrain[0].data has data
        print('Created gClipped at %s.clipped.fits'%(gTrainPath))

    rTrainPath = '%s/%s/%s_image_r_SDSS.fits'%(base_dir,num,num)
    with pyfits.open(rTrainPath) as rTrain:
        rClipped = astImages.clipImageSectionPix(rTrain[0].data, x, y, [100,100])
        astImages.saveFITS('%s.clipped.fits'%(rTrainPath), rClipped)
        print(rTrain[0].data)
        print('rClipped loaded')
        #GCLIPPED HAS NO DATA, BUT rTrain[0].data has data
        print('Created rClipped at %s.clipped.fits'%(rTrainPath))

    iTrainPath = '%s/%s/%s_image_i_SDSS.fits'%(base_dir,num,num)
    with pyfits.open(iTrainPath) as iTrain:
        iClipped = astImages.clipImageSectionPix(iTrain[0].data, x, y, [100,100])
        astImages.saveFITS('%s.clipped.fits'%(iTrainPath), iClipped)
        print(iTrain[0].data)
        print('iClipped loaded')
        #GCLIPPED HAS NO DATA, BUT iTrain[0].data has data
        print('Created iClipped at %s.clipped.fits'%(iTrainPath)) 
    print()
    return (gClipped,rClipped, iClipped)
    
def normaliseTrain(num, gClipped, rClipped, iClipped, base_dir = 'TrainingSet'):
    
    # normalising the TrainingSet of the clipped images.
    clipMinMax = [0,1]
    gTrainPath = '%s/%s/%s_image_g_SDSS.fits'%(base_dir,num,num)
    rTrainPath = '%s/%s/%s_image_r_SDSS.fits'%(base_dir,num,num)
    iTrainPath = '%s/%s/%s_image_i_SDSS.fits'%(base_dir,num,num)

    gNorm = astImages.normalise(gClipped,clipMinMax)
    print('gNorm:  '+ str(gNorm))
    astImages.saveFITS('%s.norm.fits'%(gTrainPath), gNorm)
    print('g is normalised and can be seen in %s'%('%s.norm.fits'%(gTrainPath)))

    rNorm = astImages.normalise(rClipped,clipMinMax)
    print('rNorm:  '+ str(rNorm))
    astImages.saveFITS('%s.norm.fits'%(rTrainPath), rNorm)
    print('r is normalised and can be seen in %s'%('%s.norm.fits'%(rTrainPath)))

    iNorm = astImages.normalise(iClipped,clipMinMax)
    print('iNorm:  '+ str(iNorm))
    astImages.saveFITS('%s.norm.fits'%(iTrainPath), iNorm)
    print('i is normalised and can be seen in %s'%('%s.norm.fits'%(iTrainPath)))
    return (gNorm, rNorm, iNorm)



#-------------------------------------------------------------------------------------------------------------
# MAIN
"""
Here we call the loadImages function which will declare the varaibles gDES, rDES, iDES.
Its is then clipped using the clipImages function.
And we write these images to a file in .fits format using writeClippedImagesToFile function.
"""

# create a for loop to go through the numbers of source
for num in range(0,2):
    x, y = randomXY()
    gClipped, rClipped, iClipped = makingClips(num, x, y)
    normaliseTrain(num, gClipped, rClipped, iClipped)

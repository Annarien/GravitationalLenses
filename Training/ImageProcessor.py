"""
This is used to create image grids, of the various data that is looked at.
The process of creating the images is formed under the function called : plotprogressNegativePositive().
The image grid of rgb.png images from the negative and positive data is formed under the function called: plotprogressNegativePositive().
To get the random rgb.png images from the negative and positive data is formed under the function called: makeRandomRGBArray().
The plotting of all the image grids is done under the function called: plotAndSaveRgbGrid()
"""

## Processing images into a grid, to view all images at the same time, to view the process taken. 
#IMPORTS
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
from astropy.io import fits
from astLib import astImages
from PIL import Image
from os import walk

# Open DES Processed WCS .fits files, and assign a variable to the g, r, i images.
def getDESProcessedWCS(num, base_dir = 'DES/DES_Processed'):
    """
    This is to open the files of the negative DES WCSClipped images for the g, r, and i bands.

    Args:
        gWCS(HDUList):      The data of the opened g band WCSClipped fits image.
        rWCS(HDUList):      The data of the opened r band WCSClipped fits image.
        iWCS(HDUList):      The data of the opened i band WCSClipped fits image.

    Returns:
        The opened gWCS, rWCS and iWCS fits images are returned.
    """
    gWCS = fits.open(glob.glob('%s/%s_*/g_WCSClipped.fits' % (base_dir, num))[0])
    rWCS = fits.open(glob.glob('%s/%s_*/r_WCSClipped.fits' % (base_dir, num))[0])
    iWCS = fits.open(glob.glob('%s/%s_*/i_WCSClipped.fits' % (base_dir, num))[0])
    return(gWCS, rWCS,iWCS) 

# Open DES Processed norm .fits files and assign a variable to the g, r, i images.
def getDESProcessedNorm(num, base_dir = 'DES/DES_Processed'):
    """
    This is to open the normalised files of the negative DES WCSClipped images for the g, r, and i bands.

    Args:
        gDESNorm(HDUList):      The data of the opened g band normalised fits image.
        rDESNorm(HDUList):      The data of the opened r band normalised fits image.
        iDESNorm(HDUList):      The data of the opened i band normalised fits image.

    Returns:
        The opened gDESNorm, rDESNorm, and iDESNorm normalised fits images are returned.
    """

    gDESNorm = fits.open(glob.glob('%s/%s_*/g_norm.fits' % (base_dir, num))[0])
    rDESNorm = fits.open(glob.glob('%s/%s_*/r_norm.fits' % (base_dir, num))[0])
    iDESNorm = fits.open(glob.glob('%s/%s_*/i_norm.fits' % (base_dir, num))[0])

    return(gDESNorm, rDESNorm, iDESNorm) 

# Open DESSky .fits files and assign a variable to the g, r, i images.
def getDESSky(num, basedir = 'DESSky'):
    """
    This is to open files of the background sky of the DES Original images for the g, r, and i bands.

    Args:
        gDESSky(HDUList):      The data of the opened g band sky fits image.
        rDESSky(HDUList):      The data of the opened r band sky fits image.
        iDESSky(HDUList):      The data of the opened i band sky fits image.

    Returns:
        The opened gDESSky, rDESSky, and iDESSky  background sky fits images are returned.

    """
    gDESSky = fits.open(glob.glob('%s/%s_g_sky.fits' % (basedir, num))[0])
    rDESSky = fits.open(glob.glob('%s/%s_r_sky.fits' % (basedir, num))[0])
    iDESSky = fits.open(glob.glob('%s/%s_i_sky.fits' % (basedir, num))[0])
    return(gDESSky, rDESSky, iDESSky)

# Open PositiveNoiseless .fits files and assign a variable to the ..._SDSS_g, r, images.
def getPosNoiseless(num, base_dir = 'PositiveNoiseless'):
    """
    This is to open files of the positively simulated images of gravitational lensing for the g, r, and i bands.

    Args:
        gPos(HDUList):      The data of the opened g band of the positively simulated fits image.
        rPos(HDUList):      The data of the opened r band of the positively simulated fits image.
        iPos(HDUList):      The data of the opened i band of the positively simulated fits image.

    Returns:
        The opened gPos, rPos, and iPos positively simulated fits images are returned.
    """

    gPos = fits.open(glob.glob('%s/%s/%s_image_g_SDSS.fits' % (base_dir, num, num))[0])
    rPos = fits.open(glob.glob('%s/%s/%s_image_r_SDSS.fits' % (base_dir, num, num))[0])
    iPos = fits.open(glob.glob('%s/%s/%s_image_i_SDSS.fits' % (base_dir, num, num))[0])
    return(gPos, rPos, iPos)

# Open PositiveWithDESSky  .fits files and assign a variable to the ...posSky_g, r, i images.
def getPosWDESSky(num, base_dir = 'PositiveWithDESSky'):
    """
    This is to open files of the positively simulated images of gravitational lensing for the g, r, and 
    i bands, that have the background sky added to them.

    Args:
        gPosSky(HDUList):      The data of the opened g band of the positively simulated with the background 
                               sky added fits images.
        rPosSky(HDUList):      The data of the opened r band of the positively simulated with the background 
                               sky added fits images.
        iPosSky(HDUList):      The data of the opened i band of the positively simulated with the background 
                               sky added fits images.

    Returns:
        The opened gPosSky, rPosSky, and iPosSky positively simulated fits images are returned.
    """

    gPosSky = fits.open(glob.glob('%s/%s/%s_posSky_g.fits' % (base_dir, num, num))[0])
    rPosSky = fits.open(glob.glob('%s/%s/%s_posSky_r.fits' % (base_dir, num, num))[0])
    iPosSky = fits.open(glob.glob('%s/%s/%s_posSky_i.fits' % (base_dir, num, num))[0])
    return(gPosSky, rPosSky, iPosSky)

# Open PositiveWithDESSky norm. fits images and assign a variable to the ...posSky_g, r, i_norm images.
def getPosWDESSkyNorm(num, base_dir = 'PositiveWithDESSky'):
    """
    This is to open files of the normalised positively simulated images of gravitational lensing for the g, r, and 
    i bands, that have the background sky added to them.

    Args:
        gPosSkyNorm(HDUList):      The data of the opened g band of the normalised positively simulated with 
                                   the background sky added fits images.
        rPosSkyNorm(HDUList):      The data of the opened r band of the normalised positively simulated with 
                                   the background sky added fits images.
        iPosSkyNorm(HDUList):      The data of the opened i band of the normalised positively simulated with 
                                   the background sky added fits images.

    Returns:
        The opened gPosSkyNorm, rPosSkyNorm, and iPosSkyNorm, the normalised positively simulated fits images are returned.
    """

    gPosSkyNorm = fits.open(glob.glob('%s/%s/%s_g_norm.fits' % (base_dir, num, num))[0])
    rPosSkyNorm = fits.open(glob.glob('%s/%s/%s_r_norm.fits' % (base_dir, num, num))[0])
    iPosSkyNorm = fits.open(glob.glob('%s/%s/%s_i_norm.fits' % (base_dir, num, num))[0])
    return(gPosSkyNorm, rPosSkyNorm, iPosSkyNorm)

def getNumOrRowsForGrid(numOfColsForRgbGrid, arrayRGB): #Give me a description please!!!

    lenRGB = len(arrayRGB)
    numOfRowsForRgbGrid = (lenRGB / numOfColsForRgbGrid)
    if lenRGB % numOfColsForRgbGrid != 0:
        numOfRowsForRgbGrid += 1

    return numOfRowsForRgbGrid


def getDESRGBPath(num):
    rgbDESPath = glob.glob('DES/DES_Processed/%s_*/rgb.png' % (num))[0]
    return (rgbDESPath)

def getKnownRGBPath(num):
    # get path of KnownRGBPath
    rgbKnown = glob.glob('KnownLenses/DES2017/%s_*/rgb.png' % (num))[0]
    
    #get header of g image so that we can get the DESJ tile name

    gBand = glob.glob('KnownLenses/DES2017/%s_*/g_WCSClipped.fits' % (num))[0]
    hdu1 = fits.open(gBand)
    desJ=hdu1[0].header['DESJ']
    tilename = hdu1[0].header['TILENAME']

    return(rgbKnown, desJ, tilename)

def makeRandomRGBArray(path):
    numCheck = int(raw_input("Enter how many random images are to be checked. "))
    randomNum = 0
    randomArray = []
    randomArrayIndex = 0
    rgbRandomArray = []
    imageTitleArray = []

    files = folders = 0
    for _, dirnames, filenames in os.walk(path):
    # ^ this idiom means "we won't be using this value"
        files += len(filenames)
        folders += len(dirnames)

    print ("{:,} files, {:,} folders".format(files, folders))

    for num in range(0, numCheck):
        randomNum = random.randint(0, folders - 1)
        while randomNum in randomArray:
            randomNum = random.randint(0, folders - 1)
        randomArray.append(randomNum)

    print ("RANDOM ARRAY: " + str (randomArray))
    for num in range(0, len(randomArray)):
        randomArrayIndex = randomArray[num]
        if path == 'PositiveWithDESSky':
            rgbRandomArray.append('%s/%s/%s_rgb.png' % (path,randomArrayIndex, randomArrayIndex))
            imageTitleArray.append(randomArrayIndex)

        elif path == 'DES/DES_Processed':
            rgbRandomArray.append(glob.glob('%s/%s_*/rgb.png'%(path, randomArrayIndex))[0])
            imageTitleArray.append(randomArrayIndex)

    return(rgbRandomArray, imageTitleArray)

def plotAndSaveRgbGrid(filepath, rgbImagePaths, imageTitleArray): #You should probably pass num in here or something like that and save many images
    lenRGB = len(rgbImagePaths)
    numOfColsForRgbGrid = 3
    numOfRowsForRgbGrid = getNumOrRowsForGrid(numOfColsForRgbGrid, rgbImagePaths)
    
    fig3, axs = plt.subplots(numOfRowsForRgbGrid, numOfColsForRgbGrid)
    rowNum = 0
    currentIndex = 0
    while (rowNum < numOfRowsForRgbGrid):
        imagesForRow = []
        imageIndex = 0
        while (imageIndex < numOfColsForRgbGrid and currentIndex < lenRGB):
            print("Image Index: " + str(imageIndex) + " CurrentIndex: " + str(currentIndex))
            imagesForRow.append(rgbImagePaths[currentIndex])
            currentIndex += 1
            imageIndex += 1
            
        for columnNum in range(0, len(imagesForRow)):
            img = Image.open(imagesForRow[columnNum])
            img.thumbnail((100, 100))
            axs[rowNum, columnNum].imshow(img)
            imageTitle = imageTitleArray[currentIndex]
            axs[rowNum,columnNum].set_title("%s" % imageTitle, fontdict = None, loc = 'center', color = 'k' )
            img.close()
        rowNum += 1

    fig3.savefig(filepath)
    plt.close(fig3)

def plotprogressNegativePositive(numberIterations):
    #Number of Images creating grids to view.
    numberIterations = int(sys.argv[1])
    rgbPosImagePaths = []
    rgbDESImagePaths = []
    imageTitleArray = []
    for num in range(0, numberIterations):

        gWCS, rWCS, iWCS = getDESProcessedWCS(num)
        gDESNorm, rDESNorm, iDESNorm = getDESProcessedNorm(num)
        gDESSky, rDESSky, iDESSky = getDESSky(num)
        gPos, rPos, iPos = getPosNoiseless(num)
        gPosSky, rPosSky, iPosSky = getPosWDESSky(num)
        gPosSkyNorm, rPosSkyNorm, iPosSkyNorm = getPosWDESSkyNorm(num)

        #creating grids of images
        #creating the first grid, in which the DES_Processed images are seen.   
        fig1, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(gWCS[0].data, cmap = 'gray')
        axs[0, 1].imshow(rWCS[0].data, cmap = 'gray')
        axs[0, 2].imshow(iWCS[0].data, cmap = 'gray')
        axs[1, 0].imshow(gDESNorm[0].data, cmap = 'gray')
        axs[1, 1].imshow(rDESNorm[0].data, cmap = 'gray')
        axs[1, 2].imshow(iDESNorm[0].data, cmap = 'gray')
        axs[2, 0].imshow(gDESSky[0].data, cmap = 'gray')
        axs[2, 1].imshow(rDESSky[0].data, cmap = 'gray')
        axs[2, 2].imshow(iDESSky[0].data, cmap = 'gray')

        pathToPos = 'PositiveWithDESSky/'
        pathToNeg = 'DES/DES_Processed'

        filepath1 = (glob.glob('DES/DES_Processed/%s_*' % (num)))[0]
        fig1.savefig("%s/DES_Processed_Grid.png"%(filepath1))
        plt.close(fig1)
        
        gWCS.close()
        rWCS.close()
        iWCS.close()
        gDESNorm.close()
        rDESNorm.close()
        iDESNorm.close()
        gDESSky.close()
        rDESSky.close()
        iDESSky.close()

        # creating the second grid, in which the Simulated images are seen.
        fig2, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(gPos[0].data, cmap = 'gray')
        axs[0, 1].imshow(rPos[0].data, cmap = 'gray')
        axs[0, 2].imshow(iPos[0].data, cmap = 'gray')
        axs[1, 0].imshow(gPosSky[0].data, cmap = 'gray')
        axs[1, 1].imshow(rPosSky[0].data, cmap = 'gray')
        axs[1, 2].imshow(iPosSky[0].data, cmap = 'gray')
        axs[2, 0].imshow(gPosSkyNorm[0].data, cmap = 'gray')
        axs[2, 1].imshow(rPosSkyNorm[0].data, cmap = 'gray')
        axs[2, 2].imshow(iPosSkyNorm[0].data, cmap = 'gray')

        filepath2 = ("%s/%s/%s_posSky_ImageGrid.png" % ('PositiveWithDESSky', num, num))
        fig2.savefig(filepath2)
        plt.close(fig2)

        # closing images to save RAM
        gPos.close()
        rPos.close()
        iPos.close()
        gPosSky.close()
        rPosSky.close()
        iPosSky.close()
        gPosSkyNorm.close()
        rPosSkyNorm.close()
        iPosSkyNorm.close()

        rgbPosImagePaths.append('PositiveWithDESSky/%s/%s_rgb.png' % (num, num))
        rgbDESImagePaths.append(getDESRGBPath(num))
        imageTitle = '%s' % (num)
        imageTitleArray.append(imageTitle)


    filepath3 = "PositiveWithDESSky/PositiveWithDESSky_RGB_ImageGrid.png"
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
    plotAndSaveRgbGrid(filepath3, rgbPosImagePaths, imageTitleArray)

    # creating the rgb grid for the DES Images
    filepath4 = "DES/DES_RGB_ImageGrid.png"
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
    plotAndSaveRgbGrid(filepath4, rgbDESImagePaths, imageTitleArray)

def plotKnownLenses():
    numOfKnownCheck = 0
    numOfKnownCheck = raw_input(" Please insert a number to indicate how many images you would like to check from Known Lenses. ")
    rgbKnownImagePaths = []
    imageTitleArray = []
    for num in range(0, int(numOfKnownCheck)):
        rgbKnown, desJ, tileName = getKnownRGBPath(num)
        rgbKnownImagePaths.append(rgbKnown)
        imageTitle = '%s/%s' % (num,desJ)
        imageTitleArray.append(imageTitle)
        
    
    filepath5 = "KnownLenses/DES2017_RGB_ImageGrid.png"
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
    plotAndSaveRgbGrid(filepath5, rgbKnownImagePaths, imageTitleArray)

# ___________________________________________________________________________________________________________________________________________
# MAIN 
#Number of Images creating grids to view.
numberIterations = int(sys.argv[1])

# plot progress and rgb images of negative and positive images 
plotprogressNegativePositive(numberIterations)

# Get Random RGB images from PositiveWithDESSky
path = 'PositiveWithDESSky'
filepath6 = "PositiveWithDESSky/randomRGB_ImageGrid.png"
rgbRandom, imageTitleArray = makeRandomRGBArray(path)
plotAndSaveRgbGrid(filepath6,rgbRandom, imageTitleArray)

# Get Random RGB images from NegativeDES
path = 'DES/DES_Processed'
filepath7 = "DES/randomRGB_ImageGrid.png"
rgbRandom, imageTitleArray = makeRandomRGBArray(path)
# plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
plotAndSaveRgbGrid(filepath7, rgbRandom, imageTitleArray)

# plot KnownLenses rgb images
plotKnownLenses()
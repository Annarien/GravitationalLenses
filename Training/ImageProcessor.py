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
    This is to open the files of the negative DES WCSClipped images for the g, r, and i bands,
    which have been clipped using the World Coordinate Systems (WCS).

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i .fits 
                            images of the DES Processed images which have been clipped using the WCS coordinates. 
                            The file names are in the form 'DES/DES_Processed/num_source/band_WCSClipped.fits'. 
        base_dir(string):   This is the top directory path of the background sky of the DES Original images. 
                            This is set to a default of 'DES/DES_Processed'.
    Returns:
        gWCS(HDUList):      The data of the opened g band WCSClipped fits image.
        rWCS(HDUList):      The data of the opened r band WCSClipped fits image.
        iWCS(HDUList):      The data of the opened i band WCSClipped fits image.
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
        num(integer):       This is the number of the source, which corresponds to the g, r, and i .fits 
                            images of the DES Processed images which have been normalised. 
                            The file names are in the form 'DES/DES_Processed/num_source/band_norm.fits'. 
        base_dir(string):   This is the top directory path of the background sky of the DES Original images. 
                            This is set to a default of 'DES/DES_Processed'.
    Returns:
        The images that are returned is the WCSClipped image normalised.
        gDESNorm(HDUList):  The data of the opened g band normalised .fits image.
        rDESNorm(HDUList):  The data of the opened r band normalised .fits image.
        iDESNorm(HDUList):  The data of the opened i band normalised .fits image.
    """

    gDESNorm = fits.open(glob.glob('%s/%s_*/g_norm.fits' % (base_dir, num))[0])
    rDESNorm = fits.open(glob.glob('%s/%s_*/r_norm.fits' % (base_dir, num))[0])
    iDESNorm = fits.open(glob.glob('%s/%s_*/i_norm.fits' % (base_dir, num))[0])

    return(gDESNorm, rDESNorm, iDESNorm) 

# Open DESSky .fits files and assign a variable to the g, r, i images.
def getDESSky(num, basedir = 'DESSky'):
    """
    This uses the num to open files of the background sky of the DES Original images for the g, r, and i bands.

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i .fits 
                            images of the background sky of the DES Original images.
                            The file names are in the form 'DESSky/num_band_sky.fits'. 
        base_dir(string):   This is the top directory path of the background sky of the DES Original images. 
                            This is set to a default of 'DESSky'.
    Returns:
        gDESSky(HDUList):   The data of the opened g band sky fits image.
        rDESSky(HDUList):   The data of the opened r band sky fits image.
        iDESSky(HDUList):   The data of the opened i band sky fits image.
    """
    gDESSky = fits.open(glob.glob('%s/%s_g_sky.fits' % (basedir, num))[0])
    rDESSky = fits.open(glob.glob('%s/%s_r_sky.fits' % (basedir, num))[0])
    iDESSky = fits.open(glob.glob('%s/%s_i_sky.fits' % (basedir, num))[0])
    return(gDESSky, rDESSky, iDESSky)

# Open PositiveNoiseless .fits files and assign a variable to the ..._SDSS_g, r, images.
def getPosNoiseless(num, base_dir = 'PositiveNoiseless'):
    """
    This is to open files of the smoothly positively simulated images of gravitational lensing for the g, r, and i bands.

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i positively 
                            simulated .fits.
                            The file names are in the form 'PositiveNoiseless/num/num_image_band_SDSS.fits'. 
        base_dir(string):   This is the top directory path of the positively simulated images. 
                            This is set to a default of 'PositiveNoiseless'.
    Returns:
        gPos(HDUList):      The data of the opened g band of the positively simulated fits image.
        rPos(HDUList):      The data of the opened r band of the positively simulated fits image.
        iPos(HDUList):      The data of the opened i band of the positively simulated fits image.
    """

    gPos = fits.open(glob.glob('%s/%s/%s_image_g_SDSS.fits' % (base_dir, num, num))[0])
    rPos = fits.open(glob.glob('%s/%s/%s_image_r_SDSS.fits' % (base_dir, num, num))[0])
    iPos = fits.open(glob.glob('%s/%s/%s_image_i_SDSS.fits' % (base_dir, num, num))[0])
    return(gPos, rPos, iPos)

# Open PositiveWithDESSky  .fits files and assign a variable to the ...posSky_g, r, i images.
def getPosWDESSky(num, base_dir = 'PositiveWithDESSky'):
    """
    The number is used to open files of the positively simulated images of gravitational lensing for the g, r, and 
    i bands, that have the background sky added to them. 

    Args:
        num(integer):           This is the number of the source, which corresponds to the g, r, and i positively 
                                simulated .fits with background sky from the original DES images added to it.
                                The file names are in the form 'PositiveWithDESSky/num_source/num_posSky_band.fits'.
        base_dir(string):       This is the top directory path of the positively simulated images with the 
                                background sky added to it. This is set to a default of 'PositiveWithDESSky'.
    Returns:
        gPosSky(HDUList):      The data of the opened g band of the positively simulated with the background 
                               sky added fits images.
        rPosSky(HDUList):      The data of the opened r band of the positively simulated with the background 
                               sky added fits images.
        iPosSky(HDUList):      The data of the opened i band of the positively simulated with the background 
                               sky added fits images.
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
        num(integer):           This is the number of the source, which corresponds to the g, r, and i norm. fits 
                                images of the sources. 
                                The file names are in the form 'PositiveWithDESSky/num_source/num_band_norm.fits'.
        base_dir(string):       This is the top directory path of the positively simulated images with the 
                                background sky added to it. This is set to a default of 'PositiveWithDESSky'.                        
    Returns:
       gPosSkyNorm(HDUList):    The data of the opened g band of the normalised positively simulated with 
                                the background sky added fits images.
        rPosSkyNorm(HDUList):   The data of the opened r band of the normalised positively simulated with 
                                the background sky added fits images.
        iPosSkyNorm(HDUList):   The data of the opened i band of the normalised positively simulated with 
                                the background sky added fits images.
    """

    gPosSkyNorm = fits.open(glob.glob('%s/%s/%s_g_norm.fits' % (base_dir, num, num))[0])
    rPosSkyNorm = fits.open(glob.glob('%s/%s/%s_r_norm.fits' % (base_dir, num, num))[0])
    iPosSkyNorm = fits.open(glob.glob('%s/%s/%s_i_norm.fits' % (base_dir, num, num))[0])
    return(gPosSkyNorm, rPosSkyNorm, iPosSkyNorm)

def getNumOrRowsForGrid(numOfColsForRgbGrid, arrayRGB):
    """
    This is to get a number of rows using a predetermined number of columns. 
    This is to ensure that the images form a grid, so that multiple rgb images can be viewed at once. 

    Args:
        lenRGB(integer):                The length of the array of RGB images that is used.
        numOfColsForRgbGrid(integer):   The number of columns using that is predetermined. 
    Returns:
        numOfRowsForRgbGrid(integer):   The number of rows that is calculated using the length divided 
                                        by the number of predetermined columns
    """

    lenRGB = len(arrayRGB)
    numOfRowsForRgbGrid = (lenRGB / numOfColsForRgbGrid)
    if lenRGB % numOfColsForRgbGrid != 0:
        numOfRowsForRgbGrid += 1

    return (numOfRowsForRgbGrid)

def getDESRGBPath(num):
    """
    Get the file path of the rgb.png image of the negative DES processed images.

    Args: 
        num(integer):           The number of the DES Processed source. 
                                The format of this is 'DES/DES_Processed/num_source/rgb.png
    Returns:
        rgbDESPath(string):     The path of the rgb.png image of the negative DES processed images.                                 The format of this is 'DES/DES_Processed/num_source/rgb.png
                                The format of this file path is 'DES/DES_Processed/num_source/rgb.png.
    """

    rgbDESPath = glob.glob('DES/DES_Processed/%s_*/rgb.png' % (num))[0]
    return (rgbDESPath)

def getKnownRGBPath(num):
    """
    The num is used to get the path of the rgb.png images of the DES2017 known lenses that have been previously 
    identified in previous studies. The tilename (from DES DR1) and the DESJ2000 name 
    (from the DES2017 paper) are also retrieved, as this is to get the correct names for each 
    image when creating the rgb image grids of these known lenses. The tilename and DESJ names 
    are retrieved from one of bands of the WCSClipped images of that source, here we will just use the g band.

    Args:
        num(integer):       This is the number that is used to find the rgb.png and the g_WCSClipped.fits files for that source in the DES2017 directory. 
                            The rgb.png files are in the form: 'KnownLenses/DES2017/num_source/rgb.png'.
                            The g_WCSClipped.fits files are in the form: 'KnownLenses/DES2017/num_source/g_WCSClipped.fits
    Returns:
        rgbKnown(string):   Provides the path name for the known lenses from DES2017 study.
        desJ(string):       Provides the DESJ2000 name of the known lenses.
        tilename(string):   Provides the DES DR1 tilename for the known lenses.
    """

    # get path of KnownRGBPath
    rgbKnown = glob.glob('KnownLenses/DES2017/%s_*/rgb.png' % (num))[0]
    
    #get header of g image so that we can get the DESJ tile name

    gBand = glob.glob('KnownLenses/DES2017/%s_*/g_WCSClipped.fits' % (num))[0]
    hdu1 = fits.open(gBand)
    desJ=hdu1[0].header['DESJ']
    tilename = hdu1[0].header['TILENAME']

    return(rgbKnown, desJ, tilename)

def makeRandomRGBArray(path, numberIterations):
    """
    This takes the root directory of rgb images that will create a list of random rgb images within the directory.
    An example of the necessary path of the root directory is 'PositiveWithDESSky' or 'DES/DES_Processed'.

    Makes an random list of the rgb.png images. 
    This creates a list of random rgb images within the chosen directory. 

    Args:
       path(string):                The path name in which the rgb images are found. 
    Returns:
        rgbRandomArray(list):       The list of random paths of rgb.png images, corresponding to the 
                                    randomArrayIndex as its sources.
        imageTitleArray(list):      The list of the numbers that correspond to the rgb.png images in the rgbRandomArray. 
    """

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

    for num in range(0, numberIterations):
        randomNum = random.randint(0, folders - 1)
        while randomNum in randomArray:
            randomNum = random.randint(0, folders - 1)
        randomArray.append(randomNum)

    print ("RANDOM ARRAY: " + str (randomArray) + " TYPE: " +str(type(randomArray)))
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
    """
    Using the image arrays (rgbImagePaths()) to make an image grid made of RGB images. 
    The title for each image is retrieved from the imageTitleArray(). 

    Args:
        filepath(string):               The file path name where the Figure of the image grid is to be saved. 
        rgbImagePaths(list):            This is the list of the rgb images, that are used when making the rgb image grids. 
        imageTitle(list):               This is the list of the names or titles of each image that is in the grid. 
                                        These names will either be the numbers of the sources or the source name, 
                                        depending on which data is being used. 
        Returns:
            This saves the Figure, which is all the indivivual rgb images placed in a grid. 
            These figures are saved in the path which is retrieved from when this function is called.                           
    """    
    lenRGB = len(rgbImagePaths)
    numOfColsForRgbGrid = 3
    numOfRowsForRgbGrid = getNumOrRowsForGrid(numOfColsForRgbGrid, rgbImagePaths)
    fig, axs = plt.subplots(numOfRowsForRgbGrid, numOfColsForRgbGrid)
    rowNum = 0
    currentIndex = 0
    imageTitleNum = 0
    while (rowNum < numOfRowsForRgbGrid):
        imagesForRow = []
        imageIndex = 0
        while (imageIndex < numOfColsForRgbGrid and currentIndex < lenRGB):
            imagesForRow.append(rgbImagePaths[currentIndex])
            currentIndex += 1
            imageIndex += 1

        for columnNum in range(0, len(imagesForRow)):
            img = Image.open(imagesForRow[columnNum])
            img.thumbnail((100, 100))
            axs[rowNum, columnNum].imshow(img)
            axs[rowNum, columnNum].axis('off')
            imageTitle = imageTitleArray[imageTitleNum]
            axs[rowNum, columnNum].set_title("%s" % imageTitle, fontdict = None, loc = 'center', color = 'k' )
            imageTitleNum += 1 
            img.close()

        if rowNum == numOfRowsForRgbGrid - 1:
            numOfEmptyGridsForRow = numOfColsForRgbGrid - len(imagesForRow)
            for emptyIndex in range(len(imagesForRow), numOfColsForRgbGrid):
                axs[rowNum, emptyIndex].axis('off')

        rowNum += 1
    
    fig.savefig(filepath)
    plt.close(fig)

def plotprogressNegativePositive(numberIterations):
    """
    This is the plotting of the progress taken, step by step of the negative and positive 
    data, placed into a grid for each source. This grid is made for each g, r, and i band, 
    seen in each column. 
    This also makes a rgb image grid of the positive image and negative images, following 
    the amount of images requested in the variable numberIterations

    The grid for the negative data is set where the DES Processed WCSClipped images is seen 
    in the first row, the DES Processed normalised images are seen in the second row, and 
    the DES Sky images are seen in the third row.
    
    The grid for the positive data is set where the positively noiseless simulated images
    are in the first row, the positively simulated data with background sky added to it is
    seen in the second row, the normalised positively simulated images are seen in the third 
    row.

    Args:
        numberIterations(integer):     This is the number of sources that will have the progress grids made for them.
                                       Not all sources are sometimes, since you just want to check the progress, 
                                       and how the images change overtime. This is also determines how many images from
                                       the positive and negative data sets to make the rgb image grids for positive and negative data.
    Returns:
        The Figure of the progress of the positive and negative data is saved. 
        The rgb image grids for the positive and negative data is saved as well.
    """
    #Number of Images creating grids to view.
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
        imageTitlePos = '%s' % (num)
        imageTitleArray.append(imageTitlePos)

    filepath3 = "PositiveWithDESSky/PositiveWithDESSky_RGB_ImageGrid.png"
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
    plotAndSaveRgbGrid(filepath3, rgbPosImagePaths, imageTitleArray)

    # creating the rgb grid for the DES Images
    filepath4 = "DES/DES_RGB_ImageGrid.png"
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
    plotAndSaveRgbGrid(filepath4, rgbDESImagePaths, imageTitleArray)

def plotKnownLenses(numberIterations):
    """
    This is the plotting of knownlenses. This has the requested amount of known lenses to look at and check. 
    This function opens the function getKnownRGBPath(), and gets the paths of the rgb images of the known 
    lenses from DES2017. These rgb images are used to create a rgb grid of the DES2017 known lenses. 
    This rgb grid is often looked at when checking the images. 
    Returns:
        This saves the figure containing the rgb image grids of the knownlenses. 
    """
    rgbKnownImagePaths = []
    imageTitleArray = []
    for num in range(0, numberIterations):
        rgbKnown, desJ, tileName = getKnownRGBPath(num)
        rgbKnownImagePaths.append(rgbKnown)
        imageTitle = '%s_%s' % (num,desJ)
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
rgbRandom, imageTitleArray = makeRandomRGBArray(path, numberIterations)
plotAndSaveRgbGrid(filepath6,rgbRandom, imageTitleArray)

# Get Random RGB images from NegativeDES
path = 'DES/DES_Processed'
filepath7 = "DES/randomRGB_ImageGrid.png"
rgbRandom, imageTitleArray = makeRandomRGBArray(path, numberIterations)
# plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved), list( paths of rgb images)))
plotAndSaveRgbGrid(filepath7, rgbRandom, imageTitleArray)

# plot KnownLenses rgb images
plotKnownLenses(numberIterations)
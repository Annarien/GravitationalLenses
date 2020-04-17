"""
Downloading the original DES images (10000 * 10000 pixels).
These images are the into 100*100 pixels are cut, using random x, y coordinates, these images are 
known as background sky/noise. The original images are clipped using the World Coordinate System, 
and are 100*100 pixels in size around stellar/astronomical objects, and these images will be referred 
to as negativeDES images. These negativeDES images are normalised, as well as composite RGB images are created.
"""

import os
import sys
import random
import wget
import astropy.table as atpy
import glob
import numpy as np
import pylab as plt
from astropy.io import fits
from astLib import astWCS
from astLib import astImages
from astLib import astPlots
from bs4 import BeautifulSoup

def loadDES(source, base_dir = 'DES/DES_Original'):
    """
    Firstly the .fits files are downloaded from DES DR1. 
    This contains the g, r, i magnitudes as well as the RA and DEC, for each source.
    The g, r, i .fits files are downloaded for each source from the DES DR1 server.
    DownLoading the images in a folder, only containg DES original .fits files.

    Args:
        source(string):     This is the tilename given in the DR1 database, and this is name of each source.
        base_dir(string):   This is the base directory in which the folders are made.
    Returns:
        Downloads the images from DES for g, r, and i .fits files of each source. 
        These images are downloaded to 'DES/DES_Original'.
    """
    if not os.path.exists('%s'  % (base_dir)):
        os.mkdir('%s' % (base_dir))

    # For each tile name, download the HTML, scrape it for the files and create the correct download link
    if not os.path.exists('%s/%s' % (base_dir, source)):
        os.mkdir('%s/%s'  % (base_dir, source))

    # Delete previously made file if it exists
    if os.path.exists('%s/%s/%s.html' % (base_dir, source, source)):
        os.remove('%s/%s/%s.html' % (base_dir, source, source))

    # Download HTML file containing all files in directory
    url = 'http://desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/' + source + '/'

    wget.download(url, '%s/%s/%s.html' % (base_dir, source, source))

    with open('%s/%s/%s.html' % (base_dir, source, source), 'r') as content_file:
        content = content_file.read()
        print()
        soup = BeautifulSoup(content, 'html.parser')
        for row in soup.find_all('tr'):
            for col in row.find_all('td'):
                if col.text.find("r.fits.fz") != -1 or col.text.find("i.fits.fz") != -1 or col.text.find("g.fits.fz") != -1:
                    if not os.path.exists('%s/%s/%s' % (base_dir, source, col.text)):
                        print('Downloading: ' + url + col.text)
                        wget.download(url + col.text, '%s/%s/%s' % (base_dir, source, col.text))
                        print()
                    else:
                        print('%s/%s/%s already downloaded...' % (base_dir, source, col.text))
                        print()
        print()

def randomXY(source, base_dir = 'DES/DES_Original'):
    """
    This gets random x, y coordinates in the original g band of DES images. 
    Only one band is used to get these coordinates, as the same random 
    coordinates are needed in all bands. This also ensure that images are 
    100*100 pixels in size, and all pixels are within the images. 

    Args:
        source(string):     This is the tilename of the DES DR1 images, used for each object.
        base_dir(string):   This is the root directory that contains the original DES images.
    Returns:
        xRandom(int):       The random x coordinate, within the DES Original g band image.
        yRandom(int):       The random y coordinate, within the DES Original g band image. 
    """

    with fits.open(glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0]) as bandDES:
        inHDUList = bandDES[1].header
        print ('NAXIS1: ' + str(inHDUList['NAXIS1']))
        print ('NAXIS2: ' + str(inHDUList['NAXIS2']))
        
        xMax = inHDUList['NAXIS1']
        yMax = inHDUList['NAXIS2']
        xRandom = random.randint(100, xMax - 100)
        yRandom = random.randint(100, yMax - 100)

        print("x: " + str(xRandom))
        print("y: " + str(yRandom))
        return (xRandom, yRandom)

def randomSkyClips(num, source, ra, dec, gmag, rmag, imag, base_dir = 'DES/DES_Original'):
    """
    Clipping of the g, r, and i DES Original fits images, to create a 100*100 pixel sized image of noise/sky. 
    
    Args:
        num(integer):       Number identifying the particular processed negative folder and files is being used.
        source(string):     This is the tilename of the clipped sky images.
        ra(float):          The right ascension of the clipped original image from DES.
        dec(float):         The declination of the clipped original image from DES.
        gmag(float):        The magnitude of the g band of the orignal images from DES.
        rmag(float):        The magnitude of the r band of the orignal images from DES.
        image(float):       The magnitude of the i band of the orignal images from DES.
        base_dir(string):   The root directory of the orignal DES images, which are
                            used to be clipped into the sky images. 
    Returns:
        Saves these randomly clipped 100*100 g, r, and i images to the folder called 
        'DESSky/', and saves the revelant headers, for later use or to check these 
        astronomical parameters.   
    """

    paths = {}
    paths['gBandPath'] = glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0]
    paths['rBandPath'] = glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, source, source))[0]
    paths['iBandPath'] = glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, source, source))[0]

    if not os.path.exists('DESSky'):
        os.mkdir('DESSky')
    
    madeSky = False
    clippedSky = {}
    while madeSky == False:
        allImagesValid = True
        x, y = randomXY(source)
        for band in ['g', 'r', 'i']:
            with fits.open(paths[band + 'BandPath']) as bandDES:
                bandSky = astImages.clipImageSectionPix(bandDES[1].data, x, y, [100, 100])
                
                if np.any(bandSky) == 0:
                    allImagesValid = False
                    print("randomly-chosen postage stamp position contained zero values - trying again ...")
                    break
                else:
                    clippedSky[band] = bandSky

        if allImagesValid == True:
            madeSky = True
    
    for band in clippedSky.keys():
        header = fits.Header()
        header['TILENAME'] = source
        header['RA'] = ra
        header['DEC'] = dec
        header['G_MAG'] = gmag
        header['I_MAG'] = imag
        header['R_MAG'] = rmag
        fits.writeto('DESSky/%i_%s_sky.fits' % (num, band), clippedSky[band], header = header, overwrite = True)

def clipWCS(source, num, gmag, rmag, imag, ra, dec, base_dir = 'DES/DES_Original', base_new = 'DES/DES_Processed'):
    """
    Clips the g, r, i original .fits images for each source from DES to have 100*100 pixel size or 0.0073125*0.007315 degrees.
    The WCS coordinates are used, to maintain the necessary information that may be needed in future.
    These WCSclipped images are saved at ('%s.WCSclipped.fits' % (paths[band+'BandPath']).
    The WCS images, are normalised and saved at ('%s.norm.fits' % (paths[band + 'BandPath']).

    Args:
        source(string):     This is the tilename of the original images from DES.
        num(integer):       Number identifying the particular processed negative folder and files is being used.
        gmag(float):        The magnitude of the g band of the original images from DES.
        rmag(float):        The magnitude of the r band of the original images from DES.
        image(float):       The magnitude of the i band of the original images from DES.
        ra(float):          The right ascension of the orignal images from DES.
        dec(float):         The declination of the original images from DES.
        base_dir(string):   The root directory of the orignal DES images. 
        base_new(string):   The root directory in which the WCSClipped images are saved,
                            this is defaulted to 'DES/DES_Processed'.
    Returns:
        WCSClipped (numpy array):   A numpy array of the WCSclipped, with its WCS coordinates.
        The g, r, and i WCSClipped images are saved under 'DES/DES_Processed', with the revelant
        astronomical parameters in the header of these images.
    """
    # Getting the RA and Dec of each source
    sizeWCS = [0.0073125, 0.0073125] # 100*100 pixels in degrees 
    
    paths = {}
    paths['gBandPath'] = glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0]
    paths['rBandPath'] = glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, source, source))[0]
    paths['iBandPath'] = glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, source, source))[0]

    if not os.path.exists('%s' % (base_new)):
        os.mkdir('%s' % (base_new))

    newPath = {}
    newPath = ('%s/%s_%s' % (base_new, num, source))
    if not os.path.exists(newPath):
        os.mkdir('%s/%s_%s' % (base_new, num, source))
    
    for band in ['g','r','i']:
        with fits.open(paths[band+'BandPath']) as bandDES:
            header = bandDES[1].header
            header.set('MAG_G', gmag)
            header.set('MAG_I', imag)
            header.set('MAG_R', rmag)
            header.set('RA', ra)
            header.set('DEC', dec)
            WCS=astWCS.WCS(header, mode = "pyfits") 
            WCSClipped = astImages.clipImageSectionWCS(bandDES[1].data, WCS, ra, dec, sizeWCS)
            astImages.saveFITS('%s/%s_WCSClipped.fits' % (newPath, band), WCSClipped['data'], WCS)
            print('Created %s_WCSclipped at %s/%s_WCSClipped.fits' % (band, newPath, band))

    return(WCSClipped)

def normaliseRGB(num, source, base_dir = 'DES/DES_Processed'):
    """
    This is to normalise the g, r, and i WCSClipped images and to make a rgb composite image of the three band together. 
    
    Args:
        num(integer):       Number identifying the particular processed negative folder and files is being used.
        source(string):     This is the tilename of the original images from DES.
        base_dir(string):   The root directory in which the normalised images and the rgb compostie images are saved,
                            this is defaulted to 'DES/DES_Processed'.
    Returns:
        Saves normalised images with the wcs as headers. 
        These normalised images are saved under 'DES/DES_Processed/num_source/'.
        The rgb composite images are created and saved under 'DES/DES_Processed/num_source/'.
    """
    paths = {}
    paths['iBandPath'] = glob.glob('%s/%s_%s/i_WCSClipped.fits' % (base_dir, num,source))[0]
    paths['rBandPath'] = glob.glob('%s/%s_%s/r_WCSClipped.fits' % (base_dir, num,source))[0]   
    paths['gBandPath'] = glob.glob('%s/%s_%s/g_WCSClipped.fits' % (base_dir, num,source))[0]   
    rgbDict = {}
    wcs = None

    for band in ['g', 'r', 'i']:
        with fits.open(paths[band+'BandPath']) as image:
            im = image[0].data
            normImage = (im - im.mean())/np.std(im)
            if wcs is None:
                wcs = astWCS.WCS(image[0].header, mode = 'pyfits')
            astImages.saveFITS('%s/%s_%s/%s_norm.fits' % (base_dir, num, source, band), normImage, wcs)
            rgbDict[band] = normImage
            

    minCut, maxCut = -1, 3
    cutLevels = [[minCut, maxCut], [minCut, maxCut], [minCut, maxCut]]
    plt.figure(figsize = (10, 10))
    astPlots.ImagePlot([rgbDict['i'], rgbDict['r'], rgbDict['g']],
                    wcs,
                    cutLevels = cutLevels,
                    axesLabels = None,
                    axesFontSize= 26.0,
                    axes = [0, 0, 1, 1])
    plt.savefig('%s/%s_%s/rgb.png' %(base_dir, num, source))

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
"""
Here we call the loadImages function which will declare the varaibles gDES, rDES, iDES.
Its is then clipped using the clipImages function.
And we write these images to a file in .fits format using writeClippedImagesToFile function.
""" 

tableDES = atpy.Table().read("DES/DESGalaxies_18_I_22.fits")

# ensuring there is no none numbers in the gmag, rmag, and imag in the DES table. 
# ensuring that there is no Gmag with values of 99.
for key in ['MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I']:
    tableDES = tableDES[np.isnan(tableDES[key]) == False] 

tableDES = tableDES[tableDES['MAG_AUTO_G']< 24]
lenTabDES = len(tableDES)

numStart = int(sys.argv[1])
numEnd = int(sys.argv[2])

for num in range(numStart, numEnd):
    
    tileName = tableDES['TILENAME'][num]
    print(type(tileName))
    gmag = tableDES['MAG_AUTO_G'][num]
    imag = tableDES['MAG_AUTO_I'][num]
    rmag = tableDES['MAG_AUTO_R'][num]
    ra = tableDES['RA'][num]
    dec = tableDES['DEC'][num]
    print('Gmag: ' + str(gmag))
    print('Imag: ' + str(imag))
    print('Rmag: ' + str(rmag))

    loadDES(tileName) 
    randomSkyClips(num, tileName, ra, dec, gmag, rmag, imag)  
    clipWCS(tileName, num, gmag, rmag, imag,ra, dec)
    normaliseRGB(num, tileName)
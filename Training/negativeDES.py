# getting postage stamps of DES and cutting them to the same size as the generated model 
# images as we made in positiveSet.py which is 100*100 pixels.

import os
import sys
import random
import wget
import astropy.table as atpy
import glob
import numpy as np
import img_scale
import pylab as plt
from astropy.io import fits
from astLib import *
# from astLib import astWCS
# from astLib import astImages
from bs4 import BeautifulSoup
from astropy.visualization import make_lupton_rgb

def makeInitialTable(num):
    """ 
    Writes a default of all zeroes table for g,r, i bands to include all the parameters that create the lenses and sources.
    The reason for this table is t quickly look at the images and to compare magnitudes and to look at a few sources at once from DES. 
    
    Args:
        Number(int):        Number of the Folder or source that is made.
        TileName(string):   This is the source name of the original source from DES.
        MAG_AUTO_G(float):  This is the g magnitude of the original source from DES.
        MAG_AUTO_R(float):  This is the r magnitude of the original source from DES.
        MAG_AUTO_I (float): This is the i magnitude of the original source from DES.
    
    Returns:
        tab(table):     A table is created and returned.
    """
    tab = atpy.Table()
    tab.add_column(atpy.Column( np.zeros(num), "NUM"))
    #tab.add_column(atpy.Column( np.zeros(num), "TILENAME"))
    tab.add_column(atpy.Column( np.zeros(num), "MAG_AUTO_G"))
    tab.add_column(atpy.Column( np.zeros(num), "MAG_AUTO_R"))
    tab.add_column(atpy.Column( np.zeros(num), "MAG_AUTO_I"))
    tab.add_column(atpy.Column( np.zeros(num), "RA"))
    tab.add_column(atpy.Column( np.zeros(num), "DEC"))
    return(tab)

def addRowToTable(tab, num, tileName, gmag, rmag, imag, ra, dec):
    """ 
    Writes a table that includes the values or the original g, r, i magnitudes of the sources that are made.
    The reason for this table is to quickly look at the images and to compare magnitudes and to look at a few sources at once from DES. 
    
    Args:
        Number(int):        Number of the Folder or source that is made.
        TileName(string):   This is the source name of the original source from DES.
        MAG_AUTO_G(float):  This is the g magnitude of the original source from DES.
        MAG_AUTO_R(float):  This is the r magnitude of the original source from DES.
        MAG_AUTO_I (float): This is the i magnitude of the original source from DES.
    
    Returns:
        tab(table):     A table is created and saved as 'DES/DES_Sets.fits'
    """
    tab["NUM"][num] = num
    #tab["TILENAME"][num] = tileName
    tab["MAG_AUTO_G"][num] = gmag
    tab["MAG_AUTO_R"][num] = rmag
    tab["MAG_AUTO_I"][num] = imag
    tab["RA"][num]  =   ra
    tab["DEC"][num] =   dec
    tab.write("DES/DES_Sets.csv", overwrite = True)

def loadDES(num, source, base_dir = 'DES/DES_Original'):
    """
    Firstly the .fits file was downloaded from DES DR1. This contains the g, r, i magnitudes as well as the RA and DEC, for each source.
    Then g, r, i .fits files are downloaded for each source from the DES DR1 server.
    DownLoading the images in a folder, only containg DES original .fits files.

    Args:
        url(string):        Url for the DES survey plus each source so that the source is fetched correctly. 
        source(string):     This is the tilename given in the DR1 database, and this is name of each source.
        num(integer):       Number given to identify the order the the sources are processed in.
        base_dir(string):   This is the base directory in which the folders are made.
    
    Returns:
        Downloads the images from DES for g, r, i .fits files of each source. These images are downloaded to 'DES/DES_Original'.
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
    # This code only ever gets called for the g band, so why not just make it like this:

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
        paths (dictionary):          The path for the g, r, i .fits files for each source.
        header (header):             This is tha actual header for these images, and is adjusted to include the magnitudes of g, r, i.
        RA (float):                  This is the Right Ascension of the source retrieved from the DES_Access table.
        Dec (float):                 This is the Declination of the source retrieved from the  DEC_Access table.
        sizeWCS (list):              This is a list of (x,y) size in degrees which is 100*100 pixels.
        WCS (astWCS.WCS):            This is the WCS coordinates that are retrieved from the g, r, i . fits files.
        WCSClipped (numpy array):    Clipped image section and updated the astWCS.WCS object for the clipped image section.
                                     and the coordinates of clipped section that is within the imageData in format {'data', 'wcs', 'clippedSection'}.
        im (numpy array):            Numpy array of the WCSClipped data.
        normImage(numpy array):      Normalised numpy array which is calculated as normImage = im/np.std(im) where np.std is the standard deviation.  
    
    Returns:
        WCSClipped (numpy array):    A numpy array of the WCSclipped, with its WCS coordinates.
        normImages (numpy array):    Images of the normalisation for the WCSClipped images are saved in the PositiveWithDESSky folder.
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
#tab = makeInitialTable(numEnd)

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

    #addRowToTable(tab, num, tileName, gmag, rmag, imag, ra, dec)
    loadDES(num, tileName) 
    randomSkyClips(num, tileName, ra, dec, gmag, rmag, imag)  
    clipWCS(tileName, num, gmag, rmag, imag,ra, dec) # takes DES images and clips it with RA, and DEC
    normaliseRGB(num, tileName)
    # path = "DES/DES_Processed/%s_%s" % (num, tileName)
    # rgbImageNewForNorm(num, path)
"""
Using two different tables, which ever is chosen, to create a data set of previously identified known lenses. 
The two tables are Jacobs or DES2017.
The Jacobs known lenses are from: https://arxiv.org/abs/1811.03786 . 
The DES2017 known lenses are from: https://iopscience.iop.org/article/10.3847/1538-4365/aa8667 .

The data from the chosen table is then put into a readible format, either .fits or .xlsx files. 
This data is read, and the g, r and i DES images are downloaded corresponding to the given ra, and dec
coordinates in the respective files. These original DES images are clipped using WCS, to create 
a 100*100 pixel image. These images are then normalised and a RGB composite is made. These images are the KnownLenses.
"""
# importing modules needed
import os
import sys
import wget
import random
import astropy.table as atpy
import glob
import img_scale
import numpy as np
import pylab as plt
import pandas as pd
import xlrd
import DESTiler
from astropy.io import fits
from astLib import *
from bs4 import BeautifulSoup

def loadDES(num, tileName, base_dir = 'DES/DES_Original'):
    """
    Firstly the .fits files are downloaded from DES DR1. 
    This contains the g, r, and i magnitudes as well as the RA and DEC, for each tileName.
    Then g, r, and i .fits files are downloaded for each tileName from the DES DR1 server.
    DownLoading the images in a folder, only containg DES original .fits files.

    Args:
        url(string):        Url for the DES survey plus each tileName so that the tileName is fetched correctly. 
        tileName(string):   This is the tilename given in the DR1 database, and this is name of each tileName.
        num(integer):       Number given to identify the order the the sources are processed in.
        base_dir(string):   This is the base directory in which the folders are made.
    
    Returns:
        Downloads the images from DES for g, r, and i .fits files of each tileName. 
        These images are downloaded to 'DES/DES_Original'.
    """
    if not os.path.exists('%s'  % (base_dir)):
        os.mkdir('%s' % (base_dir))

    # For each tile name, download the HTML, scrape it for the files and create the correct download link
    if not os.path.exists('%s/%s' % (base_dir, tileName)):
        os.mkdir('%s/%s'  % (base_dir, tileName))

    # Delete previously made file if it exists
    if os.path.exists('%s/%s/%s.html' % (base_dir, tileName, tileName)):
        os.remove('%s/%s/%s.html' % (base_dir, tileName, tileName))

    # Download HTML file containing all files in directory
    url = 'http://desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/' + tileName + '/'

    wget.download(url, '%s/%s/%s.html' % (base_dir, tileName, tileName))

    with open('%s/%s/%s.html' % (base_dir, tileName, tileName), 'r') as content_file:
        content = content_file.read()
        print()
        soup = BeautifulSoup(content, 'html.parser')
        for row in soup.find_all('tr'):
            for col in row.find_all('td'):
                if col.text.find("r.fits.fz") != -1 or col.text.find("i.fits.fz") != -1 or col.text.find("g.fits.fz") != -1:
                    if not os.path.exists('%s/%s/%s' % (base_dir, tileName, col.text)):
                        print('Downloading: ' + url + col.text)
                        wget.download(url + col.text, '%s/%s/%s' % (base_dir, tileName, col.text))
                        print()
                    else:
                        print('%s/%s/%s already downloaded...' % (base_dir, tileName, col.text))
                        print()
        print()

def clipWCS(tileName, num, ra, dec, pathProcessed, desTile='', base_dir = 'DES/DES_Original'):
    """
    Clips the g, r, i original .fits images for each source from DES to have 100*100 pixel size or 0.0073125*0.007315 degrees.
    The WCS coordinates are used, to maintain the necessary information that may be needed in future.
    These WCSclipped images are saved at ('%s.WCSclipped.fits' % (paths[band+'BandPath']).
    The WCS images, are normalised and saved at ('%s.norm.fits' % (paths[band + 'BandPath']).

    Args:
        paths(dictionary):         The path for the g, r, i .fits files for each source.
        header(header):            This is tha actual header for these images, and is adjusted to include the magnitudes of g, r, i.
        ra(float):                 This is the Right Ascension of the source retrieved from the DES_Access table.
        dec(float):                This is the Declination of the source retrieved from the  DEC_Access table.
        sizeWCS(list):             This is a list of (x,y) size in degrees which is 100*100 pixels.
        WCS(astWCS.WCS):           This is the WCS coordinates that are retrieved from the g, r, i . fits files.
        WCSClipped(numpy array):   Clipped image section and updated the astWCS.WCS object for the clipped image section.
                                   and the coordinates of clipped section that is within the imageData in format {'data', 'wcs',
                                   'clippedSection'}.
    
    Returns:
        WCSClipped (numpy array):   A numpy array of the WCSclipped, with its WCS coordinates.
        The g, r, and i WCSClipped images are saved under 'KnownLense/table/num_source/', with the revelant
        astronomical parameters in the header of these images.
    """

    # Getting the RA and Dec of each source
    sizeWCS = [0.0073125, 0.0073125] # 100*100 pixels in degrees 
    
    paths = {}

    paths['gBandPath'] = glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, tileName, tileName))[0]
    paths['rBandPath'] = glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, tileName, tileName))[0]
    paths['iBandPath'] = glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, tileName, tileName))[0]

    newPath = '%s/%s_%s' % (pathProcessed, num, tileName)
    if not os.path.exists('%s' % (newPath)):
        os.mkdir('%s' % (newPath))
    
    for band in ['g','r','i']:
        with fits.open(paths[band+'BandPath']) as bandDES:
            header = bandDES[1].header
            header.set('RA', ra)
            header.set('DEC', dec)
            header.set('DESJ', desTile)
            WCS=astWCS.WCS(header, mode = "pyfits") 
            WCSClipped = astImages.clipImageSectionWCS(bandDES[1].data, WCS, ra, dec, sizeWCS)
            astImages.saveFITS('%s/%s_WCSClipped.fits' % (newPath, band), WCSClipped['data'], WCS)
            print('Created %s_WCSclipped at %s/%s_WCSClipped.fits' % (band, newPath, band))

    return(WCSClipped)

def normaliseRGB(num, source, pathProcessed):
    """
    This is to normalise the g, r, and i WCSClipped images and to make a rgb composite image of the three band together. 
    
    Args:
        paths(dictionary):      The path for the g, r, i .WCSClipped fits files for each source.  
        rgbDict(dictionary):    Dictionary containing the g,r, and i normalised images, and is to be used to create the rgb image.
        wcs(instance):          This is the World Coordinate System(WCS), set to a default of None. 
                                If it is None, then the WCS is retrieved from the header of the WCSClipped fits image. 
        normImage(numpy array): Normalised Image Array where the normalisation is calculated as (im - im.mean())/np.std(im)
        minCut(integer):        Low value of pixels.
        maxCut(integer):        High value of pixels.
        cutLevels(list):        Sets the image scaling, specified manually for the r, g, b as [[r min, rmax], [g min, g max], [b min, b max]].
        axesLabels(string):     Labels of the axes, specified as None.
        axesFontSize(float):    Font size of the axes labels.   
        
    Returns:
        Saves normalised images with the wcs as headers. These normalised images are saved under 'KnownLenses/table/num_source/'.
        The rgb composite images are created and saved under 'KnownLense/table/num_source/'.
    """

    base_dir = pathProcessed
    paths = {}
    paths['iBandPath'] = '%s/%s_%s/i_WCSClipped.fits' % (base_dir, num, source)
    paths['rBandPath'] = '%s/%s_%s/r_WCSClipped.fits' % (base_dir, num, source) 
    paths['gBandPath'] = '%s/%s_%s/g_WCSClipped.fits' % (base_dir, num, source)   

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

# ____________________________________________________________________________________________________________________
# MAIN
table = ''
path = ''
while table != 'Jacobs' and table != 'DES2017':
    table = raw_input("Which table would you like to find Known Lenses: Jacobs or DES2017?")
    print ("You have chosen the table: " + str (table))

    if table == 'Jacobs': 
        tableKnown = atpy.Table.read("KnownLenses/Jacobs_KnownLenses.fits")
        pathProcessed = 'KnownLenses/Jacobs_KnownLenses'

        lenTabKnown = len(tableKnown)
        print ("The length of the knownLenses of " + str(table)+ " is  :" + str(lenTabKnown))

        for num in range(0, lenTabKnown):
            tileName = tableKnown['TILENAME'][num]
            print(tileName)
            print(type(tileName))
            gmag = tableKnown['MAG_AUTO_G'][num]
            imag = tableKnown['MAG_AUTO_I'][num]
            rmag = tableKnown['MAG_AUTO_R'][num]
            ra = tableKnown['RA'][num]
            dec = tableKnown['DEC'][num]
            print('Gmag: ' + str(gmag))
            print('Imag: ' + str(imag))
            print('Rmag: ' + str(rmag))

            loadDES(num, tileName)
            clipWCS(tileName, num, ra, dec, pathProcessed)
            normaliseRGB(num, tileName, pathProcessed)
        break


    elif table == 'DES2017':

        loc = ("KnownLenses/DES2017.xlsx") #location of a file
        wb = xlrd.open_workbook(loc) #opening a workbook
        sheet = wb.sheet_by_index(0) 
        numOfRows = sheet.nrows
        ra = 0.0
        dec = 0.0
        
        for num in range(0, (sheet.nrows)): 
            print("Num: " + str(num))
            desTile = sheet.cell_value(num, 0).encode('utf-8')
            print("DESTILE: " + (desTile) + " TYPE: " + str(type(desTile)))
            ra = sheet.cell_value(num, 1).encode('utf-8')
            colC = sheet.cell_value(num, 2)
            decDegree = sheet.cell_value(num, 4).encode('utf-8')
            
            ra = float(ra)
            if colC == 1:
                dec = 0 - float(decDegree)
            elif colC ==0:
                dec = float(decDegree)

            print("ra: " + str(ra) + " TYPE: " + str(type(ra)))
            print("dec: " + str(dec) + " TYPE: " + str(type(dec)))

            tiler = DESTiler.DESTiler("KnownLenses/DES_DR1_TILE_INFO.csv")

            # How to get tile name
            raDeg, decDeg = ra, dec
            tileName = tiler.getTileName(raDeg, decDeg)
            print('TileName: ' + tileName)

            # How to fetch all images for tile which contains given coords
            tiler.fetchTileImages(raDeg, decDeg, num, tileName)
            
            pathProcessed = 'KnownLenses/DES2017'
            #get gmag, rmag, imag
            clipWCS(tileName, num, raDeg, decDeg, pathProcessed, desTile)
            normaliseRGB(num, tileName, pathProcessed)
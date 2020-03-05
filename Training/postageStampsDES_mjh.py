# getting postage stamps and cutting them to the same size as the generated model images as we made in TrainingSet.py which is 100*100 piels

import os
import astropy.io.fits as pyfits
import random
import wget
import astropy.table as atpy
import glob
import pandas as pd
import numpy as np
import img_scale
import pylab as plt

from PIL import Image
from astLib import astWCS
from astLib import astImages
from bs4 import BeautifulSoup

def loadDES(num, source, base_dir = 'DES/DES_Original'):
    """
    Firstly the .fits file was downloaded from DES DR1. This contains the g, r, i magnitudes as well as the RA and DEC, for each source.
    Then g, r, i .fits files are downloaded for each source from the DES DR1 server.
    DownLoading the images in a folder, only containg DES original . fits files.

    Args:
        url(string):        Url for the DES survey plus each source so that the source is fetched correctly. 
        source(string):     This is the tilename given in the DR1 database, and this is name of each source.
        num(integer):       Number given to identify the order the the sources are processed in.
        base_dir(string):   This is the base directory in which the folders are made.
    
    Returns:
        The g, r, i .fits files in the directory of each source.

    """
    if not os.path.exists('%s'  % (base_dir)):
        os.mkdir('%s' % (base_dir))

    # For each tile name, download the HTML, scrape it for the files and create the correct download link
    if not os.path.exists('%s/%s_%s'  % (base_dir, num, source)):
        os.mkdir('%s/%s_%s'  % (base_dir, num, source))

    # Delete previously made file if it exists
    if os.path.exists('%s/%s_%s/%s.html' % (base_dir, num, source, source)):
        os.remove('%s/%s_%s/%s.html' % (base_dir, num,source, source))

    # Download HTML file containing all files in directory
    url = 'http://desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/' + source + '/'

    wget.download(url, '%s/%s_%s/%s.html' % (base_dir,num, source, source))

    with open('%s/%s_%s/%s.html' % (base_dir, num, source, source), 'r') as content_file:
        content = content_file.read()
        print()
        soup = BeautifulSoup(content, 'html.parser')
        for row in soup.find_all('tr'):
            for col in row.find_all('td'):
                if col.text.find("r.fits.fz") != -1 or col.text.find("i.fits.fz") != -1 or col.text.find("g.fits.fz") != -1:
                    if not os.path.exists('%s/%s_%s/%s' % (base_dir,num, source, col.text)):
                        print('Downloading: ' + url + col.text)
                        wget.download(url + col.text, '%s/%s_%s/%s' % (base_dir,num, source, col.text))
                        print()
                    else:
                        print('%s/%s_%s/%s already downloaded...' % (base_dir,num, source, col.text))
                        print()

        print()
    
def randomXY(bandDES):
    """ 
    This function creates a random x,y, coordinates that is seen in the g, r, i images DES images.
    The x,y coordinates are the same for all bands of that source.
    This has to be within the image, and not outside.

    Args:
        x(integer): random integer from 0 to 9900, since that is the width of a DES image is 10000.
        y(integer): random integer from 0 to 9900, since that is the height of a DES image is 10000. 

    Return:
        x, y coordinates, that are random and will be used in the RandomSkyClips to create 100*100 pixels images of random sky. 
    """
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

def randomSkyClips(num, source, base_dir = 'DES/DES_Original'):
    """
    This is the function which makes a folder containing clipped images for the sources that are used to check the simulation.

    This function opens the .fits file in the g, r, i bands of each source and then clips the .fits file 
    or image so that a catalogue of images is created. This is clipped for testing purposes in (x, y,clipSizePix)=(0,0,100).
    In general the clipSize of the pixels is 100*100 to all images simulated, and not is the same size.
    This random piece of sky is saved in the DESSky Directory, and is added to the mock images in postageStampsTrainingSets.py

    Args:
        gDES(.fits file):      The g band .fits image for the source from the DES catalog.
        rDES(.fits file):      The r band .fits image for the source from the DES catalog.
        iDES(.fits file):      The i band .fits image for the source from the DES catalog.
        gClipped(numpy array): clipped image of gDES.
        rclipped(numpy array): clipped image of rDES.
        iclipped(numpy array): clipped image of iDES.

    Returns:
        gClipped,rclipped and iclipped.
    """
    
    # opening .fits images of source
    paths = {}
    paths['gBandPath'] = glob.glob('%s/%s_%s/%s*_g.fits.fz' % (base_dir, num, source, source))[0]
    paths['rBandPath'] = glob.glob('%s/%s_%s/%s*_r.fits.fz' % (base_dir, num, source, source))[0]
    paths['iBandPath'] = glob.glob('%s/%s_%s/%s*_i.fits.fz' % (base_dir, num, source, source))[0]

    if not os.path.exists('DESSky'):
        os.mkdir('DESSky')

    madeSky=False
    while madeSky == False:
        clippedSky={}
        allImagesValid=True
        for band in ['g', 'r', 'i']:
            with pyfits.open(paths[band + 'BandPath']) as bandDES:
                if band == 'g':
                    x, y=randomXY(bandDES)
                bandSky=astImages.clipImageSectionPix(bandDES[1].data, x, y, [100, 100])
                if np.any(bandSky) == 0:
                    allImagesValid=False
                    print("randomly-chosen postage stamp position contained zero values - trying again ...")
                    break
                else:
                    clippedSky[band]=bandSky
        if allImagesValid == True:
            madeSky=True
    
    for band in clippedSky.keys():
        astImages.saveFITS('DESSky/%s_%s_sky.fits' % (num, band), clippedSky[band])
        
    return clippedSky


def clipWCSAndNormalise(source, num, gmag, rmag, imag, tableDES, base_dir = 'DES/DES_Original', base_new = 'DES/DES_Processed'):
    """
    Clips the g, r, i original .fits images for each source from DES to have 100*100 pixel size or 0.0073125*0.007315 degrees.
    The WCS coordinates are used, to maintain the necessary information that may be needed in future.
    These WCSclipped images are saved at ('%s.WCSclipped.fits' % (paths[band+'BandPath']).
    The WCS images, are normalised using the WCSclipped data, and the [min,max] list of [0,1].
    These Normalised images are saved at ('%s.norm.fits' % (paths[band + 'BandPath']).

    Args:
        paths(dictionary):          The path for the g, r, i .fits files for each source.
        header(header):             This is tha actual header for these images, and is adjusted to include the magnitudes of g, r, i.
        RA(float):                  This is the Right Ascension of the source retrieved from the DES_Access table.
        Dec(float):                 This is the Declination of the source retrieved from the  DEC_Access table.
        sizeWCS(list):              This is a list of (x,y) size in degrees which is 100*100 pixels.
        WCS(astWCS.WCS):            This is the WCS coordinates that are retrieved from the g, r, i . fits files.
        WCSClipped(numpy array):    Clipped image section and updated the astWCS.WCS object for the clipped image section.
                                    and the coordinates of clipped section that is within the imageData in format {'data', 'wcs', 'clippedSection'}.
        clipMinMax(list):           Minimum value of WCSclipped array, maximum value of WCSclipped array.
        normalised(numpy array):    Normalised array of WCSclipped with a minimum value of 0, and a maximum value of 1.

    Returns:
        WCSclipped(numpy array):    A numpy array of the WCSclipped, with its WCS coordinates.
    """
    # Getting the RA and Dec of each source
    RA = tableDES['RA'][num]
    print('RA: ' + str(RA))
    Dec = tableDES['DEC'][num]
    print('Dec: ' + str(Dec))
    sizeWCS =  [0.0073125,0.0073125] # 100*100 pixels in degrees 
    
    paths = {}
    paths['gBandPath'] = glob.glob('%s/%s_%s/%s*_g.fits.fz' % (base_dir, num, source, source))[0]
    paths['rBandPath'] = glob.glob('%s/%s_%s/%s*_r.fits.fz' % (base_dir, num, source, source))[0]
    paths['iBandPath'] = glob.glob('%s/%s_%s/%s*_i.fits.fz' % (base_dir, num, source, source))[0]

    if not os.path.exists('%s' % (base_new)):
        os.mkdir('%s' % (base_new))

    newPath = {}
    newPath = ('%s/%s_%s' % (base_new, num, source))
    if not os.path.exists(newPath):
        os.mkdir('%s/%s_%s'%(base_new,num,source))

    clipMinMax  = [-2.5, 2.5]
    
    for band in ['g','r','i']:
        with pyfits.open(paths[band+'BandPath']) as bandDES:
            header = bandDES[1].header
            header.set('MAG_G',gmag)
            header.set('MAG_I',imag)
            header.set('MAG_R',rmag)
            WCS=astWCS.WCS(header, mode = "pyfits") 
            WCSClipped = astImages.clipImageSectionWCS(bandDES[1].data, WCS, RA, Dec, sizeWCS)
            astImages.saveFITS('%s/%s_WCSclipped.fits' % (newPath,band), WCSClipped['data'], WCS)
            print('Created %s_WCSclipped at %s/%s_WCSclipped.fits' % (band, newPath, band))

            im = WCSClipped['data']
            normImages = im/np.std(im)
            #normImages = im /(im.max()-im.min())

            astImages.saveFITS('%s/%s_norm.fits' % (newPath,band), normImages, WCS)
            print('Normalised %s clipped images at %s/%s'%(band, newPath, band))

    return WCSClipped

def rgbImageNewForNorm(num, directory):
    """ 
    Create Red, Green and Blue images in the training set folder under the source folder number(i).
    This is saved as a jpeg file, and a png file at ('TrainingSet/%i/RGB_%i.jpeg'%(num,num)).  

    Args: 
        i_img: this is the data that is retrieved from the sources i band . fits image
        r_img: this is the data that is retrieved from the sources r band . fits image
        g_img: this is the data that is retrieved from the sources g band . fits image
        img: this is an image created using r, g , i images to get a true colour image where the min and max scale is 0 and 1000
    """
    

    i_img = pyfits.getdata('%s/i_norm.fits' % (directory))
    r_img = pyfits.getdata('%s/r_norm.fits' % (directory))
    g_img = pyfits.getdata('%s/g_norm.fits' % (directory))

    imin,imax = i_img.mean()-0.75*i_img.std(),i_img.mean()+5*i_img.std()
    rmin,rmax = r_img.mean()-0.75*r_img.std(),r_img.mean()+5*r_img.std()
    gmin,gmax = g_img.mean()-0.75*g_img.std(),g_img.mean()+5*g_img.std()

    img = np.zeros((i_img.shape[0], i_img.shape[1], 3), dtype = float)
    img[:,:,0] = img_scale.linear(i_img, scale_min=imin, scale_max=imax)
    img[:,:,1] = img_scale.linear(r_img, scale_min=rmin, scale_max=rmax)
    img[:,:,2] = img_scale.linear(g_img, scale_min=gmin, scale_max=gmax)

    plt.clf()
    plt.imshow(img,aspect = 'equal')
    plt.savefig('%s/RGB_%i.jpeg' % (directory, num))
    plt.savefig('%s/RGB_%i.png' % (directory, num))

def makeInitialTable(num):
    tab = atpy.Table()
    tab.add_column(atpy.Column( np.zeros(num), "Number"))
    tab.add_column(atpy.Column( np.zeros(num), "TileName"))
    tab.add_column(atpy.Column( np.zeros(num), "MAG_AUTO_G"))
    tab.add_column(atpy.Column( np.zeros(num), "MAG_AUTO_R"))
    tab.add_column(atpy.Column( np.zeros(num), "MAG_AUTO_I"))
    return(tab)

def addRowToTable(tab, num, tileName, gmag, rmag, imag):
    tab["Number"][num] = num
    #tab["TileName"][num] = float(tileName)
    tab["MAG_AUTO_G"][num] = gmag
    tab["MAG_AUTO_R"][num] = rmag
    tab["MAG_AUTO_I"][num] = imag
    tab.write("DES/DES_Sets.csv", overwrite = True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
"""
Here we call the loadImages function which will declare the varaibles gDES, rDES, iDES.
Its is then clipped using the clipImages function.
And we write these images to a file in .fits format using writeClippedImagesToFile function.
""" 

tableDES = atpy.Table().read("DES/DESGalaxies_20_I_24.fits")

# ensuring there is no none numbers in the gmag, rmag, and imag in the DES table. 
# ensuring that there is no Gmag with values of 99.
for key in ['MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I']:
    tableDES = tableDES[np.isnan(tableDES[key]) == False] 

tableDES = tableDES[tableDES['MAG_AUTO_G']< 24]

lenTabDES = len(tableDES)

num = 10
tab = makeInitialTable(num)

for num in range(0, num):
    
    tileName = tableDES['TILENAME'][num]
    gmag = tableDES['MAG_AUTO_G'][num]
    imag = tableDES['MAG_AUTO_I'][num]
    rmag = tableDES['MAG_AUTO_R'][num]
    print('Gmag: ' + str(gmag))
    print('Imag: ' + str(imag))
    print('Rmag: ' + str(rmag))

    addRowToTable(tab, num, tileName, gmag, rmag, imag)

    loadDES(num, tileName) 
    clippedImages = randomSkyClips(num, tileName)  
    clippedWCS = clipWCSAndNormalise(tileName, num, gmag, rmag, imag, tableDES) # takes DES images and clips it with RA, and DEC
    directory = "DES/DES_Processed/%i_%s" % (num, tileName)
    rgbImageNewForNorm(num, directory)
    

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
from astLib import astWCS
from astLib import astImages
from bs4 import BeautifulSoup

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

def randomSkyClips(num, source, ra, dec, gmag, rmag, imag, base_dir = 'DES/DES_Original'):
    """
    This is the function which makes a folder containing clipped images for the sources that are used to check the simulation.
    These clipped images are to be added to the PositiveNoiseless images to create the PositiveWithDESSky images. 
    This is clipped as (x, y, clipSizePix)=(x, y, 100), where x, y are random coordinates.     
    The clipped images (bandSky numpy array) are created, then checked to see if there is any zero value.
    If there is a zero in the clipped image, then a new clipped image is created from the orginal image with new x, y coordinates.
    This is repeated until a clipped image is created without zero in it. 
    The 'g' band is used to clip images as there is no need to go through all the bands but only need to check one band,
    as the possibility of a zero is great in the other bands as well if there is a zero in the 'g' band.

    Args:
        paths (dictionary):       The paths to the original fits DES images.
        madeSky (boolean):        A boolean value set to 'False', and in the following while loop, clipped images are made if there is no sky.
        clippedSky (dictionary):  Dictionary of all the clipped images.
        headers (dictionary):     Dictionary of the headers of the clipped images and the original images.
        allImagesValid (boolean): A boolean value set as 'True' when there is no madeSky made, 
                                  and is set to 'False' if the images there is any zeros in the bandSky.
        bandDES (numpy array):    Numpy array of the original DES images.
        x (int):                  Interger of a random x coordinate in the original DES image received from the randomXY function.
        y (int):                  Interger of a random y coordinate in the original DES image received from the randomXY function.
        bandSky (numpy array):    Clipped image that is set at (x, y) coordinates with a dimension of 100 * 100 pixels.

    Returns:
        clippedSky (dictionary):  The images the are clipped, and this is added as the noise or 
                                  background sky of the PositiveNoiseless images creating the 
                                  PositiveWithDESSky images. These clipped images are also saved in the DESSky folder.

    """
    paths = {}
    paths['gBandPath'] = glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0]
    paths['rBandPath'] = glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, source, source))[0]
    paths['iBandPath'] = glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, source, source))[0]

    if not os.path.exists('DESSky'):
        os.mkdir('DESSky')
    
    madeSky = False
    while madeSky == False:
        clippedSky = {}
        allImagesValid = True
        for band in ['g', 'r', 'i']:
            with fits.open(paths[band + 'BandPath']) as bandDES:
                if band == 'g':
                    x, y = randomXY(bandDES)
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
        fits.writeto('DESSky/%i_%s_sky.fits' % (num, band), clippedSky[band],header = header, overwrite = True)
        
    return(clippedSky)

def clipWCSAndNormalise(source, num, gmag, rmag, imag, ra, dec, base_dir = 'DES/DES_Original', base_new = 'DES/DES_Processed'):
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
        os.mkdir('%s/%s_%s'%(base_new, num, source))
    
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

            im = WCSClipped['data']
            normImage = (im-im.mean())/np.std(im)
            
            astImages.saveFITS('%s/%s_norm.fits' % (newPath,band), normImage, WCS)
            print('Normalised %s clipped images at %s/%s' % (band, newPath, band))
    return(WCSClipped)

def rgbImageNewForNorm(num, path):
    """ 
    A universal function to create Red, Green and Blue images for DES clipped Images that is gaussian normalised.
    These images are set under in the respective folders with the source folder number(i).
    This is saved as both jpeg and png files. It is saved as both files, incase so that we may use it for 
    the article or to check if everything is ok immediatley with the images. 

    Args: 
        path (string):       This is the file path in which the rgb is made and saved.
        i_img (array):       This is the data that is retrieved from the sources i band .fits image.
        r_img (array):       This is the data that is retrieved from the sources r band . fits image.
        g_img (array):       This is the data that is retrieved from the sources g band . fits image.
        imin (float):        Minimum value for the i band, where the image mean - 0.4 times of the standard deviation. 
        imax (float):        Maximum value for the i band, where the image mean + 0.4 times of the standard deviation. 
        rmin (float):        Minimum value for the r band, where the image mean - 0.4 times of the standard deviation. 
        rmax (float):        Maximum value for the r band, where the image mean + 0.4 times of the standard deviation.
        gmin (float):        Minimum value for the g band, where the image mean - 0.4 times of the standard deviation. 
        gmax (float):        Maximum value for the g band, where the image mean + 0.4 times of the standard deviation.
        img (array):         This is an image created using r, g , i images to get a true colour image with squareroot scaling 
                             where the min and max is calculated using imin,imax,rmin,rmax,gmin,gmax. 
    Returns:
        A rgb images are saved in the form of jpeg and png as images are combined with the r, g, i images. 
    """
    i_img = fits.getdata('%s/i_norm.fits' % path)
    r_img = fits.getdata('%s/r_norm.fits' % path)
    g_img = fits.getdata('%s/g_norm.fits' % path)

    # imin, imax = i_img.mean() - 0.75 * i_img.std(), i_img.mean() + 5 * i_img.std()
    # rmin, rmax = r_img.mean() - 0.75 * r_img.std(), r_img.mean() + 5 * r_img.std()
    # gmin, gmax = g_img.mean() - 0.75 * g_img.std(), g_img.mean() + 5 * g_img.std()

    # img = np.zeros((i_img.shape[0], i_img.shape[1], 3), dtype = float)
    # img[:,:,0] = img_scale.sqrt(i_img, scale_min=imin, scale_max=imax)
    # img[:,:,1] = img_scale.sqrt(r_img, scale_min=rmin, scale_max=rmax)
    # img[:,:,2] = img_scale.sqrt(g_img, scale_min=gmin, scale_max=gmax)

    img = np.zeros((i_img.shape[0], i_img.shape[1], 3), dtype = float)
    img[:,:,0] = img_scale.log(i_img, scale_min = 0, scale_max = 10)
    img[:,:,1] = img_scale.log(r_img, scale_min = 0, scale_max = 10)
    img[:,:,2] = img_scale.log(g_img, scale_min = 0, scale_max = 10)

    plt.figure(figsize = (10, 10))
    plt.axes([0, 0, 1, 1])
    plt.imshow(img, aspect = 'equal')
    plt.savefig('%s/RGB_%i.png' % (path, num))
    plt.close()

def randomXY(bandDES):
    """ 
    This function creates a random x,y, coordinates that is seen in the g, r, i images DES images.
    The x,y coordinates are the same for all bands of that source.
    This has to be within the image, and not outside.

    Args:
        x(integer):     Random integer from 0 to 9900, since that is the width of a DES image is 10000.
        y(integer):     Random integer from 0 to 9900, since that is the height of a DES image is 10000. 

    Return:
        x,y (integers): The coordinates, that are random and will be used in the RandomSkyClips to 
                        create 100*100 pixels images of random sky. 
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

num = int(sys.argv[1])

tab = makeInitialTable(num)

for num in range(0, num):
    
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

    addRowToTable(tab, num, tileName, gmag, rmag, imag, ra, dec)
    loadDES(num, tileName) 
    randomSkyClips(num, tileName, ra, dec, gmag, rmag, imag)  
    clipWCSAndNormalise(tileName, num, gmag, rmag, imag,ra, dec) # takes DES images and clips it with RA, and DEC
    path = "DES/DES_Processed/%s_%s" % (num, tileName)
    rgbImageNewForNorm(num, path)
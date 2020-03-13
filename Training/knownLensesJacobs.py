# download DES Images from known lenses
# get coordinates from colletts known lenses.
# do cutouts of 100*100 pixels

# importing modules needed
import os
import sys
import wget
import astropy.table as atpy
import glob
import numpy as np
from astropy.io import fits
from astLib import astWCS
from astLib import astImages
from bs4 import BeautifulSoup

def loadDES(num, tileName, base_dir = 'DES/DES_Original'):
    """
    Firstly the .fits file was downloaded from DES DR1. This contains the g, r, i magnitudes as well as the RA and DEC, for each tileName.
    Then g, r, i .fits files are downloaded for each tileName from the DES DR1 server.
    DownLoading the images in a folder, only containg DES original .fits files.

    Args:
        url(string):        Url for the DES survey plus each tileName so that the tileName is fetched correctly. 
        tileName(string):     This is the tilename given in the DR1 database, and this is name of each tileName.
        num(integer):       Number given to identify the order the the sources are processed in.
        base_dir(string):   This is the base directory in which the folders are made.
    
    Returns:
        Downloads the images from DES for g, r, i .fits files of each tileName. These images are downloaded to 'DES/DES_Original'.
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

def clipWCSAndNormalise(source, num, gmag, rmag, imag, ra, dec, base_dir = 'DES/DES_Original', base_new = 'KnownLenses/Known_Processed'):
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

# ____________________________________________________________________________________________________________________
# MAIN

tableKnownDES = atpy.Table().read("KnownLenses/CollettFromDES_ObjectID.fits")
lenTabKnownDES = len(tableKnownDES)
print ("The length of the knownLenses from CollettDES:" + str(lenTabKnownDES))

for num in range(0, lenTabKnownDES):
    tileName = tableKnownDES['TILENAME'][num]
    print(type(tileName))
    gmag = tableKnownDES['MAG_AUTO_G'][num]
    imag = tableKnownDES['MAG_AUTO_I'][num]
    rmag = tableKnownDES['MAG_AUTO_R'][num]
    ra = tableKnownDES['RA'][num]
    dec = tableKnownDES['DEC'][num]
    print('Gmag: ' + str(gmag))
    print('Imag: ' + str(imag))
    print('Rmag: ' + str(rmag))

    loadDES(num, tileName)
    clipWCSAndNormalise(tileName, num, gmag, rmag, imag,ra, dec)
    
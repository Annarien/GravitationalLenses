"""

Class that can figure out which DES tile given RA, dec coords are in.

"""

import os
import sys
import numpy as np
import astropy.table as atpy
import astropy.io.fits as pyfits
from astLib import astWCS
import urllib
import time
import IPython
import wget
from bs4 import BeautifulSoup

class DESTiler:
    """A class for relating RA, dec coords to DES tiled survey geometry.
    
    """
    
    def __init__(self, tileInfoCSVPath = "DES_DR1_TILE_INFO.csv"):
        
        self.WCSTabPath=tileInfoCSVPath
        t0=time.time()
        self.setUpWCSDict()
        t1=time.time()
        print("... WCS set-up took %.3f sec ..." % (t1-t0))
    
    
    def setUpWCSDict(self):
        """Sets-up WCS info, needed for fetching images. This is slow (~30 sec) if the survey is large,
        so don't do this lightly.
        
        """
        
        # Add some extra columns to speed up searching
        self.tileTab=atpy.Table().read(self.WCSTabPath)        
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'RAMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'RAMax'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'decMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'decMax'))
        self.WCSDict={}
        keyWordsToGet=['NAXIS', 'NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 
                        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2']
        for row in self.tileTab:
            newHead=pyfits.Header()
            for key in keyWordsToGet:
                if key in self.tileTab.keys():
                    newHead[key]=row[key]
            # Defaults if missing (needed for e.g. DES)
            if 'NAXIS' not in newHead.keys():
                newHead['NAXIS']=2
            if 'CUNIT1' not in newHead.keys():
                newHead['CUNIT1']='DEG'
            if 'CUNIT2' not in newHead.keys():
                newHead['CUNIT2']='DEG'
            self.WCSDict[row['TILENAME']]=astWCS.WCS(newHead.copy(), mode = 'pyfits')  
            ra0, dec0=self.WCSDict[row['TILENAME']].pix2wcs(0, 0)
            ra1, dec1=self.WCSDict[row['TILENAME']].pix2wcs(row['NAXIS1'], row['NAXIS2'])
            if ra1 > ra0:
                ra1=-(360-ra1)
            row['RAMin']=min([ra0, ra1])
            row['RAMax']=max([ra0, ra1])
            row['decMin']=min([dec0, dec1])
            row['decMax']=max([dec0, dec1])
    
    
    def getTileName(self, RADeg, decDeg):
        """Returns the DES TILENAME in which the given coordinates are found. Returns None if the coords
        are not in the DES footprint.
        
        """
        raMask=np.logical_and(np.greater_equal(RADeg, self.tileTab['RAMin']), 
                              np.less(RADeg, self.tileTab['RAMax']))
        decMask=np.logical_and(np.greater_equal(decDeg, self.tileTab['decMin']), 
                               np.less(decDeg, self.tileTab['decMax']))
        tileMask=np.logical_and(raMask, decMask)
        if tileMask.sum() == 0:
            return None
        else:
            return self.tileTab[tileMask]['TILENAME'][0] 


    def fetchTileImages(self, RADeg, decDeg, num, tileName, base_dir = 'DES/DES_Original', bands = ['g', 'r', 'i'], refetch = False):
        """Fetches DES FITS images for the tile in which the given coords are found. 
        Output is stored under outDir.
                
        """
        
        # Inside footprint check
        raMask=np.logical_and(np.greater_equal(RADeg, self.tileTab['RAMin']), 
                              np.less(RADeg, self.tileTab['RAMax']))
        decMask=np.logical_and(np.greater_equal(decDeg, self.tileTab['decMin']), 
                               np.less(decDeg, self.tileTab['decMax']))
        tileMask=np.logical_and(raMask, decMask)
        if tileMask.sum() == 0:
            return None
        
        if os.path.exists('%s/%s' % (base_dir, tileName)) == False:
            os.makedirs('%s/%s' % (base_dir, tileName))

        if os.path.exists('%s/%s/%s.html' % (base_dir, tileName, tileName)):
            os.remove('%s/%s/%s.html' % (base_dir, tileName, tileName))

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
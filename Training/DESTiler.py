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

    def __init__(self, tileInfoCSVPath="DES_DR1_TILE_INFO.csv"):
        """
        This calculates the time it takes to get the WCS set up. 
        
        Args: 
            tileInfoCSVPath(string):    This is path of where the DES DR1 tile information is. 
                                        This is set to "DES_DR1_TILE_INFO.csv".
        Returns:
            This outputs the amount of time it took to set up the WCS.
        """
        self.WCSTabPath = tileInfoCSVPath
        t0 = time.time()
        self.setUpWCSDict()
        t1 = time.time()
        print("... WCS set-up took %.3f sec ..." % (t1 - t0))

    def setUpWCSDict(self):
        """
        Sets-up WCS info, needed for fetching images. This is slow (~30 sec) if the survey is large,
        so don't do this lightly.
        """

        # Add some extra columns to speed up searching
        self.tileTab = atpy.Table().read(self.WCSTabPath)
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'RAMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'RAMax'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'decMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'decMax'))
        self.WCSDict = {}
        keyWordsToGet = ['NAXIS', 'NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                         'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2']
        for row in self.tileTab:
            newHead = pyfits.Header()
            for key in keyWordsToGet:
                if key in self.tileTab.keys():
                    newHead[key] = row[key]
            # Defaults if missing (needed for e.g. DES)
            if 'NAXIS' not in newHead.keys():
                newHead['NAXIS'] = 2
            if 'CUNIT1' not in newHead.keys():
                newHead['CUNIT1'] = 'DEG'
            if 'CUNIT2' not in newHead.keys():
                newHead['CUNIT2'] = 'DEG'
            self.WCSDict[row['TILENAME']] = astWCS.WCS(newHead.copy(), mode='pyfits')
            ra0, dec0 = self.WCSDict[row['TILENAME']].pix2wcs(0, 0)
            ra1, dec1 = self.WCSDict[row['TILENAME']].pix2wcs(row['NAXIS1'], row['NAXIS2'])
            if ra1 > ra0:
                ra1 = -(360 - ra1)
            row['RAMin'] = min([ra0, ra1])
            row['RAMax'] = max([ra0, ra1])
            row['decMin'] = min([dec0, dec1])
            row['decMax'] = max([dec0, dec1])

    def getTileName(self, RADeg, decDeg):
        """
        Gets the tilename from DES in which the given ra and dec coordinates. 

        Args:
            RADeg(float):   This is the given right ascension of the object. 
            decDeg(float):  This is the given declination of the object.
        Returns:
            Returns the tilename from DES in which the given ra and dec coordinates are found.
            If the coordinates arent in the DES footprint (the DES DR1 tile information), then 'None'
            will be returned.
        """
        raMask = np.logical_and(np.greater_equal(RADeg, self.tileTab['RAMin']),
                                np.less(RADeg, self.tileTab['RAMax']))
        decMask = np.logical_and(np.greater_equal(decDeg, self.tileTab['decMin']),
                                 np.less(decDeg, self.tileTab['decMax']))

        # print("RAMin: " + str(self.tileTab['RAMin']) + " TYPE: " + str(type(self.tileTab['RAMin'])))
        # print("RAMax: " + str(self.tileTab['RAMax']) + " TYPE: " + str(type(self.tileTab['RAMax'])))
        # print("decMin: " + str(self.tileTab['decMin']) + " TYPE: " + str(type(self.tileTab['decMin'])))
        # print("decMax: " + str(self.tileTab['decMax']) + " TYPE: " + str(type(self.tileTab['decMax'])))
        # print("raMask: " + str(raMask) + " TYPE: " + str(type(raMask)))
        # print("decMask: " + str(decMask) + "TYPE: " +str(type(decMask)))

        tileMask = np.logical_and(raMask, decMask)
        if tileMask.sum() == 0:
            return None
        else:
            return self.tileTab[tileMask]['TILENAME'][0]

    def fetchTileImages(self, RADeg, decDeg, tileName, base_dir='DES/DES_Original'):
        """
        If the ra, and dec are found in an DES info table that tile name is retrieved. 
        The g, r, and i .fits images from the DES DR1, are downloaded according to the tile name.

        Args:
            RADeg(float):   This is the given right ascension of the object. 
            decDeg(float):  This is the given declination of the object.
            tileName(string):   This is the tile name of the source from the DES info table.
            base_dir(string):   This is the root directory in which the original DES images are downloaded.
        Returns:
            Downloads the images from DES for g, r, and i .fits files of each source which contain the 
            given ra and dec coordinates. These images are downloaded to 'DES/DES_Original'.
        """

        # Inside footprint check
        raMask = np.logical_and(np.greater_equal(RADeg, self.tileTab['RAMin']),
                                np.less(RADeg, self.tileTab['RAMax']))
        decMask = np.logical_and(np.greater_equal(decDeg, self.tileTab['decMin']),
                                 np.less(decDeg, self.tileTab['decMax']))
        tileMask = np.logical_and(raMask, decMask)
        if tileMask.sum() == 0:
            return None

        if not os.path.exists('%s/%s' % (base_dir, tileName)):
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
                    if col.text.find("r.fits.fz") != -1 or col.text.find("i.fits.fz") != -1 or col.text.find(
                            "g.fits.fz") != -1:
                        if not os.path.exists('%s/%s/%s' % (base_dir, tileName, col.text)):
                            print('Downloading: ' + url + col.text)
                            wget.download(url + col.text, '%s/%s/%s' % (base_dir, tileName, col.text))
                            print()
                        else:
                            print('%s/%s/%s already downloaded...' % (base_dir, tileName, col.text))
                            print()
            print()

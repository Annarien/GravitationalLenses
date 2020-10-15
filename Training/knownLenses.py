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
import glob
# importing modules needed
import os

import DESTiler
import astropy.table as atpy
import xlrd
from astLib import *
from astropy.io import fits
from negativeDESUtils import loadDES, normaliseRGB


def clipWCS(tile_name, num, ra, dec, path_processed, des_tile='', base_dir='DES/DES_Original'):
    """
    Clips the g, r, i original .fits images for each source from DES to have 100*100 pixel size
    or 0.0073125*0.007315 degrees. The wcs coordinates are used, to maintain the necessary
    information that may be needed in future. These WCSclipped images are saved at 
    ('%s.WCSclipped.fits' % (paths[band+'BandPath']). The wcs images, are normalised and
    saved at ('%s.norm.fits' % (paths[band + 'BandPath']).

    Args:
        tilename(string):          The tilename of the DES image from DES DR1. This is the often referred to as the 
                                   source name of the image. 
        num(integer):              This is the number of the source that is to be processed.  
        ra(float):                 This is the Right Ascension of the source retrieved from the DES_Access table.
        dec(float):                This is the Declination of the source retrieved from the  DEC_Access table.
        path_processed(string):     This is the path of directory in which the wcsclipped images are to be saved.
        des_tile(string):           This is the DESJ2000 name given for these sources in the DES2017 paper.
        base_dir(string):          This is the root directory of the DES Original images, for each source 
                                   which are clipped in this clipWCS function.
    Saves:
        wcs_clipped (numpy array):   A numpy array of the WCSclipped, with its wcs coordinates.
                                    The g, r, and i wcs_clipped images are saved under '
                                    KnownLense/table/num_source/', with the revelant astronomical
                                    parameters in the header of these images.
    """

    # Getting the RA and Dec of each source
    size_wcs = [0.0073125, 0.0073125]  # 100*100 pixels in degrees

    paths = {'gBandPath': glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, tile_name, tile_name))[0],
             'rBandPath': glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, tile_name, tile_name))[0],
             'iBandPath': glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, tile_name, tile_name))[0]}

    if not os.path.exists('%s' % path_processed):
        os.mkdir('%s' % path_processed)

    new_path = '%s/%s_%s' % (path_processed, num, tile_name)
    if not os.path.exists('%s' % new_path):
        os.mkdir('%s' % new_path)

    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'BandPath']) as bandDES:
            header = bandDES[1].header
            header.set('RA', ra)
            header.set('DEC', dec)
            header.set('DESJ', des_tile)
            wcs = astWCS.WCS(header, mode="pyfits")
            wcs_clipped = astImages.clipImageSectionWCS(bandDES[1].data, wcs, ra, dec, size_wcs)
            astImages.saveFITS('%s/%s_WCSClipped.fits' % (new_path, band), wcs_clipped['data'], wcs)
            print('Created %s_WCSclipped at %s/%s_WCSClipped.fits' % (band, new_path, band))

# ____________________________________________________________________________________________________________________
# MAIN
table = ''
path = ''
while table != 'Jacobs' and table != 'DES2017':
    table = raw_input("Which table would you like to find Known Lenses: Jacobs or DES2017?")
    print ("You have chosen the table: " + str(table))

    if table == 'Jacobs':
        table_known = atpy.Table.read("KnownLenses/Jacobs_KnownLenses.fits")
        path_processed = 'KnownLenses/Jacobs_KnownLenses'

        len_table_known = len(table_known)
        print ("The length of the knownLenses of " + str(table) + " is  :" + str(len_table_known))

        for num in range(0, len_table_known):
            tile_name = table_known['TILENAME'][num]
            print(tile_name)
            print(type(tile_name))
            g_mag = table_known['MAG_AUTO_G'][num]
            i_mag = table_known['MAG_AUTO_I'][num]
            r_mag = table_known['MAG_AUTO_R'][num]
            ra = table_known['RA'][num]
            dec = table_known['DEC'][num]
            print('Gmag: ' + str(g_mag))
            print('Imag: ' + str(i_mag))
            print('Rmag: ' + str(r_mag))

            loadDES(num, tile_name)
            clipWCS(tile_name, num, ra, dec, path_processed)
            normaliseRGB(num, tile_name, path_processed)
        break

    elif table == 'DES2017':

        # location_of_file = "KnownLenses/DES2017.xlsx"  # location of a file
        location_of_file = "UnseenData/Unseen_KnownLenses.xlsx"
        workbook = xlrd.open_workbook(location_of_file)  # opening a workbook
        sheet = workbook.sheet_by_index(0)
        num_of_rows = sheet.nrows
        ra = 0.0
        dec = 0.0

        for num in range(1, sheet.nrows):
            print("Num: " + str(num))
            des_tile = sheet.cell_value(num, 0).encode('utf-8')
            print("DESTILE: " + (des_tile) + " TYPE: " + str(type(des_tile)))
            ra = str(sheet.cell_value(num, 1)).encode('utf-8')
            col_c = sheet.cell_value(num, 2)  # index of whether or not dec is +ve or -ve

            dec_degree = str(sheet.cell_value(num, 4)).encode('utf-8')

            ra = float(ra)
            if col_c == 1:
                dec = 0 - float(dec_degree)
            elif col_c == 0:
                dec = float(dec_degree)

            print("ra: " + str(ra) + " TYPE: " + str(type(ra)))
            print("dec: " + str(dec) + " TYPE: " + str(type(dec)))

            tiler = DESTiler.DESTiler("KnownLenses/DES_DR1_TILE_INFO.csv")

            # How to get tile name
            ra_deg, dec_deg = ra, dec
            tile_name = tiler.getTileName(ra_deg, dec_deg)
            print('TileName: ' + tile_name)

            # How to fetch all images for tile which contains given coords
            tiler.fetchTileImages(ra_deg, dec_deg, tile_name)
            print('done')
            path_processed = 'KnownLenses/DES2017'
            # get g_mag, r_mag, i_mag
            loadDES(tile_name)
            print('loaded image from DES')
            clipWCS(tile_name, num, ra_deg, dec_deg, path_processed, des_tile)
            normaliseRGB(num, tile_name, base_dir=path_processed)

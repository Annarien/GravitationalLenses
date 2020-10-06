"""
Functions used in creating the positively simulated images.
"""

"""
This is to create the positively simulated images. 
By using the g, r, and i magnitudes from the COSMOS_Ilbert2009.fits, the magnitudes are realistic,
and are used when creating the simulated lenses. The images have DES background sky added to them, 
to create a more realistic positively simulated image, whereas without it, the images are too smooth.
These positive images which are now referred to as PositiveWithDESSky images. 
These images are normalised and also used to create a RGB composite image. 
"""
import matplotlib
matplotlib.use('Agg')

import csv

import glob
import os
import re
from matplotlib import pyplot as plt

import numpy as np
from math import log10, floor
from csv import writer
from astLib import astWCS
from astLib import astPlots

import astropy.table as atpy
from astropy.io import fits
from __init__ import *


def cutCosmosTable(cosmos):
    """
    The cosmos table is used in order to get magnitudes,inorder to provide realistic
    magnitudes for our training set. This is used to create tables of magnitudes for
    gravitational lenses and for the sources. This ensures that the magnitudes are
    realistic in term of the g, r, and i magnitude bands, and those of the sources and lenses.

    Args:
        cosmos(table):          The table retrieved from the COSMOS_Ilbert2009.fits.
    Returns:
        sourceTable(table):     The sources_table containing objects with the revelant magnitudes of typical
                                strong galaxy-galaxy gravitational sources.
        lens_table(table):      The lens_table containing objects with the revelant magnitudes of typical
                                strong galaxy-galaxy gravitational lenses.
    """
    # cosmos = cosmos[cosmos['Rmag'] < 22]

    # create a g-r column and limit it to 0<g-r<2
    cosmos.add_column(atpy.Column(cosmos['Gmag'] - cosmos['Rmag'], 'gr'))
    # create a r-i column and limit it to 0<r-i<1
    cosmos.add_column(atpy.Column(cosmos['Rmag'] - cosmos['Imag'], 'ri'))
    gr = cosmos['gr']
    ri = cosmos['ri']
    # calculate c parallel = 0.7(g-r) + 1.2 (r-i -0.18)
    c_parallel = 0.7 * gr + 1.2 * (ri - 0.18)
    # calculate c perpendicular = (r-i) - (g-r)/4 -0.18
    c_perpendicular = ri - (gr / 4) - 0.18
    cosmos.add_column(atpy.Column(c_parallel, 'c_parallel'))
    cosmos.add_column(atpy.Column(c_perpendicular, 'c_perpendicular'))

    # limit the c_perpendicular  < 0.2
    # limit the 16 < r < 19.5
    # limit the r< 13.6 + c_// /0.3

    tableA = cosmos[abs(cosmos['c_perpendicular']) < 0.2]
    tab = tableA[np.logical_and(tableA['Rmag'] > 16, tableA['Rmag'] < 19.5)]
    # tab = tableB[tableB['Rmag'] < 13.6 + (tableB['c_parallel'])/0.3]
    fits.writeto('ReducedCOSMOS.fits', np.array(tab), overwrite=True)

    # sources_table = tab[np.logical_and(tab['zpbest'] > 1, tab['zpbest'] < 2)]
    # lens_table = tab[np.logical_and(tab['zpbest'] > 0.1, tab['zpbest'] < 0.3)]
    sources_table = tab[np.logical_and(tab['zpbest'] > 0.2, tab['zpbest'] < 2)]
    fits.writeto('SourcesCOSMOS.fits', np.array(tab), overwrite=True)

    lens_table = tab[np.logical_and(tab['zpbest'] > 0, tab['zpbest'] < 0.2)]
    lens_table = lens_table[np.logical_and(lens_table['Imag'] > 18, lens_table['Imag'] < 22)]  # The Imag<21 is
    fits.writeto('LensesCOSMOS.fits', np.array(tab), overwrite=True)

    # changed from Imag<22
    source_max_r = max(sources_table['Rmag'])
    source_max_i = max(sources_table['Imag'])
    print('SourceMaxR:' + str(source_max_r))
    print('SourceMaxI:' + str(source_max_i))
    lens_max_r = max(lens_table['Rmag'])
    lens_max_i = max(lens_table['Imag'])
    print('LensMaxR:' + str(lens_max_r))
    print('LensMaxI:' + str(lens_max_i))

    source_min_r = min(sources_table['Rmag'])
    source_min_i = min(sources_table['Imag'])
    print('SourceMinR:' + str(source_min_r))
    print('SourceMinI:' + str(source_min_i))
    lens_min_r = min(lens_table['Rmag'])
    lens_min_i = min(lens_table['Imag'])
    print('LensMinR:' + str(lens_min_r))
    print('LensMinI:' + str(lens_min_i))

    print('Row length of Sources Table ' + str(len(sources_table)) + '\n')
    print('Column length of Sources Table ' + str(len(sources_table[0])) + '\n')
    print(sources_table)
    print('Row length of Lens Table ' + str(len(lens_table)) + '\n')
    print('Column length of Lens Table ' + str(len(lens_table[0])) + '\n')
    print(lens_table)
    return sources_table, lens_table


def makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, positive_noiseless, survey="DESc"):
    """
    Writes .fits images in g, r, and i bands to create the data set of positively simulated data, of strong
    galaxy-galaxy gravitational lenses.

    Args:
        ml(dictionary):            Apparent magnitude of the lens, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        rl(float):                 Half-light radius of the lens, in arcsec.
        ql(float):                 Flattening of the lens (1 = circular, 0 = line).
        b(float):                  Einsten radius, in arcsec.
        ms(dictionary):            Apparent magnitude of the source, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        xs(float):                 Horizontal coord of the source relative to the lens centre, in arcsec.
        ys(float):                 Vertical coord of the source relative to the lens centre, in arcsec.
        qs(float):                 Flattening of the source (1 = circular, 0 = line).
        ps(float):                 Position angle of the source (degrees).
        rs(float):                 Half-light radius of the source, in arcsec.
        num(integer):              The number of object that is made. This is the source number where working
                                   with the positive simulated data.
        positive_noiseless(string):This is the file path name for the positively simulated images of the lenses.
        survey(str):               Name of survey (as defined by LensPop), and is set to default of DESc. "DESc"
                                   corresponds to optimally stacked DES images.
    Returns:
        lens_g_mag(float):    This is the g band magnitude of the simulated lens, rounded off to 5 significant figures.
        lens_r_mag(float):    This is the r band magnitude of the simulated lens, rounded off to 5 significant figures.
        lens_i_mag(float):    This is the i band magnitude of the simulated lens, rounded off to 5 significant figures.
        source_g_mag(float):  This is the g band magnitude of the simulated source, rounded off to 5 significant figures.
        source_r_mag(float):  This is the r band magnitude of the simulated source, rounded off to 5 significant figures.
        source_i_mag(float):  This is the i band magnitude of the simulated source, rounded off to 5 significant figures.
    Saves:
       img(fits image):       The images of each band g, r, and i for the source number are created and saved as fits
                              images.
       psf(fits image):       The psf images of each band g, r, and i for the source number are created and saved as
                              fits images.
    """

    S = FastLensSim(survey, fractionofseeing=1)
    S.bfac = float(2)  # Not sure what these do - need to check
    S.rfac = float(2)

    source_number = 1

    # Lens half-light radius in arcsec (weirdly, dictionary by band, all values the same, in arcsec)
    rl_dict = {}
    for band in S.bands:
        rl_dict[band] = rl

    S.setLensPars(ml, rl_dict, ql, reset=True)
    S.setSourcePars(b, ms, xs, ys, qs, ps, rs, sourcenumber=source_number)

    # Makes simulated image, convolving with PSF and adding noise
    model = S.makeLens(stochasticmode="MP")
    SOdraw = numpy.array(S.SOdraw)
    S.loadModel(model)
    S.stochasticObserving(mode="MP", SOdraw=SOdraw)
    print(num, band)
    S.ObserveLens()

    # Write FITS images
    if not os.path.exists(positive_noiseless):
        os.makedirs(positive_noiseless)

    # For writing output
    folder = ('%s/%i' % (positive_noiseless, num))
    print("Folder: " + str(folder))
    if not os.path.exists(folder):
        os.makedirs(folder)

    lens_g_mag = round_sig(ml['g_SDSS'], 5)
    lens_r_mag = round_sig(ml['r_SDSS'], 5)
    lens_i_mag = round_sig(ml['i_SDSS'], 5)
    source_g_mag = round_sig(ms['g_SDSS'], 5)
    source_r_mag = round_sig(ms['r_SDSS'], 5)
    source_i_mag = round_sig(ms['i_SDSS'], 5)

    # Adding headers to the images
    # for band in S.bands:
    # print(S.bands)
    for band in ['g_SDSS', 'r_SDSS', 'i_SDSS']:
        img = S.image[band]
        psf = S.psf[band]

        header = fits.Header()
        header.set('Lens_g_mag', lens_g_mag)
        header.set('Lens_r_mag', lens_r_mag)
        header.set('Lens_i_mag', lens_i_mag)
        header.set('Source_g_mag', source_g_mag)
        header.set('Source_r_mag', source_r_mag)
        header.set('Source_i_mag', source_i_mag)

        # print('%s/%s_image_%s.fits' % (folder, num, band))
        fits.writeto('%s/%s_image_%s.fits' % (folder, num, band), img, header=header, overwrite=True)
        fits.writeto('%s/%s_psf_%s.fits' % (folder, num, band), psf, header=header, overwrite=True)

    return lens_g_mag, lens_r_mag, lens_i_mag, source_g_mag, source_r_mag, source_i_mag


def addSky(num, positive_noiseless, sky_path, positive_path):
    """
    Adds the DESSky images to the positive noiseless images made in this python file,
    to make them more realistic with noise from real DES images.
    This is saved to 'PositiveWithDESSky/%s/%s_posSky_%s.fits'%(num,num,band).

    Args:
        num(integer):              This is the source number of the positively simulated data.
        positive_noiseless(string):This is the file path for the positively simulated images of the lenses.
        sky_path(string):          This is the file path for the background sky from the DES images created in the
                                   negativeDES.py file
        positive_path(string):     This is the file path for the positive simulated images that has background sky now
                                   added to it.
    Saves:
        with_sky(fits image):   This is the clipped sky added to the positively simulated images added together, so that
                                the simulated images have background noise, so that the positive images are realistic.
                                This is saved under the directory: positive_path/num/num_band_posSky.fits
    """
    if not os.path.exists('%s/%i' % (positive_path, num)):
        os.makedirs('%s/%i' % (positive_path, num))

    for band in ['g', 'r', 'i']:
        # print('%s/%s/%s_image_%s_SDSS.fits' % (positive_noiseless, num, num, band))
        hdulist = fits.open('%s/%s/%s_image_%s_SDSS.fits' % (positive_noiseless, num, num, band))
        lens_g_mag = hdulist[0].header.get('Lens_g_mag')
        lens_r_mag = hdulist[0].header.get('Lens_r_mag')
        lens_i_mag = hdulist[0].header.get('Lens_i_mag')
        source_g_mag = hdulist[0].header.get('Source_g_mag')
        source_r_mag = hdulist[0].header.get('Source_r_mag')
        source_i_mag = hdulist[0].header.get('Source_i_mag')

        band_sky_image = fits.open('%s/%i_%s_sky.fits' % (sky_path, num, band))
        band_pos_noiseless_image = fits.open('%s/%s/%s_image_%s_SDSS.fits' % (positive_noiseless, num, num, band))
        with_sky = band_sky_image[0].data + band_pos_noiseless_image[0].data

        header = fits.Header()
        header.set('Lens_g_mag', lens_g_mag)
        header.set('Lens_r_mag', lens_r_mag)
        header.set('Lens_i_mag', lens_i_mag)
        header.set('Source_g_mag', source_g_mag)
        header.set('Source_r_mag', source_r_mag)
        header.set('Source_i_mag', source_i_mag)

        fits.writeto('%s/%i/%i_%s_posSky.fits' % (positive_path, num, num, band), with_sky, header=header,
                     overwrite=True)


def normalise(num, positive_path):
    """
    This is to normalise the g, r, and i PositiveWithDESSky images that
    were made by adding the background sky to the noiseless positively
    simulated images. The g, r, and i normalised images are then used to create
    a RGB composite images.

    Args:
        num(integer):           This is the source number of the positively simulated data.
        positive_path(string):  This is the file path for the positive simulated images that has background sky now
                                   added to it.
    Saves:
        norm_image(numpy array):An array containing the normalised image with the wcs as a header saved
                                under 'PositiveWithDESSky/num/'.
        figure(plot):           This is the rgb composite images are created and  are saved
                                under 'PositiveWithDESSky/num/'.
    """
    paths = {'iImg': glob.glob('%s/%s/%s_i_posSky.fits' % (positive_path, num, num))[0],
             'rImg': glob.glob('%s/%s/%s_r_posSky.fits' % (positive_path, num, num))[0],
             'gImg': glob.glob('%s/%s/%s_g_posSky.fits' % (positive_path, num, num))[0]}

    rgb_dict = {}
    wcs = None
    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'Img']) as image:

            lens_g_mag = image[0].header.get('Lens_g_mag')
            lens_r_mag = image[0].header.get('Lens_r_mag')
            lens_i_mag = image[0].header.get('Lens_i_mag')
            source_g_mag = image[0].header.get('Source_g_mag')
            source_r_mag = image[0].header.get('Source_r_mag')
            source_i_mag = image[0].header.get('Source_i_mag')

            header = fits.Header()
            header.set('Lens_g_mag', lens_g_mag)
            header.set('Lens_r_mag', lens_r_mag)
            header.set('Lens_i_mag', lens_i_mag)
            header.set('Source_g_mag', source_g_mag)
            header.set('Source_r_mag', source_r_mag)
            header.set('Source_i_mag', source_i_mag)

            im = image[0].data
            norm_image = (im - im.mean()) / np.std(im)
            if wcs is None:
                wcs = astWCS.WCS(image[0].header, mode='pyfits')
            # astImages.saveFITS('%s/%s/%s_%s_norm.fits' % (positive_path, num, num, band), norm_image, header= header)
            fits.writeto('%s/%s/%s_%s_norm.fits' % (positive_path, num, num, band), norm_image, header=header,
                         overwrite=True)
            rgb_dict[band] = norm_image

    min_cut, max_cut = -1, 3
    cut_levels = [[min_cut, max_cut], [min_cut, max_cut], [min_cut, max_cut]]
    plt.figure(figsize=(10, 10))
    astPlots.ImagePlot([rgb_dict['i'], rgb_dict['r'], rgb_dict['g']],
                       wcs,
                       cutLevels=cut_levels,
                       axesLabels=None,
                       axesFontSize=26.0,
                       axes=[0, 0, 1, 1])
    plt.savefig('%s/%i/%i_rgb.png' % (positive_path, num, num))


def getNegativeNumbers(base_dir):
    """
    This is used to get the numbers of the sources, as seen in the directories of the negative training and testing set.
    This is creates an list of these numbers.

    Args:
        base_dir(string):   This is the root path name in which the directories in question are found.
    Returns:
        Numbers(list):      This is the list of the indexes of the directories, in question.
    """

    folders = {}
    numbers = []

    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value
            print(key)

            num = int(re.search(r'\d+', key).group())
            numbers.append(num)

    # print(numbers)

    return numbers


def round_sig(x, sig=3):
    """
    This is used to ensure that the specified value has no more than the specified significant figures, this is used to
    ensure that calculations are done correctly and efficiently, whilst saving memory.
    Args:
        x(float):             This is the float number in question which has too many decimals.
        sig(integer):         This is the amount of significant figures the float needs to contain.
                              This default is set to 3.
    Returns:
        rounded_value(float): This is the float that is rounded to the specified significant figures.
    """
    rounded_value = round(x, sig - int(floor(log10(abs(x)))) - 1)
    return rounded_value


def magnitudeTable(num, lens_g_mag, lens_r_mag, lens_i_mag, source_g_mag, source_r_mag, source_i_mag, positive_path):
    """
    This creates a csv file containing the source index and the g, r and i band magnitudes for the lens and the source.
    The g-r and r-i values are also calculated and inserted into the table for each source and lens.
    Args:
        num(integer):   This is the index of the source as seen in the folders for these objects.
        lens_g_mag(float):   This is the g band magnitude of the simulated lens, rounded off to 5 significant figures.
        lens_r_mag(float):   This is the r band magnitude of the simulated lens, rounded off to 5 significant figures.
        lens_i_mag(float):   This is the i band magnitude of the simulated lens, rounded off to 5 significant figures.
        source_g_mag(float): This is the g band magnitude of the simulated source, rounded off to 5 significant figures.
        source_r_mag(float): This is the r band magnitude of the simulated source, rounded off to 5 significant figures.
        source_i_mag(float): This is the i band magnitude of the simulated source, rounded off to 5 significant figures.
        positive_path(string):This is the file path for the positive simulated images that has background sky now
                              added to it.
    """

    magTable_headers = ['Index', 'Lens_g_mag', 'Lens_r_mag', 'Lens_i_mag', 'Source_g_mag', 'Source_r_mag',
                        'Source_i_mag', 'lens_gr', 'lens_ri', 'lens_gi', 'source_gr', 'source_ri', 'source_gi']
    lens_gr = round_sig(lens_g_mag - lens_r_mag)
    lens_ri = round_sig(lens_r_mag - lens_i_mag)
    lens_gi = round_sig(lens_g_mag - lens_i_mag)
    source_gr = round_sig(source_g_mag - source_r_mag)
    source_ri = round_sig(source_r_mag - source_i_mag)
    source_gi = round_sig(source_g_mag - source_i_mag)
    magTable_dictionary = [num, lens_g_mag, lens_r_mag, lens_i_mag, source_g_mag, source_r_mag, source_i_mag, lens_gr,
                           lens_ri, lens_gi, source_gr, source_ri, source_gi]

    # tab = atpy.Table()
    if not os.path.exists('%s_magnitudesTable.csv' % positive_path):
        # tab.write('%s_magnitudesTable.csv' % positive_path)
        with open('%s_magnitudesTable.csv' % positive_path, 'w') as out_csv:
            magTableWithHeaders = csv.DictWriter(out_csv, magTable_headers)
            magTableWithHeaders.writeheader()

    with open('%s_magnitudesTable.csv' % positive_path, 'a+') as writeObj:
        csvWriter = writer(writeObj)
        csvWriter.writerow(magTable_dictionary)

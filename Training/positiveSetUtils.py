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
import glob
import os
import re
import matplotlib
import numpy as np
matplotlib.use('Agg')

from astLib import *
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
        lens_table(table):       The lens_table containing objects with the revelant magnitudes of typical
                                strong galaxy-galaxy gravitational lenses.
    """
    tab = cosmos[cosmos['Rmag'] < 22]
    # DES2017
    sources_table = tab[np.logical_and(tab['zpbest'] > 1, tab['zpbest'] < 2)]
    lens_table = tab[np.logical_and(tab['zpbest'] > 0.1, tab['zpbest'] < 0.3)]
    lens_table = lens_table[np.logical_and(lens_table['Imag'] > 18, lens_table['Imag'] < 22)] # The Imag<21 is
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
    print (sources_table)
    print('Row length of Lens Table ' + str(len(lens_table)) + '\n')
    print('Column length of Lens Table ' + str(len(lens_table[0])) + '\n')
    print(lens_table)
    return sources_table, lens_table


def makeModelImage(ml, rl, ql, b, ms, xs, ys, qs, ps, rs, num, positive_noiseless, survey="DESc"):
    """
    Writes .fits images in g, r, and i bands to create the data set of positively simulated data, of strong
    galaxy-galaxy gravitational lenses.

    Args:
        ml(dictionary):     Apparent magnitude of the lens, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        rl(float):          Half-light radius of the lens, in arcsec.
        ql(float):          Flattening of the lens (1 = circular, 0 = line).
        b(float):           Einsten radius, in arcsec.
        ms(dictionary):     Apparent magnitude of the source, by band (keys: 'g_SDSS', 'r_SDSS', 'i_SDSS').
        xs(float):          Horizontal coord of the source relative to the lens centre, in arcsec.
        ys(float):          Vertical coord of the source relative to the lens centre, in arcsec.
        qs(float):          Flattening of the source (1 = circular, 0 = line).
        ps(float):          Position angle of the source (degrees).
        rs(float):          Half-light radius of the source, in arcsec.
        num(integer):       The number of object that is made. This is the source number where working
                            with the positive simulated data.
        survey(str):        Name of survey (as defined by LensPop), and is set to default of DESc.
                            "DESc" corresponds to optimally stacked DES images.
    Saves:
       img(fits image):     The images of each band g, r, and i for the source number are created and saved as fits
                            images.
       psf(fits image):     The psf images of each band g, r, and i for the source number are created and saved as fits
                            images.
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
    print (num, band)
    S.ObserveLens()

    # Write FITS images
    if not os.path.exists(positive_noiseless):
        os.makedirs(positive_noiseless)

    # For writing output
    folder = ('%s/%i' % (positive_noiseless, num))
    if not os.path.exists(folder):
        os.makedirs(folder)

    for band in S.bands:
        img = S.image[band]
        psf = S.psf[band]
        fits.PrimaryHDU(img).writeto('%s/%s_image_%s.fits' % (folder, num, band), overwrite=True)
        fits.PrimaryHDU(psf).writeto('%s/%s_psf_%s.fits' % (folder, num, band), overwrite=True)


def addSky(num, positive_noiseless, sky_path, positive_path):
    """
    Adds the DESSky images to the positive noiseless images made in this python file,
    to make them more realistic with noise from real DES images.
    This is saved to 'PositiveWithDESSky/%s/%s_posSky_%s.fits'%(num,num,band).

    Args:
        num(integer):   This is the source number of the positively simulated data.
    Saves:
        with_sky(fits image):   This is the clipped sky added to the positively simulated images added together, so that
                                the simulated images have background noise, so that the positive images are realistic.
                                This is saved under the directory: positive_path/num/num_band_posSky.fits
    """

    if not os.path.exists('%s/%i' % (positive_path, num)):
        os.makedirs('%s/%i' % (positive_path, num))

    for band in ['g', 'r', 'i']:
        band_sky_image = fits.open('%s/%i_%s_sky.fits' % (sky_path, num, band))
        band_pos_noiseless_image = fits.open('%s/%s/%s_image_%s_SDSS.fits' % (positive_noiseless, num, num, band))
        with_sky = band_sky_image[0].data + band_pos_noiseless_image[0].data
        astImages.saveFITS('%s/%i/%i_%s_posSky.fits' % (positive_path, num, num, band), with_sky)


def normalise(num, positive_path):
    """
    This is to normalise the g, r, and i PositiveWithDESSky images that
    were made by adding the background sky to the noiseless positively
    simulated images. The g, r, and i normalised images are then used to create
    a RGB composite images.

    Args:
        num(integer):   This is the source number of the positively simulated data.
    Saves:
        The normalised images with the wcs as headers.
        These normalised images are saved under 'PositiveWithDESSky/num/'.
        The rgb composite images are created and saved under 'PositiveWithDESSky/num/'.
    """
    paths = {'iImg': glob.glob('%s/%s/%s_i_posSky.fits' % (positive_path, num, num))[0],
             'rImg': glob.glob('%s/%s/%s_r_posSky.fits' % (positive_path, num, num))[0],
             'gImg': glob.glob('%s/%s/%s_g_posSky.fits' % (positive_path, num, num))[0]}

    rgb_dict = {}
    wcs = None
    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'Img']) as image:
            im = image[0].data
            norm_image = (im - im.mean()) / np.std(im)
            if wcs is None:
                wcs = astWCS.WCS(image[0].header, mode='pyfits')
            astImages.saveFITS('%s/%s/%s_%s_norm.fits' % (positive_path, num, num, band), norm_image, None)
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

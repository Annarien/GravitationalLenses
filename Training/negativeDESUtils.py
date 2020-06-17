"""
This contains all definitions used in getting the negativeDES data.
"""

import os
import random
import wget
import glob
import numpy as np
import pylab as plt
from astropy.io import fits
from astLib import astWCS
from astLib import astImages
from astLib import astPlots
from bs4 import BeautifulSoup


def getRandomIndices(desired_list_size, random_indices_list, survey_len):
    index_list = []
    while len(index_list) < desired_list_size:
        random_index = np.random.randint(0, survey_len)
        if random_index not in random_indices_list:
            random_indices_list.append(random_index)
            index_list.append(random_index)
    return index_list


def loadDES(source, base_dir='DES/DES_Original'):
    """
    Firstly the .fits files are downloaded from DES DR1.
    This contains the g, r, i magnitudes as well as the RA and DEC, for each source.
    The g, r, i .fits files are downloaded for each source from the DES DR1 server.
    DownLoading the images in a folder, only containg DES original .fits files.

    Args:
        source(string):     This is the tilename given in the DR1 database, and this is name of each source.
        base_dir(string):   This is the base directory in which the folders are made.
    Saves:
        The 10 000 * 10 000 pixel, DES Original Images for the g, r, and i fits images of each source.

    """
    if not os.path.exists('%s' % base_dir):
        os.mkdir('%s' % base_dir)

    # For each tile name, download the HTML, scrape it for the files and create the correct download link
    if not os.path.exists('%s/%s' % (base_dir, source)):
        os.mkdir('%s/%s' % (base_dir, source))

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
                if col.text.find("r.fits.fz") != -1 or col.text.find("i.fits.fz") != -1 or col.text.find(
                        "g.fits.fz") != -1:
                    if not os.path.exists('%s/%s/%s' % (base_dir, source, col.text)):
                        print('Downloading: ' + url + col.text)
                        wget.download(url + col.text, '%s/%s/%s' % (base_dir, source, col.text))
                        print()
                    else:
                        print('%s/%s/%s already downloaded...' % (base_dir, source, col.text))
                        print()
        print()


def randomXY(source, base_dir='DES/DES_Original'):
    """
    This gets random x, y coordinates in the original g band of DES images.
    Only one band is used to get these coordinates, as the same random
    coordinates are needed in all bands. This also ensure that images are
    100*100 pixels in size, and all pixels are within the images.

    Args:
        source(string):     This is the tilename of the DES DR1 images, used for each object.
        base_dir(string):   This is the root directory that contains the original DES images.
    Returns:
        x_random(int):       The random x coordinate, within the DES Original g band image.
        y_random(int):       The random y coordinate, within the DES Original g band image.
    """

    with fits.open(glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0]) as bandDES:
        in_hdu_list = bandDES[1].header
        print ('NAXIS1: ' + str(in_hdu_list['NAXIS1']))
        print ('NAXIS2: ' + str(in_hdu_list['NAXIS2']))

        x_max = in_hdu_list['NAXIS1']
        y_max = in_hdu_list['NAXIS2']
        x_random = random.randint(100, x_max - 100)
        y_random = random.randint(100, y_max - 100)

        print("x: " + str(x_random))
        print("y: " + str(y_random))
        return x_random, y_random


def randomSkyClips(num, source, ra, dec, gmag, rmag, imag, dessky_file, base_dir='DES/DES_Original'):
    """
    Clipping of the g, r, and i DES Original fits images, to create a 100*100 pixel sized image of noise/sky.

    Args:
        num(integer):       Number identifying the particular processed negative folder and files is being used.
        source(string):     This is the tilename of the clipped sky images.
        ra(float):          The right ascension of the clipped original image from DES.
        dec(float):         The declination of the clipped original image from DES.
        gmag(float):        The magnitude of the g band of the orignal images from DES.
        rmag(float):        The magnitude of the r band of the orignal images from DES.
        image(float):       The magnitude of the i band of the orignal images from DES.
        base_dir(string):   The root directory of the orignal DES images, which are
                            used to be clipped into the sky images.
    Saves:
        clipped_sky[band](dictionary):   This is 100 * 100 pixels of the g, r, and i images of the DES images at the random
                                         coordinates. This is saved under the folder of 'DESSky/'.
    """

    paths = {'gBandPath': glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0],
             'rBandPath': glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, source, source))[0],
             'iBandPath': glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, source, source))[0]}

    if not os.path.exists('%s' % dessky_file):
        os.mkdir('%s' % dessky_file)

    made_sky = False
    clipped_sky = {}
    while not made_sky:
        all_images_valid = True
        x, y = randomXY(source)
        for band in ['g', 'r', 'i']:
            with fits.open(paths[band + 'BandPath']) as band_des:
                band_sky = astImages.clipImageSectionPix(band_des[1].data, x, y, [100, 100])

                if np.any(band_sky) == 0:
                    all_images_valid = False
                    print("randomly-chosen postage stamp position contained zero values - trying again ...")
                    break
                else:
                    clipped_sky[band] = band_sky

        if all_images_valid:
            made_sky = True

    for band in clipped_sky.keys():
        header = fits.Header()
        header['TILENAME'] = source
        header['RA'] = ra
        header['DEC'] = dec
        header['G_MAG'] = gmag
        header['I_MAG'] = imag
        header['R_MAG'] = rmag
        fits.writeto('%s/%i_%s_sky.fits' % (dessky_file, num, band), clipped_sky[band], header=header, overwrite=True)


def clipWCS(source, num, gmag, rmag, imag, ra, dec, base_new, base_dir='DES/DES_Original'):
    """
    Clips the g, r, i original .fits images for each source from DES to have 100*100 pixel size or 0.0073125*0.007315 degrees.
    The wcs coordinates are used, to maintain the necessary information that may be needed in future.
    These WCSclipped images are saved at ('%s.WCSclipped.fits' % (paths[band+'BandPath']).
    The wcs images, are normalised and saved at ('%s.norm.fits' % (paths[band + 'BandPath']).

    Args:
        source(string):     This is the tilename of the original images from DES.
        num(integer):       Number identifying the particular processed negative folder and files is being used.
        gmag(float):        The magnitude of the g band of the original images from DES.
        rmag(float):        The magnitude of the r band of the original images from DES.
        image(float):       The magnitude of the i band of the original images from DES.
        ra(float):          The right ascension of the orignal images from DES.
        dec(float):         The declination of the original images from DES.
        base_dir(string):   The root directory of the orignal DES images.
        base_new(string):   The root directory in which the wcs_clipped images are saved,
                            this is defaulted to 'DES/DES_Processed'.
    Returns:
        wcs_clipped (numpy array):   A numpy array of the WCSclipped, with its wcs coordinates, and is saved under
                                     'DES/DES_Processed'.
    """
    # Getting the RA and Dec of each source
    size_wcs = [0.0073125, 0.0073125]  # 100*100 pixels in degrees

    paths = {'gBandPath': glob.glob('%s/%s/%s*_g.fits.fz' % (base_dir, source, source))[0],
             'rBandPath': glob.glob('%s/%s/%s*_r.fits.fz' % (base_dir, source, source))[0],
             'iBandPath': glob.glob('%s/%s/%s*_i.fits.fz' % (base_dir, source, source))[0]}

    if not os.path.exists('%s' % base_new):
        os.mkdir('%s' % base_new)

    new_path = ('%s/%s_%s' % (base_new, num, source))
    if not os.path.exists(new_path):
        os.mkdir('%s/%s_%s' % (base_new, num, source))

    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'BandPath']) as band_des:
            header = band_des[1].header
            header.set('MAG_G', gmag)
            header.set('MAG_I', imag)
            header.set('MAG_R', rmag)
            header.set('RA', ra)
            header.set('DEC', dec)
            wcs = astWCS.WCS(header, mode="pyfits")
            wcs_clipped = astImages.clipImageSectionWCS(band_des[1].data, wcs, ra, dec, size_wcs)
            astImages.saveFITS('%s/%s_WCSClipped.fits' % (new_path, band), wcs_clipped['data'], wcs)
            print('Created %s_WCSclipped at %s/%s_WCSClipped.fits' % (band, new_path, band))

    return wcs_clipped


def normaliseRGB(num, source, base_dir):
    """
    This is to normalise the g, r, and i WCSClipped images and to make a rgb composite image of the three band together.

    Args:
        num(integer):       Number identifying the particular processed negative folder and files is being used.
        source(string):     This is the tilename of the original images from DES.
        base_dir(string):   The root directory in which the normalised images and the rgb compostie images are saved,
                            this is defaulted to 'DES/DES_Processed'.
    Saves:
        norm_image(numpy array): This is the wcs clipped images normalised, and saved under 'DES/DES_Processed/num/source/
        rgb(png):                This is a rgb composite image created and saved under 'DES/DES_Processed/num_source/'.
    """

    paths = {'iBandPath': glob.glob('%s/%s_%s/i_WCSClipped.fits' % (base_dir, num, source))[0],
             'rBandPath': glob.glob('%s/%s_%s/r_WCSClipped.fits' % (base_dir, num, source))[0],
             'gBandPath': glob.glob('%s/%s_%s/g_WCSClipped.fits' % (base_dir, num, source))[0]}

    rgb_dict = {}
    wcs = None

    for band in ['g', 'r', 'i']:
        with fits.open(paths[band + 'BandPath']) as image:
            im = image[0].data
            norm_image = (im - im.mean()) / np.std(im)
            if wcs is None:
                wcs = astWCS.WCS(image[0].header, mode='pyfits')
            astImages.saveFITS('%s/%s_%s/%s_norm.fits' % (base_dir, num, source, band), norm_image, wcs)
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
    plt.savefig('%s/%s_%s/rgb.png' % (base_dir, num, source))

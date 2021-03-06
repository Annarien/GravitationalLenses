import glob
import os
import random
import re

import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from textwrap import wrap

from positiveSetUtils import getNegativeNumbers

train_positive_path = 'Training/PositiveAll'
train_negative_path = 'Training/Negative'


# Open DES Processed WCS .fits files, and assign a variable to the g, r, i images.
def getNegativeProcessedWCS(num, base_dir=train_negative_path):
    """
    This is to open the files of the negative DES WCSClipped images for the g, r, and i bands,
    which have been clipped using the World Coordinate Systems (WCS).

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i .fits
                            images of the DES Processed images which have been clipped using the WCS coordinates.
                            The file names are in the form 'Training/Negative/DES/num_source/band_WCSClipped.fits'.
        base_dir(string):   This is the top directory path of the background sky of the DES Original images.
                            This is set to a default of 'Training/Negative'.
    Returns:
        g_wcs(HDUList):      The data of the opened g band WCSClipped fits image.
        r_wcs(HDUList):      The data of the opened r band WCSClipped fits image.
        i_wcs(HDUList):      The data of the opened i band WCSClipped fits image.
    """
    g_wcs = fits.open(glob.glob('%s/%s_*/g_WCSClipped.fits' % (base_dir, num))[0])
    r_wcs = fits.open(glob.glob('%s/%s_*/r_WCSClipped.fits' % (base_dir, num))[0])
    i_wcs = fits.open(glob.glob('%s/%s_*/i_WCSClipped.fits' % (base_dir, num))[0])
    return g_wcs, r_wcs, i_wcs


# Open DES Processed norm .fits files and assign a variable to the g, r, i images.
def getNegativeDES(num, base_dir=train_negative_path):
    """
    This is to open the normalised files of the negative DES WCSClipped images for the g, r, and i bands.

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i .fits
                            images of the DES Processed images which have been normalised.
                            The file names are in the form 'Training/Negative/num_source/band_norm.fits'.
        base_dir(string):   This is the top directory path of the background sky of the DES Original images.
                            This is set to a default of 'Training/Negative'.
    Returns:
        The images that are returned is the WCSClipped image normalised.
        g_des_norm(HDUList):  The data of the opened g band normalised .fits image.
        r_des_norm(HDUList):  The data of the opened r band normalised .fits image.
        i_des_norm(HDUList):  The data of the opened i band normalised .fits image.
    """

    g_des_norm = fits.open(glob.glob('%s/%s_*/g_norm.fits' % (base_dir, num))[0])
    r_des_norm = fits.open(glob.glob('%s/%s_*/r_norm.fits' % (base_dir, num))[0])
    i_des_norm = fits.open(glob.glob('%s/%s_*/i_norm.fits' % (base_dir, num))[0])

    return g_des_norm, r_des_norm, i_des_norm


# Open DESSky .fits files and assign a variable to the g, r, i images.
def getDESSky(num, base_dir='Training/DESSky'):
    # print('%s/%s_g_sky.fits' % (base_dir, num))
    """
    This uses the num to open files of the background sky of the DES Original images for the g, r, and i bands.

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i .fits 
                            images of the background sky of the DES Original images.
                            The file names are in the form 'Training/DESSky/num_band_sky.fits'.
        base_dir(string):   This is the top directory path of the background sky of the DES Original images. 
                            This is set to a default of 'Training/DESSky'.
    Returns:
        g_des_sky(HDUList):   The data of the opened g band sky fits image.
        r_des_sky(HDUList):   The data of the opened r band sky fits image.
        i_des_sky(HDUList):   The data of the opened i band sky fits image.
    """
    g_des_sky = fits.open(glob.glob('%s/%s_g_sky.fits' % (base_dir, num))[0])
    r_des_sky = fits.open(glob.glob('%s/%s_r_sky.fits' % (base_dir, num))[0])
    i_des_sky = fits.open(glob.glob('%s/%s_i_sky.fits' % (base_dir, num))[0])
    return g_des_sky, r_des_sky, i_des_sky


# Open PositiveNoiseless .fits files and assign a variable to the ..._SDSS_g, r, images.
def getPosNoiseless(num, base_dir='Training/PositiveNoiselessAll'):
    """
    This is to open files of the smoothly positively simulated images of gravitational lensing for the g, r, and i bands.

    Args:
        num(integer):       This is the number of the source, which corresponds to the g, r, and i positively
                            simulated .fits.
                            The file names are in the form 'Training/PositiveNoiseless/num/num_image_band_SDSS.fits'.
        base_dir(string):   This is the top directory path of the positively simulated images.
                            This is set to a default of 'Training/PositiveNoiseless'.
    Returns:
        g_pos(HDUList):      The data of the opened g band of the positively simulated fits image.
        r_pos(HDUList):      The data of the opened r band of the positively simulated fits image.
        i_pos(HDUList):      The data of the opened i band of the positively simulated fits image.
    """

    g_pos = fits.open(glob.glob('%s/%s/%s_image_g_SDSS.fits' % (base_dir, num, num))[0])
    r_pos = fits.open(glob.glob('%s/%s/%s_image_r_SDSS.fits' % (base_dir, num, num))[0])
    i_pos = fits.open(glob.glob('%s/%s/%s_image_i_SDSS.fits' % (base_dir, num, num))[0])
    return g_pos, r_pos, i_pos


# Open PositiveWithDESSky  .fits files and assign a variable to the ...posSky_g, r, i images.
def getPosWDESSky(num, base_dir=train_positive_path):
    """
    The number is used to open files of the positively simulated images of gravitational lensing for the g, r, and
    i bands, that have the background sky added to them.

    Args:
        num(integer):           This is the number of the source, which corresponds to the g, r, and i positively
                                simulated .fits with background sky from the original DES images added to it.
                                The file names are in the form 'Training/Positive/num_source/num_posSky_band.fits'.
        base_dir(string):       This is the top directory path of the positively simulated images with the
                                background sky added to it. This is set to a default of 'Training/Positive'.
    Returns:
        g_pos_sky(HDUList):      The data of the opened g band of the positively simulated with the background
                               sky added fits images.
        r_pos_sky(HDUList):      The data of the opened r band of the positively simulated with the background
                               sky added fits images.
        i_pos_sky(HDUList):      The data of the opened i band of the positively simulated with the background
                               sky added fits images.
    """

    g_pos_sky = fits.open(glob.glob('%s/%s/%s_g_posSky.fits' % (base_dir, num, num))[0])
    r_pos_sky = fits.open(glob.glob('%s/%s/%s_r_posSky.fits' % (base_dir, num, num))[0])
    i_pos_sky = fits.open(glob.glob('%s/%s/%s_i_posSky.fits' % (base_dir, num, num))[0])
    return g_pos_sky, r_pos_sky, i_pos_sky


# Open PositiveWithDESSky norm. fits images and assign a variable to the ...posSky_g, r, i_norm images.
def getPosWDESSkyNorm(num, base_dir=train_positive_path):
    """
    This is to open files of the normalised positively simulated images of gravitational lensing for the g, r, and
    i bands, that have the background sky added to them.

    Args:
        num(integer):           This is the number of the source, which corresponds to the g, r, and i norm. fits
                                images of the sources.
                                The file names are in the form 'Training/Positive/num_source/num_band_norm.fits'.
        base_dir(string):       This is the top directory path of the positively simulated images with the
                                background sky added to it. This is set to a default of 'Training/Positive'.
    Returns:
       g_pos_sky_norm(HDUList):    The data of the opened g band of the normalised positively simulated with
                                the background sky added fits images.
        r_pos_sky_norm(HDUList):   The data of the opened r band of the normalised positively simulated with
                                the background sky added fits images.
        i_pos_sky_norm(HDUList):   The data of the opened i band of the normalised positively simulated with
                                the background sky added fits images.
    """

    g_pos_sky_norm = fits.open(glob.glob('%s/%s/%s_g_norm.fits' % (base_dir, num, num))[0])
    r_pos_sky_norm = fits.open(glob.glob('%s/%s/%s_r_norm.fits' % (base_dir, num, num))[0])
    i_pos_sky_norm = fits.open(glob.glob('%s/%s/%s_i_norm.fits' % (base_dir, num, num))[0])
    return g_pos_sky_norm, r_pos_sky_norm, i_pos_sky_norm


def getNumOrRowsForGrid(num_of_cols_for_rgb_grid, rgb_list):
    """
    This is to get a number of rows using a predetermined number of columns.
    This is to ensure that the images form a grid, so that multiple rgb images can be viewed at once.

    Args:
        num_of_cols_for_rgb_grid(integer):   The number of columns using that is predetermined.
        rgb_list(list):                      This is the list of paths for rgb images.
    Returns:
        num_of_rows_for_rgb_grid(integer):   The number of rows that is calculated using the length divided
                                             by the number of predetermined columns
    """

    len_rgb = len(rgb_list)
    num_of_rows_for_rgb_grid = (len_rgb / num_of_cols_for_rgb_grid)
    if len_rgb % num_of_cols_for_rgb_grid != 0:
        num_of_rows_for_rgb_grid += 1

    return num_of_rows_for_rgb_grid


def getNegativeDESRGBPath(num):
    """
    Get the file path of the rgb.png image of the negative DES processed images.

    Args:
        num(integer):           The number of the DES Processed source. The format of this is
                                'Training/Negative/num_source/rgb.png
    Returns:
        rgb_des_path(string):   The path of the rgb.png image of the negative DES processed images.
                               The format of this is 'Training/Negative/num_source/rgb.png.
    """

    rgb_des_path = glob.glob('%s/%s_*/rgb.png' % (train_negative_path, num))[0]
    return rgb_des_path


def getKnownRGBPath(num, known_path):
    """
    The num is used to get the path of the rgb.png images of known lenses that have been previously
    identified in studies. The tilename (from DES DR1) and the DESJ2000 name
    (from the DES2017 paper) are also retrieved, as this is to get the correct names for each
    image when creating the rgb image grids of these known lenses. The tilename and DESJ names
    are retrieved from one of bands of the WCSClipped images of that source, here we will just use the g band.

    Args:
        num(integer):        This is the number that is used to find the rgb.png and the g_WCSClipped.fits files for that
                             source in the 'UnseenData/Known?' directory. The rgb.png files are in the form:
                             'UnseenData/Known47/num_source/rgb.png'. The g_WCSClipped.fit files are in the
                             form: 'UnseenData/Known47/num_source/g_WCSClipped.fits.
        known_path(string):  This is the path name name that identifies which set of known lenses are used,
                             either 'Known47' or 'Known84'.
    Returns:
        rgb_known(string):   Provides the path name for the known lenses from previous studies study.
        des_j(string):       Provides the DESJ2000 name of the known lenses.
        tilename(string):   Provides the DES DR1 tilename for the known lenses.
    """
    # print('known path: ' + str(known_path))
    # print('num: ' + str(num))
    # get path of KnownRGBPath
    rgb_known = glob.glob('%s/%s_*/rgb.png' % (known_path, num))[0]

    # get header of g image so that we can get the DESJ tile name

    g_band = glob.glob('%s/%s_*/g_WCSClipped.fits' % (known_path, num))[0]
    hdu1 = fits.open(g_band)
    tilename = hdu1[0].header['TILENAME']

    return rgb_known, tilename


def makeRandomRGBArray(rgb_path, number_iterations, numbers_train_neg):
    """
    This takes the root directory of rgb images that will create a list of random rgb images within the directory.
    An example of the necessary path of the root directory is 'PositiveWithDESSky' or 'DES/DES_Processed'.

    Makes an random list of the rgb.png images.
    This creates a list of random rgb images within the chosen directory.

    Args:
       rgb_path(string):                The path name in which the rgb images are found.
       number_iterations(integer):      This is the number of images that will are needed in a grid.
       numbers_train_neg(list):         This is a list of Negative data numbers, which have been created at random.
    Returns:
        rgb_random_array(list):         The list of random paths of rgb.png images, corresponding to the
                                        random_array_index as its sources.
        image_title_array(list):        The list of the numbers that correspond to the rgb.png images in the
                                        rgb_random_array.
    """

    # The error is that the images in the folders are random, not consecutively from 0,1.
    # The methods below, takes images from (0,?) At random, but some of the numbers it chooses, dont exit within the
    # folders.
    # Somehow get a list of folders from the array.
    # This can be done by calling the method in the positiveSetUtils.py method

    random_num = 0
    random_array = []
    random_array_index = 0
    rgb_random_array = []
    image_title_array = []

    files = folders = 0
    for _, dir_names, file_names in os.walk(rgb_path):
        # ^ this idiom means 'we won't be using this value'
        files += len(file_names)
        folders += len(dir_names)

    print('{:,} files, {:,} folders'.format(files, folders))

    for num in range(0, number_iterations):
        # random_num = random.randint(0, folders - 1)
        random_num = random.choice(numbers_train_neg)

        while random_num in random_array:
            # random_num = random.randint(0, folders - 1)
            random_num = random.choice(numbers_train_neg)
        random_array.append(random_num)

    print('RANDOM ARRAY: ' + str(random_array) + ' TYPE: ' + str(type(random_array)))
    for num in range(0, len(random_array)):
        random_array_index = random_array[num]
        if rgb_path == train_positive_path:
            rgb_random_array.append('%s/%s/%s_rgb.png' % (rgb_path, random_array_index, random_array_index))
            image_title_array.append(random_array_index)

        elif rgb_path == train_negative_path:
            rgb_random_array.append(glob.glob('%s/%s_*/rgb.png' % (rgb_path, random_array_index))[0])
            image_title_array.append(random_array_index)

    return rgb_random_array, image_title_array


def plotAndSaveRgbGrid(file_path, rgb_image_paths, image_title_array, figure_title):
    # You should probably pass num in here or something like that and save many images
    """
    Using the image arrays (rgbImagePaths()) to make an image grid made of RGB images.
    The title for each image is retrieved from the imageTitleArray().

    Args:
        file_path(string):          The file path name where the Figure of the image grid is to be saved.
        rgb_image_paths(list):      This is the list of the rgb images, that are used when making the rgb image grids.
        image_title_array(list):    This is the list of the names or titles of each image that is in the grid.
                                    These names will either be the numbers of the sources or the source name,
                                    depending on which data is being used.
        Returns:
            This saves the Figure, which is all the indivivual rgb images placed in a grid.
            These figures are saved in the path which is retrieved from when this function is called.
    """
    len_rgb = len(rgb_image_paths)
    num_of_cols_for_rgb_grid = 3
    num_of_rows_for_rgb_grid = getNumOrRowsForGrid(num_of_cols_for_rgb_grid, rgb_image_paths)
    fig, axs = plt.subplots(num_of_rows_for_rgb_grid, num_of_cols_for_rgb_grid, figsize=(7, 7))
    row_num = 0
    current_index = 0
    image_title_num = 0
    while row_num < num_of_rows_for_rgb_grid:
        images_for_row = []
        image_index = 0
        while image_index < num_of_cols_for_rgb_grid and current_index < len_rgb:
            images_for_row.append(rgb_image_paths[current_index])
            current_index += 1
            image_index += 1

        for column_num in range(0, len(images_for_row)):
            img = Image.open(images_for_row[column_num])
            img.thumbnail((100, 100))
            axs[row_num, column_num].imshow(img, aspect='equal')
            # axs[row_num, column_num].axis('off')
            axs[row_num, column_num].set_xticks([], [])
            axs[row_num, column_num].set_yticks([], [])
            imageTitle = image_title_array[image_title_num]
            #axs[row_num, column_num].set_xlabel('%s' % imageTitle, fontsize=10)
            axs[row_num, column_num].set_title('%s' % imageTitle, fontsize=10, fontdict=None, loc='center', color='k')
            image_title_num += 1
            img.close()

        if row_num == num_of_rows_for_rgb_grid - 1:
            numOfEmptyGridsForRow = num_of_cols_for_rgb_grid - len(images_for_row)
            for emptyIndex in range(len(images_for_row), num_of_cols_for_rgb_grid):
                axs[row_num, emptyIndex].axis('off')

        row_num += 1
    fig.tight_layout(pad=2.3)
    fig.suptitle('%s' % figure_title, fontsize = 16)
    fig.savefig(file_path)
    plt.close(fig)


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
    keys = []

    for root, dirs, files in os.walk(base_dir):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value
            # print(key)

            num = int(re.search(r'\d+', key).group())
            numbers.append(num)
            keys.append(key)

    # print(numbers)

    return numbers, keys


def plotprogressNegativePositive(number_iterations):
    """
    This is the plotting of the progress taken, step by step of the negative and positive
    data, placed into a grid for each source. This grid is made for each g, r, and i band,
    seen in each column.
    This also makes a rgb image grid of the positive image and negative images, following
    the amount of images requested in the variable numberIterations

    The grid for the negative data is set where the DES Processed WCSClipped images is seen
    in the first row, the DES Processed normalised images are seen in the second row, and
    the DES Sky images are seen in the third row.

    The grid for the positive data is set where the positively noiseless simulated images
    are in the first row, the positively simulated data with background sky added to it is
    seen in the second row, the normalised positively simulated images are seen in the third
    row.

    Args:
        number_iterations(integer):    This is the number of sources that will have the progress grids made for them.
                                       Not all sources are sometimes, since you just want to check the progress,
                                       and how the images change overtime.
    Returns:
        numbers_train_neg(list):        This is a list of Negative data numbers, which have been created at random.
        The Figure of the progress of the positive and negative data is saved.
        The rgb image grids for the positive and negative data is saved as well.
    """
    # Number of Images creating grids to view.
    rgb_pos_image_paths = []
    rgb_des_image_paths = []
    image_title_array = []
    cols = ['g-band', 'r-band', 'i-band']
    rows_neg = ['Clipped DES images', 'Normalised images']
    rows_pos = ['Simulated Lenses', 'With Background Noise', 'Normalised Simulated Lenses']
    labels_neg = ['\n'.join(wrap(l, 15)) for l in rows_neg]
    labels_pos = ['\n'.join(wrap(l, 15)) for l in rows_pos]

    numbers_train_neg, keys = getNegativeNumbers(train_negative_path)
    for index in range(0, number_iterations):
        num = numbers_train_neg[index]
        key = keys[index]
        gWCS, rWCS, iWCS = getNegativeProcessedWCS(num)
        gDESNorm, rDESNorm, iDESNorm = getNegativeDES(num)
        gDESSky, rDESSky, iDESSky = getDESSky(num)
        gPos, rPos, iPos = getPosNoiseless(num)
        gPosSky, rPosSky, iPosSky = getPosWDESSky(num)
        gPosSkyNorm, rPosSkyNorm, iPosSkyNorm = getPosWDESSkyNorm(num)

        # creating the first grid, in which the DES_Processed images are seen.
        fig1, axs1 = plt.subplots(2, 3, figsize = (10,10))
        fig1.suptitle("Process to form Negative Images")
        fig1.tight_layout(pad=1.0)
        # get column names and row names
        for ax, col in zip(axs1[0], cols):
            ax.set_title(col)
        fig1.subplots_adjust(left=0.2, wspace=0.6)
        fig1.align_ylabels(axs1[:, 0])

        axs1[0, 0].set_ylabel(labels_neg[0], rotation=90, fontsize=10, labelpad=10, va='center')
        axs1[1, 0].set_ylabel(labels_neg[1], rotation=90, fontsize=10, labelpad=10, va='center')
        axs1[0, 0].imshow(gWCS[0].data, cmap='gray')
        axs1[0, 1].imshow(rWCS[0].data, cmap='gray')
        axs1[0, 2].imshow(iWCS[0].data, cmap='gray')
        axs1[1, 0].imshow(gDESNorm[0].data, cmap='gray')
        axs1[1, 1].imshow(rDESNorm[0].data, cmap='gray')
        axs1[1, 2].imshow(iDESNorm[0].data, cmap='gray')

        some_path = '%s/%s/Negative_Processed_Grid.png' % (train_negative_path, key)
        print(some_path)
        fig1.savefig(some_path)
        plt.close(fig1)

        fig3, axs3 = plt.subplots(1, 3, figsize = (8,8))
        fig3.suptitle("DES Background Sky Images", fontsize=14)
        fig3.tight_layout(pad=0.3)

        for ax, col in zip(axs3, cols):
            ax.set_title(col)
            print(col)

        axs3[0].imshow(gDESSky[0].data, cmap='gray')
        axs3[1].imshow(rDESSky[0].data, cmap='gray')
        axs3[2].imshow(iDESSky[0].data, cmap='gray')

        some_path_fig3 = '%s/%s/RandomSky.png' % (train_negative_path, key)
        print(some_path_fig3)
        fig3.savefig(some_path_fig3)
        plt.close(fig3)

        gWCS.close()
        rWCS.close()
        iWCS.close()
        gDESNorm.close()
        rDESNorm.close()
        iDESNorm.close()
        gDESSky.close()
        rDESSky.close()
        iDESSky.close()

        # creating the second grid, in which the Simulated images are seen.
        fig2, axs2 = plt.subplots(3, 3, figsize = (8,8))
        fig2.tight_layout(pad=3.1)
        fig2.subplots_adjust(left=0.2, wspace=0.6)
        fig2.align_ylabels(axs2[:, 0])
        fig2.suptitle("Process to form Positive Images")

        # get column names and row names
        for ax, col in zip(axs2[0], cols):
            ax.set_title(col)

        axs2[0, 0].set_ylabel(labels_pos[0], rotation=90, fontsize=10, labelpad=10, va='center')
        axs2[1, 0].set_ylabel(labels_pos[1], rotation=90, fontsize=10, labelpad=10, va='center')
        axs2[2, 0].set_ylabel(labels_pos[2], rotation=90, fontsize=10, labelpad=10, va='center')

        axs2[0, 0].imshow(gPos[0].data, cmap='gray')
        axs2[0, 1].imshow(rPos[0].data, cmap='gray')
        axs2[0, 2].imshow(iPos[0].data, cmap='gray')
        axs2[1, 0].imshow(gPosSky[0].data, cmap='gray')
        axs2[1, 1].imshow(rPosSky[0].data, cmap='gray')
        axs2[1, 2].imshow(iPosSky[0].data, cmap='gray')
        axs2[2, 0].imshow(gPosSkyNorm[0].data, cmap='gray')
        axs2[2, 1].imshow(rPosSkyNorm[0].data, cmap='gray')
        axs2[2, 2].imshow(iPosSkyNorm[0].data, cmap='gray')

        fig2.savefig('%s/%i/Positive_Processed_Grid.png' % (train_positive_path, num))
        plt.close(fig2)

        # closing images to save RAM
        gPos.close()
        rPos.close()
        iPos.close()
        gPosSky.close()
        rPosSky.close()
        iPosSky.close()
        gPosSkyNorm.close()
        rPosSkyNorm.close()
        iPosSkyNorm.close()

        rgb_pos_image_paths.append('%s/%s/%s_rgb.png' % (train_positive_path, num, num))
        rgb_des_image_paths.append(getNegativeDESRGBPath(num))
        image_title_pos = '%s' % num
        image_title_array.append(image_title_pos)

    file_path3 = '%s_RGB_ImageGrid.png' % train_positive_path
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved),
    # list( paths of rgb images)))
    plotAndSaveRgbGrid(file_path3, rgb_pos_image_paths, image_title_array,
                       figure_title='RGB Grid of the Positively Simulated Lenses')

    # creating the rgb grid for the DES Images
    file_path4 = '%s_RGB_ImageGrid.png' % train_negative_path
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved),
    # list( paths of rgb images)))
    plotAndSaveRgbGrid(file_path4, rgb_des_image_paths, image_title_array, figure_title='RGB Grid of the Negative '
                                                                                        'Images')
    return numbers_train_neg


def plotKnownLenses(number_iterations, known_path=''):
    """
    This is the plotting of knownlenses. This has the requested amount of known lenses to look at and check.
    This function opens the function getKnownRGBPath(), and gets the paths of the rgb images of the known
    lenses from DES2017. These rgb images are used to create a rgb grid of the DES2017 known lenses.
    This rgb grid is often looked at when checking the images.
    Args:
        number_iterations(integer):    This is the number of sources that will have the progress grids made for them.
                                       Not all sources are sometimes, since you just want to check the progress,
                                       and how the images change overtime.
        known_path(string):            This is the path name name that identifies which set of known lenses are used,
                                       either 'Known47' or 'Known84'.
    Returns:
        This saves the figure containing the rgb image grids of the knownlenses.
    """

    random_list = []
    folders = {}
    random_numbers = []

    for root, dirs, files in os.walk(known_path):
        for folder in dirs:
            key = folder
            value = os.path.join(root, folder)
            folders[key] = value
            # print(key)

            num = int(re.search(r'\d+', key).group())
            random_numbers.append(num)

    # print('Numbers: ' + str(random_numbers))
    # print('Numbers Length: ' + str(len(random_numbers)))

    # for i in range(0, 388):
    #     random_num = random.int(0, 388):
    # get random num in range 0_388, append this to a random list.
    # Call this  in for loop below as num = random[i]
    # This retrieve 9 random rgb known lenses.

    rgb_known_image_paths = []
    image_title_array = []
    for i in range(0, number_iterations):
        num = random_numbers[i]
        print('Random Number Selected: ' + str(num))
        rgbKnown, tileName = getKnownRGBPath(num, known_path)
        rgb_known_image_paths.append(rgbKnown)
        imageTitle = '%s_%s' % (num, tileName)
        image_title_array.append(imageTitle)

    file_path5 = '%s_RGB_ImageGrid.png' % known_path
    # plotAndSaveRgbGrid( int(number of Rows), int(number of Columns), str(filename for where RGB will be saved),
    # list( paths of rgb images)))
    if known_path == 'UnseenData/KnownLenses':
        rgb_title = 'RGB Grid of the Unseen Real Lenses'
    elif known_path == 'UnseenData/SelectingSimilarLensesToPositiveSimulated':
        rgb_title = 'RGB Grid of the Selected Unseen Real Lenses'

    plotAndSaveRgbGrid(file_path5, rgb_known_image_paths, image_title_array, rgb_title)

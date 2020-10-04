"""
Make the colour magnitude diagram r vs g-r for the positive lenses
"""
# open the training magnitude table and get the g and r values.
# open the testing magnitude table and get the g and r values.
# And the r and g-r values to an array
# create a colour magnitude diagram

# imports
import csv
import matplotlib.pyplot as plt
import numpy as np

# Global Variables
from matplotlib.ticker import LinearLocator

train_positive = 'Training/g_r_PositiveAll'
test_positive = 'Testing/g_r_PositiveAll'


def getMagnitudeTable(positive_path):
    """
    This is a function to get the values of the Magniutde table
    Args:
        positive_path(string):  This is the file path for the postive data in which the simulated lenses and sources can be seen.
    Returns:
        lens_r_list(list):      This list contains the r magnitudes for the lenses.
        lens_gr_list(list):     This list contains the g-r band magnitudes for the lenses.
        source_r_list(list):    This list contains the r magnitudes of the sources.
        source_gr_list(list):   This list contains the g-r band magnitudes for the sources.
    """
    with open('%s_magnitudesTable.csv' % positive_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        lens_r_list = []
        lens_gr_list = []
        source_r_list = []
        source_gr_list = []

        for line in csvfile.readlines():
            array = line.split(',')

            num = array[0]
            lens_g_mag = array[1]
            lens_r_mag = array[2]
            print(lens_g_mag)
            print(lens_r_mag)
            lens_i_mag = array[3]
            source_g_mag = array[4]
            source_r_mag = array[5]
            source_i_mag = array[6]
            lens_gr = array[7]
            lens_ri = array[8]
            # lens_gi =array ['Lens_gi']
            source_gr = array[9]
            source_ri = array[10]
            # source_gi =array ['Source_gi']

            lens_r_list.append(lens_r_mag)
            # print(type(lens_r_list))
            lens_gr_list.append(lens_gr)
            source_r_list.append(source_r_mag)
            source_gr_list.append(source_gr)

        return lens_r_list, lens_gr_list, source_r_list, source_gr_list


def colourMagnitudeDiagram(lens_r_list, lens_gr_list, source_r_list, source_gr_list, positive_path):
    """
    This function creates a colour magnitude diagram (r vs g-r) for the lenses and sources of the simulated data.
    Args:
        lens_r_list(list):      This list contains the r magnitudes for the lenses.
        lens_gr_list(list):     This list contains the g-r band magnitudes for the lenses.
        source_r_list(list):    This list contains the r magnitudes of the sources.
        source_gr_list(list):   This list contains the g-r band magnitudes for the sources.

    """
    fig = plt.figure()
    lens_r_list = lens_r_list[1:]
    lens_gr_list = lens_gr_list[1:]
    source_r_list = source_r_list[1:]
    source_gr_list = source_gr_list[1:]

    # GETTING THE MIN AND MAX:
    lens_r_min = min(lens_r_list)
    lens_r_max = max(lens_r_list)
    lens_gr_min = min(lens_gr_list)
    lens_gr_max = max(lens_gr_list)
    source_r_min = min(source_r_list)
    source_r_max = max(source_r_list)
    source_gr_min = min(source_gr_list)
    source_gr_max = max(source_gr_list)

    r_min = min(lens_r_min,source_r_min)
    r_max = max(lens_r_max,source_r_max)
    gr_min = min(lens_gr_min, source_gr_min)
    gr_max = max(lens_gr_max,source_gr_max)

    x = lens_gr_list
    print(x)
    y = lens_r_list
    x2 = source_gr_list
    y2 = source_r_list

    plt.scatter(x, y, c='blue', label='Lenses')
    plt.scatter(x2, y2, color='red', label='Sources')
    plt.xlabel('g-r')
    plt.ylabel('r')

    axes = plt.axes()
    # x_min = min(float(gr_min), float(gr_max))
    # print(x_min)
    # # x_max = max(float(gr_min), float(gr_max))
    # print((x_max))
    x_min = float(gr_min)
    x_max = float(gr_max)
    print(x_min)
    print(x_max)

    # y_min = min(float(r_min), float(r_max))
    y_min = float(r_min)
    y_max = float(r_max)
    print(y_min)
    print(y_max)

    # y_max = max(float(r_min), float(r_max))
    # axes.set_xlim(0, 3)
    # axes.set_ylim(16, 22)
    # axes.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    axes.get_yaxis().set_major_locator(LinearLocator(numticks=10))
    axes.get_xaxis().set_major_locator(LinearLocator(numticks=10))

    if positive_path == 'Training/g_r_PositiveAll':
        plt.title('Training Colour Magnitude')
    elif positive_path == 'Testing/g_r_PositiveAll':
        plt.title('Testing Colour Magnitude')
    plt.legend()
    plt.show()
    fig.savefig('%s_colourMagnitudeDiagram.png' % positive_path)


# ______________________________________________________________________________________
# MAIN
#
train_lens_r, train_lens_gr, train_source_r, train_source_gr = getMagnitudeTable(train_positive)
colourMagnitudeDiagram(train_lens_r, train_lens_gr, train_source_r, train_source_gr, positive_path=train_positive)

test_lens_r, test_lens_gr, test_source_r, test_source_gr = getMagnitudeTable(test_positive)
colourMagnitudeDiagram(test_lens_r, test_lens_gr, test_source_r, test_source_gr, positive_path=test_positive)

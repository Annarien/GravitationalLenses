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
            first_item = array[0]
            print(first_item)

            num = array[0]
            lens_g_mag = array[1]
            lens_r_mag = array[2]
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
            print(type(lens_r_list))
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
    fig, ax = plt.subplots()
    lens_r_list = lens_r_list[1:]
    lens_gr_list = lens_gr_list[1:]
    source_r_list = source_r_list[1:]
    source_gr_list = source_gr_list[1:]

    x = lens_gr_list
    y = lens_r_list
    ax.scatter(x, y, c='blue', label='Lenses')
    x2 = source_gr_list
    y2 = source_r_list
    ax.scatter(x2, y2, c='red', label='Sources')

    ax.set_xlabel('g-r')
    ax.set_ylabel('r')

    max_x = max(source_gr_list)
    print(max_x)
    max_y = max(source_r_list)
    print(max_y)

    ax.set_xticks(ax.get_xticks()[::16])
    ax.set_yticks(ax.get_yticks()[::16])

    ax.legend()
    ax.grid(True)

    plt.show()
    fig.savefig('%s_colourMagnitudeDiagram.png' % positive_path)


# ______________________________________________________________________________________
# MAIN
#
train_lens_r, train_lens_gr, train_source_r, train_source_gr = getMagnitudeTable(train_positive)
colourMagnitudeDiagram(train_lens_r, train_lens_gr, train_source_r, train_source_gr, positive_path=train_positive)

test_lens_r, test_lens_gr, test_source_r, test_source_gr = getMagnitudeTable(test_positive)
colourMagnitudeDiagram(test_lens_r, test_lens_gr, test_source_r, test_source_gr, positive_path=test_positive)


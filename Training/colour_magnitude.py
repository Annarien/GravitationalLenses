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
    with open('%s_magnitudesTable.csv' % positive_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        lens_r_array = []
        lens_gr_array = []
        source_r_array = []
        source_gr_array = []

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

            lens_r_array.append(lens_r_mag)
            lens_gr_array.append(lens_gr)
            source_r_array.append(source_r_mag)
            source_gr_array.append(source_gr)

        return lens_r_array, lens_gr_array, source_r_array, source_gr_array


def colourMagnitudeDiagram(lens_r_array, lens_gr_array, source_r_array, source_gr_array):
    # plt.locator_params(axis='x', nbins=20)
    fig, ax = plt.subplots()
    lens_r_array = lens_r_array[1:]
    lens_gr_array = lens_gr_array[1:]
    source_r_array = source_r_array[1:]
    source_gr_array = source_gr_array[1:]

    x = lens_gr_array
    y = lens_r_array
    ax.scatter(x, y, c='black', label='Lenses')
    x2 = source_gr_array
    y2 = source_r_array
    ax.scatter(x2, y2, c='red', label='Sources')

    ax.set_xlabel('g-r')
    ax.set_ylabel('r')

    max_x = max(source_gr_array)
    print(max_x)
    max_y = max(source_r_array)
    print(max_y)

    ax.set_xticks(ax.get_xticks()[::16])
    ax.set_yticks(ax.get_yticks()[::16])

    # plt.xticks(np.arange(0, 50,5))
    # plt.yticks(np.arange(0, 20, 10))

    ax.legend()
    ax.grid(True)

    plt.show()

# ______________________________________________________________________________________
# MAIN
#
lens_r_array, lens_gr_array, source_r_array, source_gr_array = getMagnitudeTable(train_positive)
colourMagnitudeDiagram(lens_r_array, lens_gr_array, source_r_array, source_gr_array)

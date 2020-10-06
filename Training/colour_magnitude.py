"""
Make the colour magnitude diagram r vs g-r for the positive lenses
"""
# open the training magnitude table and get the g and r values.
# open the testing magnitude table and get the g and r values.
# And the r and g-r values to an array
# create a colour magnitude diagram

# imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

# Global Variables
g_r_train_positive = 'Training/g_r_PositiveAll'
g_r_test_positive = 'Testing/g_r_PositiveAll'


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
        lens_gi_list = []
        source_r_list = []
        source_gr_list = []
        source_gi_list = []

        for line in csvfile.readlines():
            array = line.split(',')

            num = array[0]
            lens_g_mag = array[1]
            lens_r_mag = array[2]
            lens_i_mag = array[3]
            source_g_mag = array[4]
            source_r_mag = array[5]
            source_i_mag = array[6]
            lens_gr = array[7]
            lens_ri = array[8]
            lens_gi = array[9]
            source_gr = array[10]
            source_ri = array[11]
            source_gi = array[12]

            lens_r_list.append(lens_r_mag)
            lens_gr_list.append(lens_gr)
            lens_gi_list.append(lens_gi)
            source_r_list.append(source_r_mag)
            source_gr_list.append(source_gr)
            source_gi_list.append(source_gi)

        lens_r_list = lens_r_list[1:]
        lens_gr_list = lens_gr_list[1:]
        lens_gi_list = lens_gi_list[1:]
        source_r_list = source_r_list[1:]
        source_gr_list = source_gr_list[1:]
        source_gi_list = source_gi_list[1:]

        return lens_r_list, lens_gr_list, lens_gi_list, source_r_list, source_gr_list, source_gi_list


def colourMagnitudeDiagram(lens_r_list, lens_gr_list, lens_gi_list, source_r_list, source_gr_list,
                           source_gi_list, positive_path):
    """
    This function creates a colour magnitude diagram (r vs g-r) for the lenses and sources of the simulated data.
    Args:
        lens_r_list(list):      This list contains the r magnitudes for the lenses.
        lens_gr_list(list):     This list contains the g-r band magnitudes for the lenses.
        source_r_list(list):    This list contains the r magnitudes of the sources.
        source_gr_list(list):   This list contains the g-r band magnitudes for the sources.
    Saves:
        This saves the a colour magnitude plots of r vs g-r, for the simulated lenses and sources used in the
        positively simulated data. The x axes limits of the graph are set the min(g-r) -0.2 and max(g-r)+0.2.
         The y axes limits of the graph are set to min(r)-0.2 and max(r)+0.2

    """

    int_lens_r_list = []
    int_lens_gr_list = []
    int_lens_gi_list = []
    int_source_r_list = []
    int_source_gr_list = []
    int_source_gi_list = []

    # creating lists containing floats and integrs that can be used in the graph
    for i in range(0, len(lens_r_list)):
        int_lens_r = float(lens_r_list[i])
        int_lens_r_list.append(int_lens_r)
        int_lens_gr = float(lens_gr_list[i])
        int_lens_gr_list.append(int_lens_gr)
        int_lens_gi = float(lens_gi_list[i])
        int_lens_gi_list.append(int_lens_gi)
        int_source_r = float(source_r_list[i])
        int_source_r_list.append(int_source_r)
        int_source_gr = float(source_gr_list[i])
        int_source_gr_list.append(int_source_gr)
        int_source_gi = float(source_gi_list[i])
        int_source_gi_list.append(int_source_gi)

    # Getting the min and max fro the axees
    lens_r_min = min(int_lens_r_list)
    lens_r_max = max(int_lens_r_list)
    lens_gr_min = min(int_lens_gr_list)
    lens_gr_max = max(int_lens_gr_list)
    lens_gi_min = min(int_lens_gi_list)
    lens_gi_max = max(int_lens_gi_list)
    source_r_min = min(int_source_r_list)
    source_r_max = max(int_source_r_list)
    source_gr_min = min(int_source_gr_list)
    source_gr_max = max(int_source_gr_list)
    source_gi_min = min(int_source_gi_list)
    source_gi_max = max(int_source_gi_list)

    r_min = min(lens_r_min, source_r_min)
    r_max = max(lens_r_max, source_r_max)
    gr_min = min(lens_gr_min, source_gr_min)
    gr_max = max(lens_gr_max, source_gr_max)
    gi_min = min(lens_gi_min, source_gi_min)
    gi_max = max(lens_gi_max, source_gi_max)

    y_min = gr_min - 0.2
    y_max = gr_max + 0.2
    x_min = r_min - 0.2
    x_max = r_max + 0.2
    a_min = gi_min -0.2
    a_max = gi_max+0.2

    x = int_lens_r_list
    y = int_lens_gr_list
    x2 = int_source_r_list
    y2 = int_source_gr_list
    gi_source = int_source_gi_list
    gi_lens = int_lens_gi_list

    # plotting the colour magnitude graph
    fig1 = plt.figure()
    plt.scatter(x, y, color='b', marker='.', label='Lenses')
    plt.scatter(x2, y2, color='r', marker='.', label='Sources')
    plt.xlabel('r')
    plt.ylabel('g-r')

    axes = plt.axes()
    axes.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    axes.get_yaxis().set_major_locator(LinearLocator(numticks=12))
    axes.get_xaxis().set_major_locator(LinearLocator(numticks=12))

    if positive_path == 'Training/g_r_PositiveAll':
        plt.title('Positively Simulated Training Data Colour Magnitude (g-r vs r) Diagram')
    elif positive_path == 'Testing/g_r_PositiveAll':
        plt.title('Positively Simulated Testing Data Colour Magnitude (g-r vs r) Diagram')
    elif positive_path == 'Training/g_r_AllData':
        plt.title('All Positively Simulated Data Colour Magnitude (g-r vs r) Diagram')
    plt.legend()
    plt.show()
    fig1.savefig('%s_g_r_colourMagnitudeDiagram.png' % positive_path)

    fig2 = plt.figure()
    plt.scatter(gi_lens, y, color = 'b', marker= '.', label = 'Lenses')
    plt.scatter(gi_source, y2, color ='r', marker='.', label = 'Sources')
    if positive_path == 'Training/g_r_PositiveAll':
        plt.title('Positively Simulated Training Data Colour Magnitude (g-r vs g-i ) Diagram')
    elif positive_path == 'Testing/g_r_PositiveAll':
        plt.title('Positively Simulated Testing Data Colour Magnitude (g-r vs g-i) Diagram')
    elif positive_path == 'Training/g_r_AllData':
        plt.title('All Positively Simulated Data Colour Magnitude (g-r vs g-i) Diagram')

    axes = plt.axes()
    axes.set(xlim=(a_min, a_max), ylim=(y_min, y_max))
    axes.get_yaxis().set_major_locator(LinearLocator(numticks=12))
    axes.get_xaxis().set_major_locator(LinearLocator(numticks=12))

    plt.xlabel('g-i')
    plt.ylabel('g-r')
    plt.legend()
    plt.show()
    fig2.savefig('%s_g_i_colourMagnitudeDiagram.png' % positive_path)
# ______________________________________________________________________________________
# MAIN
#
train_lens_r, train_lens_gr, train_lens_gi,train_source_r, train_source_gr, train_source_gi = \
    getMagnitudeTable(g_r_train_positive)
colourMagnitudeDiagram(train_lens_r, train_lens_gr, train_lens_gi, train_source_r, train_source_gr,
                       train_source_gi,positive_path=g_r_train_positive)

test_lens_r, test_lens_gr, test_lens_gi, test_source_r, test_source_gr, test_source_gi = \
    getMagnitudeTable(g_r_test_positive)
colourMagnitudeDiagram(test_lens_r, test_lens_gr, test_lens_gi, test_source_r, test_source_gr,
                       test_source_gi, positive_path=g_r_test_positive)

all_lens_r = train_lens_r + test_lens_r
all_lens_gr = train_lens_gr + test_lens_gr
all_lens_gi = train_lens_gi + test_lens_gi
all_source_r = train_source_r + test_source_r
all_source_gr = train_source_gr + test_source_gr
all_source_gi = train_source_gi + test_source_gi
colourMagnitudeDiagram(all_lens_r, all_lens_gr,all_lens_gi, all_source_r, all_source_gr, all_source_gi, positive_path='Training/g_r_AllData')

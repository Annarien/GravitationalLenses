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


# Global Variables
train_positive = 'Training/g_r_PositiveAll'
test_positive = 'Testing/g_r_PositiveAll'


def getMagnitudeTable(positive_path):
    with open('%s_magnitudesTable.csv' % positive_path, 'r') as file:
        reader = csv.reader(file)
        lens_r_array = []
        lens_gr_array = []
        source_r_array = []
        source_gr_array = []
        for row in reader:
            num = row['Index']
            lens_g_mag = row['Lens_g_mag']
            lens_r_mag = row['Lens_r_mag']
            lens_i_mag = row['Lens_i_mag']
            source_g_mag = row['Source_g_mag']
            source_r_mag = row['Source_r_mag']
            source_i_mag = row['Source_i_mag']
            lens_gr = row['Lens_gr']
            lens_ri = row['Lens_ri']
            lens_gi = row['Lens_gi']
            source_gr = row['Source_gr']
            source_ri = row['Source_ri']
            source_gi = row['Source_gi']
            lens_r_array = lens_r_array.append(lens_r_mag)
            lens_gr_array = lens_gr_array.append(lens_gr)
            source_r_array = source_r_array.append(source_r_mag)
            source_gr_array=source_gr_array.append(source_gr)

            return lens_r_array,lens_gr_array,source_r_array,source_gr_array

def colourMagnitudeDiagram(lens_r_array, lens_gr_array, source_r_array, source_gr_array):

    for i in range(0,len(lens_r_array)):
        x = lens_gr_array[i]
        y = lens_r_array[i]
        plt.scatterplot(x, y, 'o', color='blue')
        x2 = source_gr_array[i]
        y2 = source_r_array[i]
        plt.scatter(x2, y2, 'o', color='red')
    plt.xlabel('g-r')
    plt.ylabel('r')

# ______________________________________________________________________________________
# MAIN
#
lens_r_array, lens_gr_array, source_r_array, source_gr_array = getMagnitudeTable(train_positive)
colourMagnitudeDiagram(lens_r_array, lens_gr_array, source_r_array, source_gr_array)



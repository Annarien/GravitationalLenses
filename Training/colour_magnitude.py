"""
Make the colour magnitude diagram r vs g-r for the positive lenses
"""
# open the training magnitude table and get the g and r values.
# open the testing magnitude table and get the g and r values.
# And the r and g-r values to an array
# create a colour magnitude diagram

# imports
import csv

# Global Variables
test_positive = 'Testing/g_r_PositiveAll'
train_positive = 'Training/g_r_PositiveAll'


def getTrainingMagnitudeTable(positive_path):
    with open('%s_magnitudesTable.csv' % positive_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)


#______________________________________________________________________________________
# MAIN

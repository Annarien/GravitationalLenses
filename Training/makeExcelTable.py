"""
This makes an excel spreadsheet to make the the results from ml_Lenses more readible, instead of writing it in the terminal.
"""
import astropy.table  as atpy
import numpy as np
import csv
from csv import writer

def makeInitialTable():
    
    tab = atpy.Table()
    tab.add_column(atpy.Column(np.zeros(1), "Description"))
    tab.add_column(atpy.Column(np.zeros(1), "imageTrain_std"))
    tab.add_column(atpy.Column(np.zeros(1), "imageTrain_mean"))
    tab.add_column(atpy.Column(np.zeros(1), "imageTrain_shape"))
    tab.add_column(atpy.Column(np.zeros(1), "imageLabels_shape"))
    tab.add_column(atpy.Column(np.zeros(1), "train_percent"))
    tab.add_column(atpy.Column(np.zeros(1), "test_percent"))
    tab.add_column(atpy.Column(np.zeros(1), "xTrain_shape"))
    tab.add_column(atpy.Column(np.zeros(1), "xTest_shape"))
    tab.add_column(atpy.Column(np.zeros(1), "yTrain_shape"))
    tab.add_column(atpy.Column(np.zeros(1), "yTest_shape"))
    tab.add_column(atpy.Column(np.zeros(1), "n_splits"))
    tab.add_column(atpy.Column(np.zeros(1), "random_state"))
    tab.add_column(atpy.Column(np.zeros(1), "AccuracyScore"))
    tab.add_column(atpy.Column(np.zeros(1), "KFoldAccuracy"))
    tab.add_column(atpy.Column(np.zeros(1), "AccuracyScore_47"))
    tab.add_column(atpy.Column(np.zeros(1), "KFoldAccuracy_47"))
    tab.add_column(atpy.Column(np.zeros(1), "AccuracyScore_84"))
    tab.add_column(atpy.Column(np.zeros(1), "KFoldAccuracy_84"))
    tab.add_column(atpy.Column(np.zeros(1), "AccuracyScore_131"))
    tab.add_column(atpy.Column(np.zeros(1), "KFoldAccuracy_131"))
    tab.write('../Results/ml_Lenses_results.csv', overwrite = True)
    return (tab)

def appendRowAsList(filename, elementList):
    with open(filename, 'a+') as writeObj:
        csvWriter = writer(writeObj)
        csvWriter.writerow(elementList)
        

def getElementList(description, imageTrain_std, imageTrain_mean, imageTrain_shape, imageLabels_shape, train_percent, test_percent, xTrain_shape, xTest_shape, yTrain_shape, yTest_shape, n_splits, random_state, AccuracyScore, KFoldAccuracy, AccuracyScore_47, KFoldAccuracy_47, AccuracyScore_84, KFoldAccuracy_84, AccuracyScore_131, KFoldAccuracy_131):
    elementList = [description, imageTrain_std, imageTrain_mean, imageTrain_shape, imageLabels_shape, train_percent, test_percent, xTrain_shape, xTest_shape, yTrain_shape, yTest_shape, n_splits, random_state, AccuracyScore, KFoldAccuracy, AccuracyScore_47, KFoldAccuracy_47, AccuracyScore_84, KFoldAccuracy_84, AccuracyScore_131, KFoldAccuracy_131]
    return (elementList)

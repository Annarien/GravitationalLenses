"""This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the
gravitational lenses. """
# IMPORTS
from cnnUtils import getPositiveSimulatedTrain, getNegativeDESTrain, getPositiveSimulatedTest, getNegativeDESTest, \
    makeTrainTest, useKerasModel, plotModel

print("PAST IMPORTS")

# _________________________________________________________________________________________________________________________
# MAIN

# Get training data
positive_train = getPositiveSimulatedTrain()
negative_train = getNegativeDESTrain()

# Get Testing data
positive_test = getPositiveSimulatedTest()
negative_test = getNegativeDESTest()

# make Train Test
x_train, \
 x_test, \
 y_train, \
 y_test, \
 train_percent, \
 test_percent, \
 image_train_std, \
 image_train_mean, \
 image_train_shape, \
 image_labels_shape, \
 x_train_shape, \
 x_test_shape, \
 y_train_shape, \
 y_test_shape = makeTrainTest(positive_train, negative_train, positive_test, negative_test)

# use the Keras model
seq_model, model, accuracy_score = useKerasModel(x_train, x_test, y_train, y_test)
plotModel(seq_model)

print("DONE")

# visualizeKeras(model)

# ______________________________________________________________________________________________________________________
# n_splits, random_state, k_fold_accuracy, k_fold_std, neural_network = getKerasKFold(x_train, x_test, yTrain, y_test)
#
# # calculating the amount of things accurately identified
# # looking at Known131
# # 1 = gravitational lens
# # 0 = negative lens
#
# #
# #_____________________________________________________________________________________________________________________
# known_des_2017, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47 =
# testUnseenDES2017(model, neural_network, n_splits)
# known_jacobs, accuracy_score_84, k_fold_accuracy_84, k_fold_std_84 =
# testUnseenJacobs(model, neural_network, n_splits)
# accuracy_score_131, k_fold_accuracy_131, k_fold_std_131 =
# testUnseenDES2017AndJacobs(known_des_2017, known_jacobs, model, neural_network, n_splits)

 # write to ml_Lenses_results.xlsx
# makeExcelTable.makeInitialTable()
# element_list = makeExcelTable.getElementList(description, image_train_std, image_train_mean, image_train_shape,
# image_labels_shape, train_percent, test_percent, x_train_shape, x_test_shape, y_train_shape, y_test_shape, n_splits,
# random_state, accuracy_score, k_fold_accuracy, k_fold_std, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47,
# accuracy_score_84, k_fold_accuracy_84, k_fold_std_84, accuracy_score_131, k_fold_accuracy_131, k_fold_std_131)
# file_name = '../Results/ml_Lenses_results.csv' makeExcelTable.appendRowAsList(file_name, element_list)
